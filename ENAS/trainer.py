"""用于训练 ENAS 的模块。"""

import contextlib
import math

import numpy as np
import scipy.signal
from ENAS.tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel

import models  # 导入自定义的模型模块
from ENAS import utils

# 初始化日志记录器，用于在训练过程中输出信息
logger = utils.get_logger()

def _apply_penalties(extra_out, args):
    """根据 `args` 的设置，选择性地添加正则化惩罚项。
    这些惩罚项用于激活正则化、时间激活正则化和/或隐藏状态范数稳定。

    参数：
        extra_out (dict): 包含模型的中间输出：
            - 'dropped': 经过 Dropout 后的激活值。
            - 'hiddens': 一个批次序列的所有隐藏状态。
            - 'raw': 经过 Dropout 前的激活值。

    返回：
        penalty (float): 总的正则化惩罚项，将被添加到损失函数中。

    参考文献：
        - Regularizing and Optimizing LSTM Language Models (Merity et al., 2017)
        - Regularizing RNNs by Stabilizing Activations (Krueger & Memsevic, 2016)
    """
    penalty = 0  # 初始化惩罚项为 0

    # 激活正则化
    if args.activation_regularization:
        # 惩罚激活值的 L2 范数，抑制过大的激活值
        penalty += (args.activation_regularization_amount *
                    extra_out['dropped'].pow(2).mean())

    # 时间激活正则化（抑制激活值的快速变化）
    if args.temporal_activation_regularization:
        raw = extra_out['raw']
        # 惩罚相邻时间步激活值的差异，鼓励平滑的激活序列
        penalty += (args.temporal_activation_regularization_amount *
                    (raw[1:] - raw[:-1]).pow(2).mean())

    # 范数稳定器正则化
    if args.norm_stabilizer_regularization:
        # 惩罚隐藏状态的范数偏离固定点的程度，稳定隐藏状态的数值范围
        penalty += (args.norm_stabilizer_regularization_amount *
                    (extra_out['hiddens'].norm(dim=-1) -
                     args.norm_stabilizer_fixed_point).pow(2).mean())

    return penalty  # 返回总的正则化惩罚项

def discount(x, amount):
    """对奖励序列应用折扣因子。

    参数：
        x (array-like): 奖励序列。
        amount (float): 折扣因子。

    返回：
        array-like: 折扣后的奖励序列。
    """
    # 使用线性滤波器计算折扣奖励
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def _get_optimizer(name):
    """根据名称返回对应的优化器类。

    参数：
        name (str): 优化器名称，例如 'sgd' 或 'adam'。

    返回：
        optim (Optimizer): 对应的 PyTorch 优化器类。
    """
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam
    return optim

def _get_no_grad_ctx_mgr():
    """根据 PyTorch 版本，返回适当的 `torch.no_grad` 上下文管理器。

    返回：
        context manager: 用于禁用梯度计算的上下文管理器。
    """
    if float(torch.__version__[0:3]) >= 0.4:
        # 对于 PyTorch 版本 >= 0.4，使用 torch.no_grad()
        return torch.no_grad()
    # 对于较早的版本，使用一个空的上下文管理器
    return contextlib.suppress()

def _check_abs_max_grad(abs_max_grad, model):
    """检查模型中新的最大梯度，以跟踪梯度爆炸。

    参数：
        abs_max_grad (float): 当前的最大绝对梯度值。
        model (nn.Module): 要检查的模型。

    返回：
        float: 更新后的最大绝对梯度值。
    """
    # 获取所有非 None 的梯度张量
    finite_grads = [p.grad.data for p in model.parameters() if p.grad is not None]
    # 找到梯度中的最大值和最小值
    new_max_grad = max([grad.max() for grad in finite_grads])
    new_min_grad = min([grad.min() for grad in finite_grads])
    # 计算新的最大绝对梯度值
    new_abs_max_grad = max(new_max_grad, abs(new_min_grad))
    # 如果新的最大绝对梯度值更大，更新并记录
    if new_abs_max_grad > abs_max_grad:
        logger.info(f'abs max grad {abs_max_grad}')
        return new_abs_max_grad
    return abs_max_grad  # 返回更新后的最大绝对梯度值

class Trainer(object):
    """封装训练代码的类。"""
    def __init__(self, args, dataset):
        """训练算法的构造函数。

        参数：
            args (Namespace): 从命令行解析的参数。
            dataset: 包含训练、验证和测试数据集的对象。

        初始化：
            - 数据加载器：用于训练、验证和测试集。
            - 模型：共享模型和控制器模型。
            - 优化器：用于更新共享模型和控制器参数的优化器。
            - 损失函数：用于训练共享模型的交叉熵损失。
        """
        self.args = args
        self.controller_step = 0  # 控制器训练步数计数器
        self.cuda = args.cuda  # 是否使用 CUDA（GPU）
        self.dataset = dataset  # 数据集对象
        self.epoch = 0  # 训练轮数计数器
        self.shared_step = 0  # 共享模型训练步数计数器
        self.start_epoch = 0  # 起始轮数（用于从检查点加载时）

        # 记录正在使用的正则化技术
        logger.info('regularizing:')
        for regularizer in [('activation regularization',
                             self.args.activation_regularization),
                            ('temporal activation regularization',
                             self.args.temporal_activation_regularization),
                            ('norm stabilizer regularization',
                             self.args.norm_stabilizer_regularization)]:
            if regularizer[1]:
                logger.info(f'{regularizer[0]}')

        # 对训练数据进行批处理
        self.train_data = utils.batchify(dataset.train,
                                         args.batch_size,
                                         self.cuda)
        # 对验证数据进行批处理，用于计算控制器训练中的奖励
        self.valid_data = utils.batchify(dataset.valid,
                                         args.batch_size,
                                         self.cuda)
        # 对验证数据进行批处理，用于评估困惑度（batch_size = 1）
        self.eval_data = utils.batchify(dataset.valid,
                                        args.test_batch_size,
                                        self.cuda)
        # 对测试数据进行批处理
        self.test_data = utils.batchify(dataset.test,
                                        args.test_batch_size,
                                        self.cuda)

        self.max_length = self.args.shared_rnn_max_length  # 最大序列长度

        # 初始化 TensorBoard 日志记录器（如果启用）
        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None

        self.build_model()  # 构建共享模型和控制器模型

        # 如果提供了加载路径，则加载预训练的模型
        if self.args.load_path:
            self.load_model()

        # 获取共享模型和控制器的优化器类
        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)

        # 初始化优化器，设置参数和超参数
        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            lr=self.shared_lr,
            weight_decay=self.args.shared_l2_reg)

        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()  # 用于共享模型训练的损失函数

    def build_model(self):
        """创建并初始化共享模型和控制器模型。"""
        if self.args.network_type == 'rnn':
            # 创建基于 RNN 的共享模型
            self.shared = models.RNN(self.args, self.dataset)
        elif self.args.network_type == 'cnn':
            # 创建基于 CNN 的共享模型
            self.shared = models.CNN(self.args, self.dataset)
        else:
            raise NotImplementedError(f'未定义的网络类型 `{self.args.network_type}`')

        # 创建控制器模型
        self.controller = models.Controller(self.args)

        # 如果启用了 CUDA，将模型移动到 GPU
        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            # 尚未实现多 GPU 支持
            raise NotImplementedError('`num_gpu > 1` 的支持正在开发中')

    def train(self, single=False):
        """主训练循环，交替训练共享参数和控制器参数，如 ENAS 论文中所述。

        参数：
            single (bool): 如果为 True，则不会训练控制器，并使用相同的 DAG 而不是派生新架构。
        """
        # 如果在单一模式下运行，加载固定的 DAG
        dag = utils.load_dag(self.args) if single else None

        # 初始训练步骤（如果指定）
        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        # 主训练循环，遍历多个轮次
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. 训练子模型的共享参数 omega
            self.train_shared(dag=dag)

            # 2. 训练控制器参数 theta
            if not single:
                self.train_controller()

            # 定期保存模型和评估
            if self.epoch % self.args.save_epoch == 0:
                with _get_no_grad_ctx_mgr():
                    best_dag = dag if dag else self.derive()
                    self.evaluate(self.eval_data,
                                  best_dag,
                                  'val_best',
                                  max_num=self.args.batch_size * 100)
                self.save_model()

            # 在指定的轮次后，更新共享模型的学习率
            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

    def get_loss(self, inputs, targets, hidden, dags):
        """使用采样的架构计算共享模型的损失。

        参数：
            inputs (Tensor): 输入数据批次。
            targets (Tensor): 目标标签批次。
            hidden (Tensor): 模型的隐藏状态。
            dags (list 或 dict): 从控制器采样的架构（DAGs）。

        返回：
            loss (Tensor): 计算的损失。
            hidden (Tensor): 更新后的隐藏状态。
            extra_out (dict): 模型的额外输出。
        """
        if not isinstance(dags, list):
            dags = [dags]  # 确保 dags 是一个列表

        loss = 0  # 初始化损失
        for dag in dags:
            # 使用给定的 DAG 通过共享模型进行前向传播
            output, hidden, extra_out = self.shared(inputs, dag, hidden=hidden)
            output_flat = output.view(-1, self.dataset.num_tokens)  # 展平输出
            # 计算交叉熵损失，并根据样本数量进行平均
            sample_loss = (self.ce(output_flat, targets) /
                           self.args.shared_num_sample)
            loss += sample_loss  # 累加损失

        # 确保只返回一个隐藏状态（因为只使用了一个 DAG）
        assert len(dags) == 1, '多个 DAG 对应多个隐藏状态'
        return loss, hidden, extra_out  # 返回损失、隐藏状态和额外输出

    def train_shared(self, max_step=None, dag=None):
        """训练共享模型参数。

        参数：
            max_step (int): 最大训练步数（用于预热）。
            dag (dict): 固定的架构，如果提供则使用它而不是从控制器采样。
        """
        model = self.shared  # 引用共享模型
        model.train()  # 设置模型为训练模式
        self.controller.eval()  # 设置控制器为评估模式

        hidden = self.shared.init_hidden(self.args.batch_size)  # 初始化隐藏状态

        # 确定训练的最大步数
        if max_step is None:
            max_step = self.args.shared_max_step
        else:
            max_step = min(self.args.shared_max_step, max_step)

        # 用于跟踪最大梯度和隐藏状态范数的变量
        abs_max_grad = 0
        abs_max_hidden_norm = 0
        step = 0  # 训练步数计数器
        raw_total_loss = 0  # 原始损失累加器
        total_loss = 0  # 总损失（包括惩罚项）累加器
        train_idx = 0  # 训练数据中的索引

        # 训练循环
        while train_idx < self.train_data.size(0) - 1 - 1:
            if step > max_step:
                break  # 如果达到最大步数，退出循环

            # 从控制器采样架构，或使用固定的 DAG
            dags = dag if dag else self.controller.sample(
                self.args.shared_num_sample)
            # 获取一批数据
            inputs, targets = self.get_batch(self.train_data,
                                             train_idx,
                                             self.max_length)

            # 计算损失，获取更新的隐藏状态和额外输出
            loss, hidden, extra_out = self.get_loss(inputs,
                                                    targets,
                                                    hidden,
                                                    dags)
            hidden.detach_()  # 分离隐藏状态，防止反向传播穿过时间
            raw_total_loss += loss.data  # 累加原始损失

            # 将正则化惩罚项添加到损失中
            loss += _apply_penalties(extra_out, self.args)

            # 更新模型参数
            self.shared_optim.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播

            # 监控隐藏状态范数的最大值
            h1tohT = extra_out['hiddens']
            new_abs_max_hidden_norm = utils.to_item(
                h1tohT.norm(dim=-1).data.max())
            if new_abs_max_hidden_norm > abs_max_hidden_norm:
                abs_max_hidden_norm = new_abs_max_hidden_norm
                logger.info(f'max hidden {abs_max_hidden_norm}')
            # 检查梯度爆炸
            abs_max_grad = _check_abs_max_grad(abs_max_grad, model)
            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          self.args.shared_grad_clip)
            self.shared_optim.step()  # 更新参数

            total_loss += loss.data  # 累加总损失

            # 定期记录训练进度
            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_shared_train(total_loss, raw_total_loss)
                raw_total_loss = 0  # 重置原始损失累加器
                total_loss = 0  # 重置总损失累加器

            step += 1  # 增加步数计数器
            self.shared_step += 1  # 增加共享模型步数计数器
            train_idx += self.max_length  # 移动到下一个批次

    def get_reward(self, dag, entropies, hidden, valid_idx=0):
        """计算采样架构的奖励（困惑度的倒数）。

        参数：
            dag (dict): 从控制器采样的架构（DAG）。
            entropies (numpy.ndarray): 控制器决策的熵值。
            hidden (Tensor): 共享模型的隐藏状态。
            valid_idx (int): 验证数据中的索引。

        返回：
            rewards (numpy.ndarray): 计算的奖励。
            hidden (Tensor): 更新后的隐藏状态。
        """
        if not isinstance(entropies, np.ndarray):
            # 如果不是 numpy 数组，将熵值转换为 numpy 数组
            entropies = entropies.data.cpu().numpy()

        # 获取一批验证数据
        inputs, targets = self.get_batch(self.valid_data,
                                         valid_idx,
                                         self.max_length,
                                         volatile=True)
        # 在验证数据上计算损失
        valid_loss, hidden, _ = self.get_loss(inputs, targets, hidden, dag)
        valid_loss = utils.to_item(valid_loss.data)  # 获取标量值

        valid_ppl = math.exp(valid_loss)  # 计算困惑度

        # 根据困惑度计算奖励
        if self.args.ppl_square:
            # 使用平方困惑度计算奖励
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        # 根据熵调整奖励（如果指定）
        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'未知的熵模式：{self.args.entropy_mode}')

        return rewards, hidden  # 返回奖励和更新后的隐藏状态

    def train_controller(self):
        """使用 REINFORCE 算法训练控制器参数。

        控制器使用策略梯度进行更新，旨在最大化期望奖励，
        奖励基于采样架构在验证数据上的性能。
        """
        model = self.controller  # 引用控制器模型
        model.train()  # 设置控制器为训练模式
        # self.shared.eval()  # 不要在此处将共享模型设置为评估模式

        avg_reward_base = None  # 初始化平均奖励基线
        baseline = None  # 初始化移动平均基线
        adv_history = []  # 优势函数历史
        entropy_history = []  # 熵值历史
        reward_history = []  # 奖励历史

        hidden = self.shared.init_hidden(self.args.batch_size)  # 初始化隐藏状态
        total_loss = 0  # 总损失累加器
        valid_idx = 0  # 验证数据中的索引

        # 控制器的训练循环
        for step in range(self.args.controller_max_step):
            # 从控制器采样架构
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)

            # 计算采样架构的奖励
            np_entropies = entropies.data.cpu().numpy()
            with _get_no_grad_ctx_mgr():
                rewards, hidden = self.get_reward(dags,
                                                  np_entropies,
                                                  hidden,
                                                  valid_idx)

            # 对奖励应用折扣因子（如果指定）
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            # 记录奖励和熵值
            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # 更新移动平均基线
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            # 计算优势函数（奖励减去基线）
            adv = rewards - baseline
            adv_history.extend(adv)

            # 计算策略损失（负对数似然乘以优势）
            loss = -log_probs * utils.get_variable(adv,
                                                   self.cuda,
                                                   requires_grad=False)
            # 添加熵正则化（如果指定）
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # 在批次上求和

            # 更新控制器参数
            self.controller_optim.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播

            # 应用梯度裁剪
            if self.args.controller_grad_clip > 0:
                torch
