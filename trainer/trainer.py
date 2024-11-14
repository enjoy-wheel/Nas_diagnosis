import math
import os
import time

import numpy as np
import scipy.signal
import torch
import visdom
from munch import Munch
from tensorboard.program import TensorBoard
from torch import nn

from torchvision.utils import make_grid

import utils
from base import BaseTrainer
from utils import inf_loop, MetricTracker, AverageMeter

class EnasTrainer(BaseTrainer):
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device, logger,
                 data_loader, valid_data_loader=None, lr_schedulers=None, len_epoch=None):
        # 保存模型
        if isinstance(models, tuple) and len(models) == 2:
            self.shared_model, self.controller_model = models
        else:
            self.shared_model = models
            self.controller_model = None
        # 保存优化器
        if isinstance(optimizers, tuple) and len(optimizers) == 2:
            self.shared_optimizer, self.controller_optimizer = optimizers
        else:
            self.shared_optimizer = optimizers
            self.controller_optimizer = None
        # 保存学习率调度器
        if isinstance(lr_schedulers, tuple) and len(lr_schedulers) == 2:
            self.shared_lr_scheduler, self.controller_lr_scheduler = lr_schedulers
        else:
            self.shared_lr_scheduler = lr_schedulers
            self.controller_lr_scheduler = None
        # 保存配置对象和设备信息
        self.config = Munch(config.config).trainer
        self.device = device
        self.logger = logger
        # 保存训练数据的 DataLoader
        self.data_loader = data_loader
        self.epoch = 0
        self.shared_step = 0
        self.controller_step = 0
        # 调用父类的初始化函数，并传入模型、损失函数、评估指标、优化器和配置对象
        super().__init__(self.shared_model, criterion, metric_ftns, self.shared_optimizer, config)
        # 如果没有传入 len_epoch，使用基于 epoch 的训练
        if len_epoch is None:
            # epoch-based training epoch 训练模式
            # len_epoch 为训练数据加载器的长度（即一个 epoch 中的批次数）
            # 即一个epoch训练完数据集所有数据
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            # 基于迭代的训练模式
            # 使用无限循环的数据加载器 (inf_loop) 来代替标准的数据加载器
            # 每个epoch不训练完数据集所有数据，适用于加速训练
            # 例如有1000个数据，一个batch_size=10,那么一个数据集就有100个批次，如果只想训练20个批次，则可以用这个模式
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        # 保存验证数据的 DataLoader（如果有的话）
        self.valid_data_loader = valid_data_loader
        # 判断是否进行验证，只有在提供验证数据加载器时才进行验证
        self.do_validation = self.valid_data_loader is not None
        # 设置每个日志记录步骤的间隔（基于批次大小的平方根）
        self.log_step = int(np.sqrt(data_loader.batch_size))
        # 初始化用于跟踪训练指标的 MetricTracker
        # 包括 'loss' 和所有传入的评估函数，使用 writer 来记录这些指标
        # *[m.__name__ for m in self.metric_ftns] 生成了一个包含所有评估函数名称的列表，并将其解包为独立的参数传递给 MetricTracker 类
        # 目的是方便地将多个评估指标（函数名）作为参数传递
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # 初始化用于跟踪验证指标的 MetricTracker
        # 包括 'loss' 和所有传入的评估函数，使用 writer 来记录这些指标
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        if self.config.trainer.Misc.use_tensorboard:
            self.tb = TensorBoard(self.config.trainer.save_dir)
        else:
            self.tb = None
        self.ce = nn.CrossEntropyLoss()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        一个 epoch 的训练逻辑
        :param epoch: Integer, current training epoch.整数类型，表示当前训练的 epoch（轮次）
        :return: A log that contains average loss and metric in this epoch.一个日志，包含该 epoch 中的平均损失和评估指标
        """
        # 将模型设置为训练模式
        self.model.train()
        # 重置训练指标，以确保每个 epoch 开始时指标从零开始
        self.train_metrics.reset()
        # 遍历训练数据加载器，逐批获取数据和目标
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # 将数据和目标移动到指定的设备（例如 GPU 或 CPU）
            data, target = data.to(self.device), target.to(self.device)
            # 清空优化器的梯度
            self.optimizer.zero_grad()
            # 前向传播：将输入数据传入模型，得到输出
            output = self.model(data)
            # 计算损失：根据模型输出和目标值计算损失
            loss = self.criterion(output, target)
            loss.backward()
            # 更新模型参数
            self.optimizer.step()
            # 设置当前训练步骤，用于记录日志和监控训练过程
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # 更新训练指标，记录损失
            self.train_metrics.update('loss', loss.item())
            # 对于每个评估函数，计算对应的指标并更新到训练指标中
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
            # 每隔一定步骤 (log_step)，记录日志信息
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # 记录输入数据到日志中，并显示为图像形式
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            # 当迭代次数达到设置的 len_epoch 时，结束该 epoch 的训练
            if batch_idx == self.len_epoch:
                break
        # 获取并返回该 epoch 的所有训练指标的结果
        log = self.train_metrics.result()
        # 如果启用了验证，调用验证函数并记录验证指标
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        # 如果使用学习率调度器，更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # 返回该 epoch 的日志，包含损失和评估指标
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        在训练一个 epoch 后进行验证
        :param epoch: Integer, current training epoch.整数类型，表示当前的训练 epoch（轮次）
        :return: A log that contains information about validation 一个日志，包含有关验证过程的信息
        """
        # 将模型设置为评估模式，禁用 dropout 和 batch normalization 的训练行为
        self.model.eval()
        # 重置验证指标，以确保验证开始时指标从零开始
        self.valid_metrics.reset()
        # 在验证过程中不需要计算梯度，因此使用 torch.no_grad() 来禁用自动求导机制
        with torch.no_grad():
            # 遍历验证数据加载器，逐批获取数据和目标
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                # 将数据和目标移动到指定的设备（例如 GPU 或 CPU）
                data, target = data.to(self.device), target.to(self.device)
                # 前向传播：将输入数据传入模型，得到输出
                output = self.model(data)
                # 计算损失：根据模型输出和目标值计算损失
                loss = self.criterion(output, target)
                # 设置当前验证步骤，用于记录日志和监控验证过程
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # 更新验证指标，记录损失
                self.valid_metrics.update('loss', loss.item())
                # 对于每个评估函数，计算对应的指标并更新到验证指标中
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # 记录输入数据到日志中，并显示为图像形式
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        # 将模型的参数以直方图的形式添加到 TensorBoard 中
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        # 返回验证指标的结果
        return self.valid_metrics.result()

    def train_shared_cnn(epoch,
                             controller,
                             shared_cnn,
                             data_loaders,
                             shared_cnn_optimizer,
                             fixed_arc=None):
            """
            通过从控制器中采样架构来训练共享的 CNN（shared_cnn）。
            参数：
                epoch: 当前的训练轮数。
                controller: 生成待训练架构的控制器模块。
                shared_cnn: 包含所有可能架构的共享 CNN，具有共享权重。
                data_loaders: 包含数据加载器的字典。
                shared_cnn_optimizer: 用于更新 shared_cnn 的优化器。
                fixed_arc: 要训练的固定架构，如果提供，将覆盖控制器的采样。
                ...
            返回：
            无。
            """
            global vis_win  # 声明全局变量 vis_win，用于可视化

            controller.eval()  # 将控制器设置为评估模式，防止其参数在训练 shared_cnn 时被更新

            if fixed_arc is None:
                # 如果没有提供固定架构，在搜索架构时使用训练集的子集
                train_loader = data_loaders['train_subset']
            else:
                # 如果提供了固定架构，使用完整的训练集进行训练
                train_loader = data_loaders['train_dataset']

            train_acc_meter = AverageMeter()  # 初始化训练精度的度量器
            loss_meter = AverageMeter()  # 初始化损失的度量器

            for i, (images, labels) in enumerate(train_loader):  # 遍历训练数据加载器中的批次
                start = time.time()  # 记录当前时间，用于计算每个批次的耗时
                images = images.cuda()  # 将图像数据移动到 GPU
                labels = labels.cuda()  # 将标签数据移动到 GPU

            if fixed_arc is None:
                with torch.no_grad():
                    controller()  # 前向传播控制器，生成新的架构
                sample_arc = controller.sample_arc  # 获取控制器采样的架构
            else:
                sample_arc = fixed_arc  # 使用提供的固定架构

            shared_cnn.zero_grad()  # 清零 shared_cnn 的梯度
            pred = shared_cnn(images, sample_arc)  # 使用采样的架构进行前向传播，得到预测结果
            loss = nn.CrossEntropyLoss()(pred, labels)  # 计算交叉熵损失
            loss.backward()  # 反向传播，计算梯度
            grad_norm = torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), args.child_grad_bound)  # 对梯度进行裁剪，防止梯度爆炸
            shared_cnn_optimizer.step()  # 使用优化器更新 shared_cnn 的参数

            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))  # 计算训练精度

            train_acc_meter.update(train_acc.item())  # 更新训练精度的度量器
            loss_meter.update(loss.item())  # 更新损失的度量器

            end = time.time()  # 记录当前时间，计算每个批次的耗时

            if (i) % args.log_every == 0:
                learning_rate = shared_cnn_optimizer.param_groups[0]['lr']  # 获取当前的学习率
                display = 'epoch=' + str(epoch) + \
                          '\tch_step=' + str(i) + \
                          '\tloss=%.6f' % (loss_meter.val) + \
                          '\tlr=%.4f' % (learning_rate) + \
                          '\t|g|=%.4f' % (grad_norm.item()) + \
                          '\tacc=%.4f' % (train_acc_meter.val) + \
                          '\ttime=%.2fit/s' % (1. / (end - start))  # 格式化日志信息
                print(display)  # 输出日志信息

        # 更新共享 CNN 精度的可视化曲线
        vis_win['shared_cnn_acc'] = visdom.line(
            X=np.array([epoch]),  # 横坐标为当前轮数
            Y=np.array([train_acc_meter.avg]),  # 纵坐标为平均训练精度
            win=vis_win['shared_cnn_acc'],  # 指定窗口
            opts=dict(title='shared_cnn_acc', xlabel='Iteration', ylabel='Accuracy'),  # 图表选项
            update='append' if epoch > 0 else None)  # 如果不是第一个轮次，则追加数据

        # 更新共享 CNN 损失的可视化曲线
        vis_win['shared_cnn_loss'] = vis.line(
            X=np.array([epoch]),  # 横坐标为当前轮数
            Y=np.array([loss_meter.avg]),  # 纵坐标为平均损失
            win=vis_win['shared_cnn_loss'],  # 指定窗口
            opts=dict(title='shared_cnn_loss', xlabel='Iteration', ylabel='Loss'),  # 图表选项
            update='append' if epoch > 0 else None)  # 如果不是第一个轮次，则追加数据

        controller.train()  # 将控制器设置回训练模式，以便在架构搜索阶段更新其参数

    def _progress(self, batch_idx):
        """
        返回当前训练进度的字符串表示形式
        :param batch_idx: 当前的批次索引
        :return: 一个包含当前进度信息的字符串，格式为 [当前/总共 (百分比%)]
        """
        # 定义显示格式，格式为 [当前/总共 (百分比%)]
        base = '[{}/{} ({:.0f}%)]'
        # 如果 data_loader 有 'n_samples' 属性（说明有样本总数），计算基于样本数量的进度
        if hasattr(self.data_loader, 'n_samples'):
            # 计算当前处理的样本数量：批次索引乘以批次大小
            current = batch_idx * self.data_loader.batch_size
            # 获取总样本数
            total = self.data_loader.n_samples
        else:
            # 如果没有 'n_samples' 属性，使用基于批次数量的进度
            current = batch_idx
            # 使用 len_epoch 作为总批次数
            total = self.len_epoch
        # 返回格式化的进度字符串，包含当前处理的数量、总数和百分比
        return base.format(current, total, 100.0 * current / total)

    def _check_abs_max_grad(self, abs_max_grad, model):
        """
        检查模型的梯度，以确定本轮迭代中的最大绝对梯度，用于跟踪梯度爆炸情况。
        Args:
            abs_max_grad: 当前记录的最大绝对梯度值。
            model: 需要检查的模型。
        Returns:
            更新后的最大绝对梯度值。
        """
        # 从模型参数中提取有效的梯度
        finite_grads = [p.grad.data
                        for p in model.parameters()
                        if p.grad is not None]
        # 找到当前模型参数中的新最大梯度
        new_max_grad = max([grad.max() for grad in finite_grads])
        # 找到当前模型参数中的新最小梯度
        new_min_grad = min([grad.min() for grad in finite_grads])
        # 计算当前模型的绝对最大梯度
        new_abs_max_grad = max(new_max_grad, abs(new_min_grad))
        # 如果新的绝对最大梯度超过当前记录的最大绝对梯度，更新记录并打印日志
        if new_abs_max_grad > abs_max_grad:
            self.logger.info(f'abs max grad {abs_max_grad}')
            return new_abs_max_grad  # 返回更新后的最大绝对梯度
        return abs_max_grad  # 返回未更新的最大绝对梯度

    def get_loss(self, inputs, targets, hidden, dags):
        """计算同一批次数据在多个模型上的损失。
        这相当于对损失的估计，然后用于计算共享模型的梯度估计。
        """
        if not isinstance(dags, list):
            dags = [dags]
        loss = 0
        for dag in dags:
            # 前向传播，获取输出、更新后的隐藏状态和额外输出
            output, hidden, extra_out = self.shared_model(inputs, dag, hidden)
            # 调整输出形状
            output = output.view(-1, output.size(2))  # [seq_length * batch_size, num_tokens]
            targets_flat = targets.view(-1)  # [seq_length * batch_size]
            # 计算损失并归一化
            sample_loss = self.criterion(output, targets_flat) / len(dags)
            # 累加损失
            loss += sample_loss
        # 如果处理多个 dag，需要处理隐藏状态的问题
        assert len(dags) == 1, 'There are multiple `hidden` for multiple `dags`'
        return loss, hidden, extra_out


    def _summarize_controller_train(self, total_loss, adv_history, entropy_history, reward_history,
                                    avg_reward_base, dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.config.Misc.log_step
        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)
        if avg_reward_base is None:
            avg_reward_base = avg_reward
        self.logger.info(
            f'| epoch {self.epoch:3d} | lr {self.config.learning.train.controller_optimizer.args.lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')
        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)

            paths = []
            for dag in dags:
                fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                         f'{avg_reward:6.4f}.png')
                path = os.path.join(self.args.model_dir, 'networks', fname)
                utils.draw_network(dag, path)
                paths.append(path)

            self.tb.image_summary('controller/sample',
                                  paths,
                                  self.controller_step)

    def _summarize_shared_train(self, total_loss, raw_total_loss):
        """Logs a set of training steps."""
        cur_loss = utils.to_item(total_loss) / self.args.log_step
        # NOTE(brendan): The raw loss, without adding in the activation
        # regularization terms, should be used to compute ppl.
        cur_raw_loss = utils.to_item(raw_total_loss) / self.args.log_step
        ppl = math.exp(cur_raw_loss)

        self.logger.info(f'| epoch {self.epoch:3d} '
                         f'| lr {self.shared_lr:4.2f} '
                         f'| raw loss {cur_raw_loss:.2f} '
                         f'| loss {cur_loss:.2f} '
                         f'| ppl {ppl:8.2f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('shared/loss',
                                   cur_loss,
                                   self.shared_step)
            self.tb.scalar_summary('shared/perplexity',
                                   ppl,
                                   self.shared_step)
