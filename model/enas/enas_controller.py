import collections
import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
import utils
from utils import Node
from base import BaseModel

class Controller(BaseModel):
    """神经架构搜索（NAS）的控制器，基于 RNN。"""

    def __init__(self, args):
        # 初始化控制器，设置网络参数和结构
        super(Controller, self).__init__()  # 调用父类构造函数
        self.args = args

        # 根据网络类型选择不同的参数
        if self.args.network_type == 'rnn':
            # 对于 RNN，设置激活函数和块数
            self.num_tokens = [len(args.shared_rnn_activations)]
            for idx in range(self.args.num_blocks):
                self.num_tokens += [idx + 1, len(args.shared_rnn_activations)]
            self.func_names = args.shared_rnn_activations
        elif self.args.network_type == 'cnn':
            # 对于 CNN，设置共享的 CNN 类型和块数
            self.num_tokens = [len(args.shared_cnn_types), self.args.num_blocks]
            self.func_names = args.shared_cnn_types

        num_total_tokens = sum(self.num_tokens)  # 计算总的令牌数量
        # 定义嵌入层和 LSTM 单元
        self.encoder = torch.nn.Embedding(num_total_tokens, args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
        # 定义解码器
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(args.controller_hid, size) for size in self.num_tokens]
        )

        self.reset_parameters()  # 重置参数
        # 创建静态隐藏状态和输入的默认值
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        self.static_inputs = utils.keydefaultdict(lambda key: utils.get_variable(
            torch.zeros(key, self.args.controller_hid), self.args.cuda, requires_grad=False))

    def reset_parameters(self):
        """重置模型的参数，使其在指定范围内均匀分布。"""
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)  # 初始化为随机值
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)  # 偏置初始化为0

    @abstractmethod
    def forward(self, inputs, hidden, block_idx, is_embed):
        """前向传播逻辑，根据输入和隐藏状态计算输出。"""
        embed = self.encoder(inputs) if not is_embed else inputs  # 嵌入输入
        hx, cx = self.lstm(embed, hidden)  # 通过 LSTM 单元计算隐藏状态
        logits = self.decoders[block_idx](hx) / self.args.softmax_temperature  # 解码器输出

        if self.args.mode == 'train':
            logits = (self.args.tanh_c * F.tanh(logits))  # 训练模式下应用激活函数

        return logits, (hx, cx)  # 返回输出和新的隐藏状态

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """从控制器中采样生成计算节点，构建 DAG。"""
        if batch_size < 1:
            raise Exception(f'错误的 batch_size: {batch_size} < 1')  # 确保 batch_size 合法

        inputs = self.static_inputs[batch_size]  # 获取静态输入
        hidden = self.static_init_hidden[batch_size]  # 获取初始化隐藏状态

        # 用于保存激活函数、熵、对数概率和前序节点的列表
        activations, entropies, log_probs, prev_nodes = [], [], [], []

        for block_idx in range(2 * (self.args.num_blocks - 1) + 1):
            # 在每个块上进行前向传播
            logits, hidden = self.forward(inputs, hidden, block_idx, is_embed=(block_idx == 0))
            probs = F.softmax(logits, dim=-1)  # 计算概率分布
            log_prob = F.log_softmax(logits, dim=-1)  # 计算对数概率
            entropy = -(log_prob * probs).sum(1, keepdim=False)  # 计算熵

            action = probs.multinomial(num_samples=1).data  # 从概率中采样
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))  # 获取选择的对数概率

            # 保存熵和对数概率
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 处理当前块的输入
            mode = block_idx % 2  # 交替处理激活函数和前序节点
            inputs = utils.get_variable(action[:, 0] + sum(self.num_tokens[:mode]), requires_grad=False)

            if mode == 0:
                activations.append(action[:, 0])  # 保存激活函数
            elif mode == 1:
                prev_nodes.append(action[:, 0])  # 保存前序节点

        # 将采样得到的前序节点和激活函数转换为张量
        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        # 构建 DAG
        dags = self.construct_dags(prev_nodes, activations)

        # 如果指定了保存目录，绘制 DAG 图
        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag, os.path.join(save_dir, f'graph{idx}.png'))

        return (dags, torch.cat(log_probs), torch.cat(entropies)) if with_details else dags

    def init_hidden(self, batch_size):
        """初始化隐藏状态为零。"""
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

    def construct_dags(self, prev_nodes, activations):
        """根据前序节点和激活函数构建有向无环图（DAG）。"""
        dags = []
        for nodes, func_ids in zip(prev_nodes, activations):
            dag = collections.defaultdict(list)  # 使用默认字典存储 DAG

            # 添加第一个节点
            dag[-1] = [Node(0, self.func_names[func_ids[0]])]
            dag[-2] = [Node(0, self.func_names[func_ids[0]])]

            # 添加剩余节点
            for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
                dag[utils.to_item(idx)].append(Node(jdx + 1, self.func_names[func_id]))

            leaf_nodes = set(range(self.args.num_blocks)) - dag.keys()  # 找到叶节点

            # 处理叶节点
            for idx in leaf_nodes:
                dag[idx] = [Node(len(self.func_names), 'avg')]  # 叶节点输出为平均值

            last_node = Node(len(self.func_names) + 1, 'h[t]')  # 最后一个节点
            dag[len(self.func_names)] = [last_node]
            dags.append(dag)  # 添加到 DAG 列表

        return dags

    def __str__(self):
        """返回控制器的字符串表示，包括基类信息。"""
        return super().__str__() + '\n控制器特定的细节...'
