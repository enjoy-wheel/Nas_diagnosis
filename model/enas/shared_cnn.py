import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel

'''
Implementation Notes:
-Setting track_running_stats to True in BatchNorm layers seems to hurt validation
    and test performance for some reason, so here it is disabled even though it
    is used in the official implementation.
'''


class FactorizedReduction(nn.Module):
    '''
    将空间维度（宽度和高度）缩小一半，并且可以选择更改输出过滤器的数量
    原链接: melodyguan/enas - general_child.py, Line 129
    '''
    def __init__(self, in_planes, out_planes, stride=2):
        super(FactorizedReduction, self).__init__()

        assert out_planes % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        if stride == 1:
            self.fr = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False))
        else:
            self.path1 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))

            self.path2 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))
            self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        if self.stride == 1:
            return self.fr(x)
        else:
            path1 = self.path1(x)

            # pad the right and the bottom, then crop to include those pixels
            path2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
            path2 = path2[:, :, 1:, 1:]
            path2 = self.path2(path2)

            out = torch.cat([path1, path2], dim=1)
            out = self.bn(out)
            return out


class ENASLayer(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
    '''
    def __init__(self, layer_id, in_planes, out_planes):
        super(ENASLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3)
        self.branch_1 = ConvBranch(in_planes, out_planes, kernel_size=3, separable=True)
        self.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5)
        self.branch_3 = ConvBranch(in_planes, out_planes, kernel_size=5, separable=True)
        self.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
        self.branch_5 = PoolBranch(in_planes, out_planes, 'max')
        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x, prev_layers, sample_arc):
        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []
        if layer_type == 0:
            out = self.branch_0(x)
        elif layer_type == 1:
            out = self.branch_1(x)
        elif layer_type == 2:
            out = self.branch_2(x)
        elif layer_type == 3:
            out = self.branch_3(x)
        elif layer_type == 4:
            out = self.branch_4(x)
        elif layer_type == 5:
            out = self.branch_5(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out


class FixedLayer(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
    '''
    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(FixedLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.sample_arc = sample_arc

        self.layer_type = sample_arc[0]
        if self.layer_id > 0:
            self.skip_indices = sample_arc[1]
        else:
            self.skip_indices = torch.zeros(1)

        if self.layer_type == 0:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=3)
        elif self.layer_type == 1:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=3, separable=True)
        elif self.layer_type == 2:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=5)
        elif self.layer_type == 3:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=5, separable=True)
        elif self.layer_type == 4:
            self.branch = PoolBranch(in_planes, out_planes, 'avg')
        elif self.layer_type == 5:
            self.branch = PoolBranch(in_planes, out_planes, 'max')
        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # Use concatentation instead of addition in the fixed layer for some reason
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        self.dim_reduc = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_planes, track_running_stats=False))

    def forward(self, x, prev_layers, sample_arc):
        out = self.branch(x)

        res_layers = []
        for i, skip in enumerate(self.skip_indices):
            if skip == 1:
                res_layers.append(prev_layers[i])
        prev = res_layers + [out]
        prev = torch.cat(prev, dim=1)

        out = self.dim_reduc(prev)
        return out


class SeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, bias):
        super(SeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                                   padding=padding, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L483
    '''
    def __init__(self, in_planes, out_planes, kernel_size, separable=False):
        super(ConvBranch, self).__init__()
        assert kernel_size in [3, 5], "Kernel size must be either 3 or 5"

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.separable = separable

        self.inp_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, track_running_stats=False),
            nn.ReLU())

        if separable:
            self.out_conv = nn.Sequential(
                SeparableConv(in_planes, out_planes, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False),
                nn.ReLU())
        else:
            padding = (kernel_size - 1) // 2
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          padding=padding, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False),
                nn.ReLU())

    def forward(self, x):
        out = self.inp_conv1(x)
        out = self.out_conv(out)
        return out


class PoolBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L546
    '''
    def __init__(self, in_planes, out_planes, avg_or_max):
        super(PoolBranch, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.avg_or_max = avg_or_max

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, track_running_stats=False),
            nn.ReLU())

        if avg_or_max == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        elif avg_or_max == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out


class SharedCNN(BaseModel):
    def __init__(self,
                 num_layers=12,
                 num_branches=6,
                 out_filters=24,
                 keep_prob=1.0,
                 fixed_arc=None
                 ):
        super(SharedCNN, self).__init__()

        self.num_layers = num_layers  # 网络的层数
        self.num_branches = num_branches  # 每层分支数
        self.out_filters = out_filters  # 输出的通道数
        self.keep_prob = keep_prob  # 保持的概率（用于dropout）
        self.fixed_arc = fixed_arc  # 固定的架构（如果有的话）

        pool_distance = self.num_layers // 3  # 每隔三分之一的层数进行一次池化
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]  # 需要池化的层

        # 初始卷积层 (stem_conv)
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, out_filters, kernel_size=3, padding=1, bias=False),  # 卷积层，3x3卷积核
            nn.BatchNorm2d(out_filters, track_running_stats=False))  # 批归一化层

        # 用于存储各个层的ModuleList
        self.layers = nn.ModuleList([])
        self.pooled_layers = nn.ModuleList([])

        # 构建每一层
        for layer_id in range(self.num_layers):
            # 如果架构不固定，使用 ENASLayer
            if self.fixed_arc is None:
                layer = ENASLayer(layer_id, self.out_filters, self.out_filters)
            else:
                # 使用 FixedLayer 并传入固定架构
                layer = FixedLayer(layer_id, self.out_filters, self.out_filters, self.fixed_arc[str(layer_id)])
            self.layers.append(layer)

            # 如果当前层需要池化，进行下采样
            if layer_id in self.pool_layers:
                for i in range(len(self.layers)):
                    # 如果架构不固定，使用 FactorizedReduction 将输出维度保持不变
                    if self.fixed_arc is None:
                        self.pooled_layers.append(FactorizedReduction(self.out_filters, self.out_filters))
                    else:
                        # 固定架构时，将输出维度加倍
                        self.pooled_layers.append(FactorizedReduction(self.out_filters, self.out_filters * 2))
                if self.fixed_arc is not None:
                    self.out_filters *= 2  # 输出通道数加倍

        # 全局平均池化和分类层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化到1x1
        self.dropout = nn.Dropout(p=1. - self.keep_prob)  # dropout层
        self.classify = nn.Linear(self.out_filters, 10)  # 全连接分类层，10类输出（适用于CIFAR-10）

        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, sample_arc):
        # 前向传播
        x = self.stem_conv(x)  # 初始卷积层
        prev_layers = []  # 存储前几层的输出
        pool_count = 0  # 记录池化次数
        for layer_id in range(self.num_layers):
            # 每一层的前向传播
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)  # 保存当前层的输出到前几层列表

            # 如果当前层需要池化，则对所有前几层的输出进行下采样
            if layer_id in self.pool_layers:
                for i, prev_layer in enumerate(prev_layers):
                    prev_layers[i] = self.pooled_layers[pool_count](prev_layer)  # 对前几层进行池化
                    pool_count += 1
                x = prev_layers[-1]  # 更新当前层的输出为最新的池化输出

        # 全局平均池化、展平、dropout 和分类
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)  # 展平
        x = self.dropout(x)  # dropout
        out = self.classify(x)  # 分类层

        return out  # 返回最终的分类结果

