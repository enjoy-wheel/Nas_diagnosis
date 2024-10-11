import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        # 调用父类的初始化函数，并传入模型、损失函数、评估指标、优化器和配置对象
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        # 保存配置对象和设备信息
        self.config = config
        self.device = device

        # 保存训练数据的 DataLoader
        self.data_loader = data_loader

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

        # 保存学习率调度器（如果有的话）
        self.lr_scheduler = lr_scheduler

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
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        # 如果使用学习率调度器，更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 返回该 epoch 的日志，包含损失和评估指标
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
