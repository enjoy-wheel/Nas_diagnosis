import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        # 将传入的 validation_split 参数（验证集的占比）保存为实例的属性
        self.validation_split = validation_split

        # 将传入的 shuffle 参数（是否打乱数据）保存为实例的属性
        self.shuffle = shuffle

        # 初始化批次索引为0，表示从数据集的开始处开始
        self.batch_idx = 0

        # 获取数据集的样本总数，并将其保存在实例的 n_samples 属性中
        self.n_samples = len(dataset)

        # 调用 _split_sampler 方法，根据验证集占比来划分训练集和验证集，并返回两个采样器
        # sampler 用于训练集，valid_sampler 用于验证集
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # 创建一个字典保存初始化所需的参数，稍后会传递给 DataLoader
        self.init_kwargs = {
            'dataset': dataset,  # 数据集对象
            'batch_size': batch_size,  # 每个批次的样本数量
            'shuffle': self.shuffle,  # 是否对数据进行打乱
            'collate_fn': collate_fn,  # 用于处理批数据的合并函数
            'num_workers': num_workers  # 加载数据时使用的子进程数
        }

        # 调用父类（通常是 torch.utils.data.DataLoader）的构造函数，初始化 DataLoader 实例
        # 传入 sampler 进行数据采样，使用 **self.init_kwargs 解包字典，将其他参数传递给父类的构造函数
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        # 如果不需要划分验证集，返回 None
        if split == 0.0:
            return None, None

        # 创建一个包含所有样本索引的数组
        idx_full = np.arange(self.n_samples)

        # 设置随机种子，确保每次运行时打乱的顺序相同
        np.random.seed(0)

        # 打乱索引数组
        np.random.shuffle(idx_full)

        # 如果 split 是整数，表示指定验证集的大小
        if isinstance(split, int):
            # 验证验证集大小是否合理
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            # 设置验证集的大小为指定的整数
            len_valid = split
        else:
            # 如果 split 是比例（0.2 等），则根据数据集大小计算验证集的样本数
            len_valid = int(self.n_samples * split)

        # 划分出验证集的索引
        valid_idx = idx_full[0:len_valid]
        # 剩余的索引是训练集的索引
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # 使用 SubsetRandomSampler 来从训练集的索引中随机采样
        train_sampler = SubsetRandomSampler(train_idx)
        # 使用 SubsetRandomSampler 来从验证集的索引中随机采样
        valid_sampler = SubsetRandomSampler(valid_idx)

        # 禁用 shuffle，因为我们已经通过 Sampler 实现了数据的随机化
        self.shuffle = False
        # 更新样本数量为训练集的大小
        self.n_samples = len(train_idx)

        # 返回训练集和验证集的采样器
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
