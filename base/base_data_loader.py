import numpy as np
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, sampler=None):
        # 将传入的 validation_split 参数（验证集的占比）保存为实例的属性
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        # 若提供了sampler，则不需要根据validation_split来划分数据集
        if sampler is None:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        else:
            self.sampler = sampler
            self.valid_sampler = None
            self.validation_split = 0.0

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
            return BaseDataLoader(
                dataset=self.init_kwargs['dataset'],
                batch_size=self.init_kwargs['batch_size'],
                shuffle=False,
                validation_split=0.0,
                num_workers=self.init_kwargs['num_workers'],
                collate_fn=self.init_kwargs['collate_fn'],
                sampler=self.valid_sampler
            )

    def get_subset(self, subset_size, random_subset=True):
        """
        Returns a DataLoader for a subset of the dataset.

        Parameters:
        - subset_size (int): The number of samples in the subset.
        - random_subset (bool): If True, the subset is randomly sampled.
                                 If False, the subset consists of the first `subset_size` samples.

        Returns:
        - DataLoader: A DataLoader instance for the subset.
        """
        assert subset_size > 0, "subset_size must be a positive integer."
        assert subset_size <= self.n_samples, "subset_size cannot exceed the number of samples in the dataset."

        if random_subset:
            np.random.seed(0)
            subset_indices = np.random.choice(self.n_samples, subset_size, replace=False)
            subset_sampler = SubsetRandomSampler(subset_indices)

            return BaseDataLoader(
                dataset=self.init_kwargs['dataset'],
                batch_size=self.init_kwargs['batch_size'],
                shuffle=False,
                validation_split=0.0,
                num_workers=self.init_kwargs['num_workers'],
                collate_fn=self.init_kwargs['collate_fn'],
                sampler=subset_sampler
            )
        else:
            # 顺序取前 subset_size 个样本
            subset_indices = np.arange(subset_size)
            # 同样使用 Subset 来创建一个子集dataset
            subset_dataset = Subset(self.init_kwargs['dataset'], subset_indices)

            # 此时不需要 sampler，直接传子集dataset即可
            return BaseDataLoader(
                dataset=subset_dataset,
                batch_size=self.init_kwargs['batch_size'],
                shuffle=False,
                validation_split=0.0,
                num_workers=self.init_kwargs['num_workers'],
                collate_fn=self.init_kwargs['collate_fn']
            )
