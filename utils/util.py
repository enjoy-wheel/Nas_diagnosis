
import json

import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


try:
    import scipy.misc
    imread = scipy.misc.imread
    imresize = scipy.misc.imresize
    imsave = imwrite = scipy.misc.imsave
except:
    import cv2
    imread = cv2.imread
    imresize = lambda img, size: cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
    imsave = imwrite = cv2.imwrite

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    """
    wrapper function for endless data loader.
    """
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricTracker:
    def __init__(self, *keys, writer=None):
        """
        初始化 MetricTracker 类，用于跟踪多个指标的总值、计数和平均值
        :param *keys: 传入的指标名称，用于定义需要跟踪的指标
        :param writer: 可选的 TensorBoard writer，用于记录指标的变化
        """
        # 保存 TensorBoard writer（如果提供的话）
        self.writer = writer
        # 创建一个 Pandas DataFrame，用于存储每个指标的总值（total）、计数（counts）和平均值（average）
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        # 重置所有指标的值
        self.reset()

    def reset(self):
        """
        重置所有指标的 total、counts 和 average 值为 0
        """
        # 将 DataFrame 中的所有列的值重置为 0
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        更新指定指标的值
        :param key: 要更新的指标名称
        :param value: 指标的最新值
        :param n: 更新的样本数量，默认是 1
        """
        # 如果 writer 不为空，将当前值记录到 TensorBoard 中
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        # 累加指标的总值，total = total + value * n
        self._data.total[key] += value * n
        # 累加计数，counts = counts + n
        self._data.counts[key] += n
        # 计算新的平均值，average = total / counts
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        返回指定指标的当前平均值。

        :param key: 指标名称。
        :return: 指标的平均值。
        """
        return self._data.average[key]

    def result(self):
        """
        返回所有指标的平均值。

        :return: 一个字典，键为指标名称，值为对应的平均值。
        """
        return dict(self._data.average)

