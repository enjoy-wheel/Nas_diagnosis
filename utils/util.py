import collections
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import pygraphviz as pgv


from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import logger

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

##########################
# Network visualization
##########################

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('x'):
        color = 'white'
    elif label.startswith('h'):
        color = 'skyblue'
    elif label == 'tanh':
        color = 'yellow'
    elif label == 'ReLU':
        color = 'pink'
    elif label == 'identity':
        color = 'orange'
    elif label == 'sigmoid':
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in  ['x', 'avg', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )

def draw_network(dag, path):
    os.makedirs(os.path.dirname(path))
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open') # not work?

    checked_ids = [-2, -1, 0]

    if -1 in dag:
        add_node(graph, -1, 'x[t]')
    if -2 in dag:
        add_node(graph, -2, 'h[t-1]')

    add_node(graph, 0, dag[-1][0].name)

    for idx in dag:
        for node in dag[idx]:
            if node.id not in checked_ids:
                add_node(graph, node.id, node.name)
                checked_ids.append(node.id)
            graph.add_edge(idx, node.id)

    graph.layout(prog='dot')
    graph.draw(path)

def make_gif(paths, gif_path, max_frame=50, prefix=""):
    import imageio

    paths.sort()

    skip_frame = len(paths) // max_frame
    paths = paths[::skip_frame]

    images = [imageio.imread(path) for path in paths]
    max_h, max_w, max_c = np.max(
            np.array([image.shape for image in images]), 0)

    for idx, image in enumerate(images):
        h, w, c = image.shape
        blank = np.ones([max_h, max_w, max_c], dtype=np.uint8) * 255

        pivot_h, pivot_w = (max_h-h)//2, (max_w-w)//2
        blank[pivot_h:pivot_h+h,pivot_w:pivot_w+w,:c] = image

        images[idx] = blank

    try:
        images = [Image.fromarray(image) for image in images]
        draws = [ImageDraw.Draw(image) for image in images]
        font = ImageFont.truetype("assets/arial.ttf", 30)

        steps = [int(os.path.basename(path).rsplit('.', 1)[0].split('-')[1]) for path in paths]
        for step, draw in zip(steps, draws):
            draw.text((max_h//20, max_h//20),
                      f"{prefix}step: {format(step, ',d')}", (0, 0, 0), font=font)
    except IndexError:
        pass

    imageio.mimsave(gif_path, [np.array(img) for img in images], duration=0.5)

##########################
# ETC
##########################

Node = collections.namedtuple('Node', ['id', 'name'])

class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()

def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_dag(args, dag, name):
    save_path = os.path.join(args.model_dir, name)
    logger.info("[*] Save dag : {}".format(save_path))
    json.dump(dag, open(save_path, 'w'))

def load_dag(args):
    load_path = os.path.join(args.dag_path)
    logger.info("[*] Load dag : {}".format(load_path))
    with open(load_path) as f:
        dag = json.load(f)
    dag = {int(k): [Node(el[0], el[1]) for el in v] for k, v in dag.items()}
    save_dag(args, dag, "dag.json")
    draw_network(dag, os.path.join(args.model_dir, "dag.png"))
    return dag

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

##########################
# Torch
##########################

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data



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

