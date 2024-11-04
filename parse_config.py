import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        该类的构造函数用于解析配置的JSON文件。它负责处理训练中的超参数、模块初始化、检查点保存以及日志模块的配置
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param config: 字典类型，包含训练的配置和超参数，通常是`config.json`文件的内容
        :param resume: String, path to the checkpoint being loaded.
        :param resume: 字符串类型，表示加载的检查点路径
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param modification: 字典类型，以keychain:value形式指定要替换config字典中的位置和值
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        :param run_id: 唯一的训练进程标识符，用于保存检查点和训练日志。如果没有提供run_id，默认使用当前的时间戳
        """
        # load config file and apply modification
        # 加载配置文件并应用修改（如果有）
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        # 设置保存训练模型和日志的目录
        save_dir = Path(self.config['trainer']['save_dir'])

        # 从配置中定义实验名称
        exper_name = self.config['name']

        # 如果没有提供run_id，则使用当前时间戳作为默认的run-id
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        # 定义保存模型和日志的目录，按照实验名称和运行ID组织
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        # 创建保存检查点和日志文件的目录
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        # 将更新后的配置文件保存到检查点目录中
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        # 配置日志模块，将日志保存到log目录中
        setup_logging(self.log_dir)

        # 定义日志级别，将不同的详细级别映射到对应的日志常量
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        从命令行参数初始化该类。用于训练或测试阶段
        :param args: 命令行参数，通常为 argparse.Namespace 实例。
        :param options: 其他自定义命令行选项，默认值为空字符串
        """
        # 遍历提供的选项，并将其添加为命令行参数
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        # 如果args不是元组类型，则解析命令行参数
        if not isinstance(args, tuple):
            args = args.parse_args()

        # 如果指定了设备，则设置 CUDA 设备的环境变量
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        # 如果指定了检查点路径，则加载对应的配置文件
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            # 如果没有指定配置文件，则抛出错误
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        # 读取配置文件
        config = read_json(cfg_fname)

        # 如果提供了新的配置文件和检查点，则进行微调，更新配置
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        # 解析自定义命令行选项，将其转换为字典  
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    # init_obj 让你可以根据配置文件中的信息灵活地初始化模块或类，而无需手动指定每一个类或参数
    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        根据配置文件中给定的'name'来找到函数句柄，并使用相应的参数进行实例化
        例：
        config = {
        "model": {
            "type": "ResNet",  # 类名
             "args": {
                "num_classes": 10,
                "input_channels": 3  # 初始化参数
                }
            }
        }
        class ResNet:
        def __init__(self, num_classes, input_channels):
            self.num_classes = num_classes
            self.input_channels = input_channels
        def __repr__(self):
            return f"ResNet(num_classes={self.num_classes}, input_channels={self.input_channels})"
        model_instance = config.init_obj('model', models)
        print(model_instance)  # 输出: ResNet(num_classes=10, input_channels=3)
        :param name: 在配置中指定的模块名称。
        :param module: 模块，其中包含要实例化的类或函数。
        :param args: 传递给类或函数的非关键字参数。
        :param kwargs: 传递给类或函数的关键字参数。
        :return: 返回通过类或函数句柄初始化的实例。
        """
        # 从配置中获取模块的名称
        module_name = self[name]['type']
        # 从配置中获取模块的初始化参数
        module_args = dict(self[name]['args'])

        # 确保不允许覆盖配置文件中定义的参数
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'

        # 更新参数，将kwargs合并到module_args中
        module_args.update(kwargs)
        # 使用反射（getattr）来调用模块中的类或函数，并传入参数进行实例化
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        根据配置文件中的内容动态地生成带有固定参数的函数
        示例同上
        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    """该函数用于根据传入的 modification 字典更新 config（通常是一个嵌套字典，代表整个配置文件）"""
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys.
    在嵌套的字典（tree）中，通过一系列的键（keys）设置一个值"""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys.
    通过一系列键（keys）来访问嵌套的字典（tree）"""
    return reduce(getitem, keys, tree)
