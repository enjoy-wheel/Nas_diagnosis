import importlib
from datetime import datetime


class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        # 初始化方法，设置Tensorboard的日志目录、日志记录器和是否启用Tensorboard
        self.writer = None  # 初始化 writer 属性，用于保存 TensorBoard 写入对象
        self.selected_module = ""  # 初始化选中的模块名

        if enabled:  # 如果启用 Tensorboard
            log_dir = str(log_dir)  # 将log_dir转换为字符串类型

            # 尝试导入 tensorboardX 或 torch.utils.tensorboard 模块
            succeeded = False  # 标记模块导入是否成功
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    # 尝试导入对应的模块，并初始化 SummaryWriter
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True  # 导入成功
                    break  # 跳出循环
                except ImportError:
                    succeeded = False  # 导入失败
                self.selected_module = module  # 记录当前尝试的模块

            # 如果两个模块都没有成功导入，输出警告信息
            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0  # 初始化 step 属性，表示当前训练的步数
        self.mode = ''  # 初始化 mode 属性，表示当前的模式（训练或验证）

        # 记录 TensorBoard 支持的操作方法
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        # 记录不需要添加 mode 标签的操作方法
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        # 初始化计时器
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        # 设置当前步数和模式的方法
        self.mode = mode  # 设置当前模式（例如：'train' 或 'valid'）
        self.step = step  # 设置当前步数
        if step == 0:
            self.timer = datetime.now()  # 重置计时器
        else:
            duration = datetime.now() - self.timer  # 计算与上次记录的时间差
            # 记录每秒训练步数
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()  # 重置计时器

    def __getattr__(self, name):
        """
        获取属性方法，动态返回 TensorBoard 写入方法，并在调用时加入步骤（step）和模式（train/valid）信息
        如果没有该方法，则返回空函数。
        """
        if name in self.tb_writer_ftns:  # 如果是 TensorBoard 支持的操作方法之一
            add_data = getattr(self.writer, name, None)  # 获取对应的 TensorBoard 写入方法

            # 定义一个包装器，在调用 TensorBoard 写入方法时加入步骤和模式信息
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # 如果是需要添加 mode 标签的方法
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)  # 将 mode（train/valid）加到标签上
                    add_data(tag, data, self.step, *args, **kwargs)  # 调用实际的写入方法
            return wrapper  # 返回包装器函数
        else:
            # 如果方法不在支持的列表中，尝试返回类的其他属性方法
            try:
                attr = object.__getattr__(name)  # 获取类中其他的方法
            except AttributeError:
                # 如果没有该方法，抛出 AttributeError 异常
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr  # 返回属性方法
