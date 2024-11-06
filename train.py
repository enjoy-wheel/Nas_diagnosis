import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.enas.enas_controller as controller_arch
import model.enas.shared_cnn as shared_cnn_arch
import model.enas.shared_rnn as shared_cnn_arch
from parse_config import ConfigParser
from trainer import EnasTrainer
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # 初始化一个日志记录器，用于记录训练过程中的信息、警告、错误等。日志记录器有助于调试和监控模型训练的进展
    logger = config.get_logger('train')

    # 根据配置文件中关于数据加载器的设置，初始化一个数据加载器对象
    # setup data_loader instances
    # @module_data: 数据加载器类
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # 根据配置文件中关于模型架构的设置，初始化模型实例
    # build model architecture, then print to console
    shared_model = config.init_obj('arch', module_arch)
    controller_model = config.init_obj('arch', )
    # 将模型的结构信息记录到日志中，便于查看和验证模型架构
    logger.info('regularizing:')
    for regularizer in [('activation regularization',
                         config.config["PTB_regularizations"]["activation_regularization"]["state"]),
                        ('temporal activation regularization',
                         config.config["PTB_regularizations"]["temporal_activation_regularization"]["state"]),
                        ('norm stabilizer regularization',
                         config.config["PTB_regularizations"]["norm_stabilizer_regularization"]["state"])]:
        if regularizer[1]:
            logger.info(f'{regularizer[0]}')
    logger.info(shared_model)
    logger.info(controller_model)
    # 根据配置文件中指定的 GPU 数量，准备训练设备
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    shared_model = shared_model.to(device)
    controller_model = controller_model.to(device)
    # 如果使用多个 GPU，则使用 DataParallel 将模型并行分布到多个 GPU 上，以加速训练过程
    if len(device_ids) > 1:
        shared_model = torch.nn.DataParallel(shared_model, device_ids=device_ids)
        controller_model = torch.nn.DataParallel(controller_model, device_ids=device_ids)
    # 获取损失函数和评估指标
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # 构建优化器和学习率调度器
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # 过滤出模型中需要更新的参数（requires_grad = True），有些模型参数可能被冻结（如预训练模型的某些层），不需要在训练中更新
    shared_trainable_params = filter(lambda p: p.requires_grad, shared_model.parameters())
    controller_trainable_params = filter(lambda p: p.requires_grad, controller_model.parameters())
    # 使用配置对象初始化优化器
    shared_optimizer = config.init_obj('shared_optimizer', torch.optim, shared_trainable_params)
    controller_optimizer = config.init_obj('controller_optimizer', torch.optim, controller_trainable_params)
    # 使用配置对象初始化学习率调度器
    shared_lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, shared_optimizer)
    controller_lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, controller_optimizer)

    # 创建训练器Trainer实例，并开始训练
    trainer = EnasTrainer(models=(shared_model, controller_model),
                          criterion=criterion,
                          metric_ftns=metrics,
                          optimizers=(shared_optimizer, controller_optimizer),
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_schedulers=(shared_lr_scheduler, controller_lr_scheduler),
                          logger=logger
                          )
    trainer.train()


if __name__ == '__main__':
    # 使用 argparse 库来解析命令行参数
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    # 自定义命令行选项，用于修改 JSON 文件中给定的默认配置
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    # 使用 ConfigParser 从命令行参数解析配置
    config = ConfigParser.from_args(args, options)
    main(config)
