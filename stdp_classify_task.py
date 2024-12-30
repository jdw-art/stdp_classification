import argparse
import json
import os
import random
import socket
import sys
import time
import warnings
from datetime import datetime
from math import ceil, floor
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

import utils.checkpoint
import utils.meters
import utils.metrics
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import STDPClassifyExcInhNet
from models.neuron_models import IafPscDelta
from models.spiking_model import SpikingProtoNet, LatencyEncodeNet

# from models.spiking_model_4conv import SpikingProtoNet

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='STDP Classification task training')
parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                    help='Number of time steps for each item in the sequence (default: 100)')

parser.add_argument('--dir', default='./data', type=str, metavar='DIR',
                    help='Path to dataset (default: ./data)')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers; set to 0 due to pickling issue (default: 4)')
parser.add_argument('--prefetch_factor', default=2, type=int, metavar='N',
                    help='Prefetch prefetch_factor * workers examples (default: 2)')
parser.add_argument('--pin_data_to_memory', default=1, choices=[0, 1], type=int, metavar='PIN_DATA_TO_MEMORY',
                    help='Pin data to memory (default: 1)')

parser.add_argument('--embedding_size', default=64, type=int, metavar='N',
                    help='Embedding size (default: 64)')
parser.add_argument('--memory_size', default=100, type=int, metavar='N',
                    help='Size of the memory matrix (default: 100)')
parser.add_argument('--input_size', default=784, type=int, metavar='N',
                    help='Embedding size (default: 64)')
parser.add_argument('--hidden_size', default=400, type=int, metavar='N',
                    help='Embedding size (default: 64)')
parser.add_argument('--output_size', default=100, type=int, metavar='N',
                    help='Size of the output layer (default: 784)')
parser.add_argument('--w_max', default=1.0, type=float, metavar='N',
                    help='Soft maximum of Hebbian weights (default: 1.0)')
parser.add_argument('--gamma_pos', default=0.3, type=float, metavar='N',
                    help='Write factor of Hebbian rule (default: 0.3)')
parser.add_argument('--gamma_neg', default=0.3, type=float, metavar='N',
                    help='Forget factor of Hebbian rule (default: 0.3)')
parser.add_argument('--tau_trace', default=20.0, type=float, metavar='N',
                    help='Time constant of key- and value-trace (default: 20.0)')
parser.add_argument('--readout_delay', default=1, type=int, metavar='N',
                    help='Synaptic delay of the feedback-connections from value-neurons to key-neurons in the '
                         'reading layer (default: 1)')

parser.add_argument('--thr', default=0.05, type=float, metavar='N',
                    help='Spike threshold (default: 0.05)')
parser.add_argument('--perfect_reset', action='store_true',
                    help='Set the membrane potential to zero after a spike')
parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                    help='The number of time steps the neuron is refractory (default: 3)')
parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                    help='Neuron membrane time constant (default: 20.0)')
parser.add_argument('--dampening_factor', default=1.0, type=float, metavar='N',
                    help='Scale factor for spike pseudo-derivative (default: 1.0)')

parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='Number of total epochs to run (default: 120)')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='Mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning_rate', default=0.001, type=float, metavar='N',
                    help='Initial learning rate (default: 0.001)')
parser.add_argument('--learning_rate_decay', default=0.85, type=float, metavar='N',
                    help='Learning rate decay (default: 0.85)')
parser.add_argument('--decay_learning_rate_every', default=20, type=int, metavar='N',
                    help='Decay the learning rate every N epochs (default: 20)')
parser.add_argument('--max_grad_norm', default=40.0, type=float, metavar='N',
                    help='Gradients with an L2 norm larger than max_grad_norm will be clipped '
                         'to have an L2 norm of max_grad_norm. If None, then the gradient will '
                         'not be clipped. (default: 40.0)')
parser.add_argument('--l2', default=1e-7, type=float, metavar='N',
                    help='L2 rate regularization factor (default: 1e-7)')
parser.add_argument('--target_rate', default=0.0, type=float, metavar='N',
                    help='Target firing rate in Hz for L2 regularization (default: 0.0)')

parser.add_argument('--image_protonet_path', default='results/protonet_checkpoints/Oct26_20-06-51_bicserver'
                                                     '-MNIST_classification_task_best.pth.tar',
                    type=str, metavar='PATH', help='Path to the MNIST ProtoNet checkpoint (default: none)')
parser.add_argument('--freeze_protonet', default=0, choices=[0, 1], type=int,
                    help='Freeze pre-trained ProtoNets after conversion (default: 0)')
parser.add_argument('--learn_if_thresholds', action='store_true',
                    help='Learn IF neuron thresholds of the spiking CNN via threshold balancing (if never done yet)')
parser.add_argument('--fix_cnn_thresholds', action='store_false', help='Do not adjust firing threshold after conversion'
                                                                       '(default: will adjust v_th via a bias)')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='Manual epoch number (useful on restarts, default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate the model on the test set')

parser.add_argument('--logging', action='store_true',
                    help='Write tensorboard logs')
parser.add_argument('--print_freq', default=1, type=int, metavar='N',
                    help='Print frequency (default: 1)')
parser.add_argument('--world_size', default=-1, type=int, metavar='N',
                    help='Number of nodes for distributed training (default: -1)')
parser.add_argument('--rank', default=-1, type=int, metavar='N',
                    help='Node rank for distributed training (default: -1)')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, metavar='DIST_URL',
                    help='URL used to set up distributed training (default: tcp://127.0.0.1:23456)')
parser.add_argument('--dist_backend', default='nccl', choices=['nccl', 'mpi', 'gloo'], type=str, metavar='DIST_BACKEND',
                    help='Distributed backend to use (default: nccl)')
parser.add_argument('--seed', default=None, type=int, metavar='N',
                    help='Seed for initializing training (default: none)')
parser.add_argument('--data_seed', default=None, type=int, metavar='N',
                    help='Seed for the dataset (default: none)')
parser.add_argument('--gpu', default=None, type=int, metavar='N',
                    help='GPU id to use (default: none)')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--check_params', default=1, type=int, choices=[0, 1], metavar='CHECK_PARAMS',
                        help='When loading from a checkpoint check if the model was trained with the same parameters '
                             'as requested now (default: 1)')
parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')

best_loss = np.inf
log_dir = ''
writer = None
time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')
suffix = ''

with open('version.txt') as f:
    version = f.readline()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    num_gpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, num_gpus_per_node, args)


def main_worker(gpu, num_gpus_per_node, args):
    global best_loss
    global log_dir
    global writer
    global time_stamp
    global version
    global suffix
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Data loading code

    train_set = torchvision.datasets.MNIST(root=args.dir, train=True, download=True,
                                               transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(root=args.dir, train=False, download=True, transform=transforms.ToTensor())

    ## 训练集和测试集标签
    train_targets = train_set.targets
    test_targets = test_set.targets

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_data_to_memory,
        prefetch_factor=args.prefetch_factor)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_data_to_memory,
        prefetch_factor=args.prefetch_factor)

    # Create ProtoNet
    image_embedding_layer = LatencyEncodeNet(IafPscDelta(thr=args.thr,
                                                         perfect_reset=args.perfect_reset,
                                                         refractory_time_steps=args.refractory_time_steps,
                                                         tau_mem=args.tau_mem,
                                                         spike_function=SpikeFunction,
                                                         dampening_factor=args.dampening_factor),
                                             output_size=args.input_size,
                                             num_time_steps=args.num_time_steps,
                                             refractory_time_steps=args.refractory_time_steps)

    # Create model
    print("=> creating model '{model_name}'".format(model_name=STDPClassifyExcInhNet.__name__))
    model = STDPClassifyExcInhNet(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        readout_delay=args.readout_delay,
        batch_size=args.batch_size,
        tau_trace=args.tau_trace,
        image_embedding_layer=image_embedding_layer,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg),
        dynamics=IafPscDelta(thr=args.thr,
                             perfect_reset=args.perfect_reset,
                             refractory_time_steps=args.refractory_time_steps,
                             tau_mem=args.tau_mem,
                             spike_function=SpikeFunction,
                             dampening_factor=args.dampening_factor))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cpu')
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            log_dir = checkpoint['log_dir']
            time_stamp = checkpoint['time_stamp']

            # Checkpoint parameters have to match current parameters. If not, abort.
            ignore_keys = ['workers', 'prefetch_factor', 'pin_data_to_memory', 'epochs', 'start_epoch', 'resume',
                           'evaluate', 'logging', 'print_freq', 'world_size', 'rank', 'dist_url', 'dist_backend',
                           'seed', 'gpu', 'multiprocessing_distributed', 'distributed', 'dir']
            if args.evaluate:
                ignore_keys.append('batch_size')

            for key, val in vars(checkpoint['params']).items():
                if key not in ignore_keys:
                    if vars(args)[key] != val:
                        print("=> You tried to restart training of a model that was trained with different parameters "
                              "as you requested now. Aborting...")
                        sys.exit()

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # load checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load checkpoint
    if args.checkpoint_path:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = utils.checkpoint.load_checkpoint(args.checkpoint_path, device)
        if args.check_params:
            for key, val in vars(args).items():
                if key not in ['check_params', 'seed', 'data_seed', 'checkpoint_path']:
                    if vars(checkpoint['params'])[key] != val:
                        print("=> You tried to load a model that was trained on different parameters as you requested "
                              "now. You may disable this check by setting `check_params` to 0. Aborting...")
                        sys.exit()

        new_state_dict = OrderedDict()
        # print("checkpoint['state_dict']:", checkpoint['state_dict'])
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[len('module.'):]  # remove `module.`
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # input_weight_clone = model.classify_layer.input_weight.clone().detach().to('cpu').numpy()
    # hidden_weight_clone = model.classify_layer.hidden_weight.clone().detach().to('cpu').numpy()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and
                                                args.rank % num_gpus_per_node == 0):
        if log_dir and args.logging:
            # Use the directory that is stored in checkpoint if we resume training
            writer = SummaryWriter(log_dir=log_dir)
        elif args.logging:
            log_dir = os.path.join('results', 'auto_encoder', 'stdp_classify', 'logs', time_stamp +
                                   f'_thr-{args.thr}-{suffix}_attention_mnist_memory')
            writer = SummaryWriter(log_dir=log_dir)

            def pretty_json(hp):
                json_hp = json.dumps(hp, indent=2, sort_keys=False)
                return "".join('\t' + line for line in json_hp.splitlines(True))

            writer.add_text('Info/params', pretty_json(vars(args)))
    print("\n")
    print("=> Training..........................\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}...")
        train(train_loader, model, criterion, args, is_train=True)

    print("\n")
    print("=> Testing...........................\n")
    train_output = train(train_loader, model, criterion, args, is_train=False)
    test_output = validate(test_loader, model, criterion, args)

    output_train_max = train_output.max(1).values.to('cpu').numpy()
    output_train_sum = train_output.sum(1).to('cpu').numpy()

    output_test_max = test_output.max(1).values.to('cpu').numpy()
    output_test_sum = test_output.sum(1).to('cpu').numpy()

    # STDP分类参数==========================================================================
    print("\n")
    print("Begin to classification.......................")
    seed = 1
    # 创建分类器
    clf = LinearSVC(max_iter=3000, random_state=seed)
    # 使用训练集的输出 output_train_max 和对应的标签 y_train 来训练分类器。
    train_length = output_train_max.shape[0]
    clf.fit(output_train_max, train_targets[:train_length])
    # 使用训练好的分类器对测试集的输出 output_test_max 进行预测，并将预测结果保存在变量 y_pred 中
    test_length = output_test_max.shape[0]
    y_pred = clf.predict(output_test_max)
    # 使用最大值读出分类的准确定（方法一）
    acc = accuracy_score(test_targets[:test_length], y_pred)
    print("\n")
    print(f"Accuracy with method 1 (max) : {acc}")

    clf = LinearSVC(max_iter=3000, random_state=seed)
    clf.fit(output_train_sum, train_targets[:train_length])
    y_pred = clf.predict(output_test_sum)
    # 通过计算输出值之和计算准确率
    acc = accuracy_score(test_targets[:test_length], y_pred)
    print("\n")
    print(f"Accuracy with method 2 (sum) : {acc}")

    if args.logging:
        print("\n")
        print("saving checkpoint....................\n")
        utils.checkpoint.save_checkpoint({
            'state_dict': model.state_dict(),
            'time_stamp': time_stamp,
            'params': args
        }, True, filename=os.path.join(
            'results', 'auto_encoder', 'stdp_classify',
            time_stamp + '_' + f'_thr-{args.thr}-{suffix}_classification' + f'_times-{args.num_time_steps}'))
        print("\n")
        print("Saving completed.....................\n")

def train(data_loader, model, criterion, args, is_train):

    # Switch to evaluate mode
    model.eval()
    spk_output = torch.empty(0, args.num_time_steps, args.output_size).cuda(args.gpu)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            images, labels = sample

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)

            # Compute output
            output = model(images, train = is_train)
            spk_output = torch.cat((spk_output, output), dim=0)

    return spk_output

def validate(data_loader, model, criterion, args):

    # Switch to evaluate mode
    model.eval()
    spk_output = torch.empty(0, args.num_time_steps, args.output_size).cuda(args.gpu)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            images, labels = sample

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)

            # Compute output
            output = model(images, train = False)
            spk_output = torch.cat((spk_output, output), dim=0)

    return spk_output


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by X% every N epochs"""
    lr = args.learning_rate * (args.learning_rate_decay ** (epoch // args.decay_learning_rate_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
