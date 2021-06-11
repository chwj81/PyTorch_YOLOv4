import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.models import *
from utils.datasets import create_dataloader
from utils.general import (
    check_img_size, torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors,
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results,
    get_latest_run, check_git_status, check_file, increment_dir, print_mutation, plot_evolution)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov4', help='initial model type')
    parser.add_argument('--nc', type=int, default=2, help='initial class size')
    parser.add_argument('--size', type=int, default=320, help='initial image size')
    opt = parser.parse_args()

    #print(select_device('', batch_size=32))
    model = './cfg/' + opt.model + ".cfg"
    nc = opt.nc
    size = opt.size
    net = Darknet(model).to('cpu')
    #net = model(model, ch=3, nc=nc, anchors=None).to('cpu')
    macs, params = get_model_complexity_info(net, (3, size, size), as_strings=True, print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

