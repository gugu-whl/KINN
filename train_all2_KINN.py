# encoding=utf8
## train files
from densenet_ori_train import densenet_ori_train
from densenet_iccnn_train import densenet_single_train
from densenet_iccnn_multi_train import densenet_multi_train
#from vgg_train import
#from resnet_train import
#resnet
from resnet_iccnn_multi_train import resnet_multi_train
from resnet_iccnn_train2_KINN import resnet_single_train
from resnet_ori_train import resnet_ori_train ###
#vgg
from vgg_iccnn_train import vgg_single_train
from vgg_iccnn_multi_train import vgg_multi_train
from vgg_ori_train import vgg_ori_train
##
import argparse
import random
import os
import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    torch.set_num_threads(5)    

    set_seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # torch.backends.cudnn.enable = True
    # torch.backends.cudnn.benchmark = True
    # 打印全部数据
    # torch.set_printoptions(threshold=np.inf)

    resnet_single_train()

