# encoding=utf8
## train files
import resnet_iccnn_train2_KINN
import resnet_iccnn_train2_KINN_NOknowledeembeding
import resnet_iccnn_train2_KINN_NONMF
import resnet_iccnn_train2_KINN_NONMF_NOknowledeembeding


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

    print("=================resnet_iccnn_train2_KINN begin======================")
    resnet_iccnn_train2_KINN.resnet_single_train()
    print("=================resnet_iccnn_train2_KINN end======================")

    print("=================resnet_iccnn_train2_KINN_NOknowledeembeding begin======================")
    resnet_iccnn_train2_KINN_NOknowledeembeding .resnet_single_train()
    print("=================resnet_iccnn_train2_KINN_NOknowledeembeding end======================")

    print("=================resnet_iccnn_train2_KINN_NONMF begin======================")
    resnet_iccnn_train2_KINN_NONMF.resnet_single_train()
    print("=================resnet_iccnn_train2_KINN_NONMF end======================")

    print("=================resnet_iccnn_train2_KINN_NONMF_NOknowledeembeding begin======================")
    resnet_iccnn_train2_KINN_NONMF_NOknowledeembeding.resnet_single_train()
    print("=================resnet_iccnn_train2_KINN_NONMF_NOknowledeembeding end======================")

