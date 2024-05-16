import os
import re
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader

class Celeb(Dataset):
    def __init__(self, data_file, dst_path='cropped_CelebA', training=True, transform=None, train_num=16000):
        src_path = data_file + 'CelebA_info'
        if train_num == 10240:
            category = 'celeb_sample_10240.txt'
        else:
            category = 'list_attr_celeba.txt'
        fn = open(src_path + '/Anno/' + category, 'r')
        fh2 = open(src_path + '/Eval/list_eval_partition.txt', 'r')
        imgs = []
        lbls = []
        ln = 0
        train_bound = 162770 + 2
        test_bound = 182638 + 2
        regex = re.compile('\s+')
        for line in fn:
            ln += 1
            if ln <= 2:
                continue
            if ln < test_bound and not training:
                continue
            if (ln - 2 <= train_num and training and ln <=train_bound) or\
                (ln - test_bound < train_num  and not training):
                line = line.rstrip('\n')
                line_value = regex.split(line)
                imgs.append(line_value[0])
                lbls.append(list(int(i) if int(i) > 0 else 0 for i in line_value[1:]))
        self.imgs = imgs
        self.lbls = lbls
        self.is_train = training
        self.dst_path = data_file + dst_path
        if transform is None:
            if training:
                self.transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ])
            else:
                self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ])
        else:
            self.transform = transform

    def __getitem__(self, idx):
        fn = self.imgs[idx]
        lbls = self.lbls[idx]
        if self.is_train:
            imgs = default_loader(self.dst_path + '/train/' + fn)
        else:
            imgs = default_loader(self.dst_path + '/test/' + fn)
        imgs = self.transform(imgs)
        lbls = torch.Tensor(lbls)
        return [imgs, lbls]

    def __len__(self):
        return len(self.imgs)

def sample_celeb(data_file, category='list_attr_celeba.txt', training=True, sample_num=10240, train_num=162770):
    src_path = data_file + 'CelebA_info'
    fn = open(src_path + '/Anno/' + category, 'r')
    sample_path = src_path + '/Anno/celeb_sample_'+str(sample_num)+'.txt'
    if os.path.exists(sample_path):
        os.system('rm '+ sample_path)
    sample_fh = open(sample_path, 'w')
    ln = 0
    train_bound = 162770 + 2
    test_bound = 182638 + 2
    regex = re.compile('\s+')
    content = []
    trainnum_list = np.arange(0, train_bound-2)
    sample_num_list = random.sample(trainnum_list.tolist(), sample_num)
    for line in fn:
        ln += 1
        if ln <= 2:
            sample_fh.write(line)
        if ln < test_bound and not training:
            continue
        if (ln - 2 <= train_num and training and ln <=train_bound) or\
            (ln - test_bound < train_num  and not training):
            content.append(line)
    
    for idx in sample_num_list:
        sample_fh.write(content[idx])
    sample_fh.close()

if __name__ == '__main__':
    data_file = '/home/wzh/project/fjq/dataset/CelebA/'
    sample_celeb(data_file, sample_num=10240)
"""
这段代码定义了一个Celeb继承自DatasetPyTorch 框架类的类。它用于加载和预处理深度学习模型的 CelebA 数据集。该类将数据集的文件路径、保存裁剪图像的目标路径、指示数据集是用于训练还是测试的布尔标志、应用于图像的转换函数以及训练示例的数量作为输入。使用。

该__init__函数通过打开CelebA数据集的属性文件和分区文件来初始化数据集，然后读取图像的文件名和标签。它还设置了应用于图像的变换函数。

该__getitem__函数检索与给定索引对应的图像和标签。default_loader它首先使用给定代码中未定义的函数加载图像，然后将转换函数应用于图像。

该__len__函数返回数据集中的图像数量。

该sample_celeb函数用于从 CelebA 数据集中随机抽取图像子集。它以数据集的文件路径、属性文件的名称、指示数据集是用于训练还是测试的布尔标志、要采样的图像数以及要使用的训练示例数作为输入。它从属性文件中随机选择一个图像子集并将它们写入一个新文件。
"""