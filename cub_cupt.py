import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import numpy as np
# object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
#                      'bottle', 'bus', 'car', 'cat', 'chair',
#                      'cow', 'diningtable', 'dog', 'horse',
#                      'motorbike', 'person', 'pottedplant',
#                      'sheep', 'sofa', 'train', 'tvmonitor']
object_categories = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']

class CUB_CUPT(Dataset):

    def __init__(self, root, dataname, mytype, train=True, knowledge = False, transform=None, loader=default_loader, is_frac=None, sample_num=-1):
        self.root = os.path.expanduser(root)
        self.dataname = dataname
        self.mytype = mytype
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.is_frac = is_frac
        self.sample_num = sample_num
        self.knowledge = knowledge
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        data_txt = 'info.txt'

        self.data = pd.read_csv(os.path.join(self.root, data_txt),
                             names=['img_id','file_path','m','target','is_training_img'])

        if self.knowledge:
            self.data = self.data[self.data.is_training_img == 2]
        else:
            # 根据是否训练的参数过滤训练和验证数据集
            if self.train:
                self.data = self.data[self.data.is_training_img == 1]
            else:
                self.data = self.data[self.data.is_training_img == 0]

        # 特征数据集只保留target=0的图片
        if self.is_frac is not None:
            self.data = self.data[self.data.target == self.is_frac]

        if self.sample_num != -1:
            self.data = self.data[0:self.sample_num]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, row.file_path)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.file_path)
        target = sample.target  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
"""
此代码定义了一个 PyTorch 数据集类，用于CUB_VOC加载图像数据集。它采用各种参数，例如 root、dataname、mytype、train、transform、loader、is_frac 和 sample_num，并实现 和__len__方法__getitem__。

该__init__方法通过设置根目录、数据集名称、类型、转换函数、加载器、训练标志、is_frac 和 sample_num 来初始化数据集。该_load_metadata方法从文本文件加载数据集的元数据信息，并根据训练标志和 is_frac 参数过滤数据。该_check_integrity方法通过确保所有图像文件都存在来检查数据集的完整性。该__len__方法返回数据集中的图像数量，该__getitem__方法加载图像及其对应的标签，将变换函数应用于图像，并将图像和标签作为元组返回。

似乎此实现旨在处理不同类型的数据集，例如 CUB、VOC 和 Helen，并且可以根据数据是用于训练还是验证来过滤数据。它还可用于对数据进行子采样并应用图像变换。
"""