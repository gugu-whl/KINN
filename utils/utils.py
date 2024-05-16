#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch

class EMA_FM():
    def __init__(self, decay=0.9, first_decay=0.0, channel_size=512, f_map_size=196, is_use = False):
        self.decay = decay
        self.first_decay = first_decay
        self.is_use = is_use
        self.shadow = {}
        self.epsional = 1e-5
        if is_use:
            self._register(channel_size=channel_size, f_map_size= f_map_size)

    def _register(self, channel_size=512, f_map_size=196):
        Init_FM = torch.zeros((f_map_size, channel_size),dtype=torch.float)
        self.shadow['FM'] = Init_FM.cuda().clone()
        self.is_first = True

    def update(self, input):
        B, C, _ = input.size()
        if not(self.is_use):
            return torch.ones((C,C), dtype=torch.float)
        decay = self.first_decay if self.is_first else self.decay
        ####### FEATURE SIMILARITY MATRIX EMA ########
        # Mu = torch.mean(input,dim=0)   
        self.shadow['FM'] = (1.0 - decay) * torch.mean(input,dim=0) + decay * self.shadow['FM']
        self.is_first = False
        return self.shadow['FM']
    """
     此代码定义了一个类EMA_FM，代表特征图的指数移动平均线。此类的目的是计算给定输入张量input沿批次维度的特征图的指数移动平均数 (EMA)。这是使用 EMA 的公式完成的：

EMA(t) = (1 - decay) * x(t) + decay * EMA(t-1)

其中decay是控制衰减率的超参数，x(t)是时间 t 的特征图，EMA(t-1)是特征图到时间 t-1 的 EMA。

该类有几个属性，包括decay（衰减率）、first_decay（用于第一次迭代的衰减率）、channel_size（输入张量中的通道数）、f_map_size（特征图的大小）和is_use（指示是否是否使用 EMA）。

该类还有几个方法，包括__init__、_register和update。

__init__初始化类的属性并调用方法_register来初始化 EMA 影子变量。

_register为特征图初始化 EMA 阴影变量。

update计算给定输入张量的特征图的 EMA 并返回 EMA。如果is_use为 False，该方法返回一个与协方差矩阵形状相同的张量。如果is_use为 True，则使用上述公式计算 EMA 并将其存储在shadow类的变量中。
    """

class Cluster_loss():
    def __init__(self):
        pass

    def update(self, correlation, loss_mask_num, loss_mask_den, labels):
        batch, channel, _ = correlation.shape#不需要知道图片的大小，只需要知道批次和通道数
        c, _, _ = loss_mask_num.shape#只要通道数
        if labels is not None:
            label_mask = (1 - labels).view(batch, 1, 1)
            ## smg_loss if only available for positive sample
            correlation = correlation * label_mask
        correlation = (correlation / batch).view(1, batch, channel, channel).repeat(c, 1, 1, 1)

        new_Num = torch.sum(correlation * loss_mask_num.view(c, 1, channel, channel).repeat(1, batch, 1, 1),
                            dim=(1, 2, 3))
        new_Den = torch.sum(correlation * (loss_mask_den).view(c, 1, channel, channel).repeat(1, batch, 1, 1),
                            dim=(1, 2, 3))
        ret_loss = -torch.sum(new_Num / (new_Den + 1e-5))
        return ret_loss
    """
    这个函数似乎是在实现多类损失。输入是一个correlation形状(batch, channel, _)为_batchchannel_

labels如果输入不是 None，该函数首先根据输入计算标签掩码。标签掩码是一个形状(batch, 1, 1)为0或1的张量。标签掩码用于将负样本的相关性设置为0。

接下来，相关张量按批量大小归一化并重塑为新形状(1, batch, channel, channel)。c然后张量沿第一个维度重复多次，其中c是类的数量。

然后通过分别取相关张量与和张量的元素乘积来计算两个张量new_Num和。这些张量也沿着第二个维度重复多次，以对批次执行求和。生成的张量的形状为.new_Denloss_mask_numloss_mask_denbatch(c,)

最后，损失计算为new_Num除以的负和(new_Den + 1e-5)。损失是标量张量。
    """
# 惩罚输入特征图中向量对之间的相似性
class Multiclass_loss():
    def __init__(self, class_num=None):
        self.class_num = class_num

    def get_label_mask(self, label):
        label = label.cpu().numpy()
        sz = label.shape[0]
        label_mask_num = []
        label_mask_den = []
        for i in range(self.class_num):
            idx = np.where(label == i)[0]
            cur_mask_num = np.zeros((sz, sz))
            cur_mask_den = np.zeros((sz, sz))
            for j in idx:
                cur_mask_num[j][idx] = 1
                cur_mask_den[j][:] = 1
            label_mask_num.append(np.expand_dims(cur_mask_num, 0))
            label_mask_den.append(np.expand_dims(cur_mask_den, 0))
        label_mask_num = np.concatenate(label_mask_num, axis=0)
        label_mask_den = np.concatenate(label_mask_den, axis=0)
        return torch.from_numpy(label_mask_num).float().cuda(), torch.from_numpy(label_mask_den).float().cuda()
"""
这个函数似乎是在实现某种损失函数的更新步骤。让我们逐行浏览函数：
batch，channel，_=correlation.shape：这一行提取相关性张量的批量大小和通道维度。_用于忽略第三个维度，该维度大概表示图像的大小。
c、 _，_=loss_mask_num.shape：这一行提取loss_mask _num张量中的通道数。
如果标签不是None:…：此块检查是否提供了标签。如果是，它将基于标签值创建一个二进制掩码，其中1对应于正样本，0对应于负样本。掩码用于将负样本的相关值设置为0，从而有效地将它们从损失计算中删除。
correlation=（correlation/batch）.view（1，batch，channel，channel）.repaint（c，1，1，l）：这一行通过批量大小对相关性张量进行归一化，然后将其重新整形为四个维度。然后，重复函数沿着第一维度创建该张量的c个副本，有效地为每个通道复制它。

new_Num=torc.sum（correlation*loss_mask_Num.view（c，1，channel，channel）.reeat（1，batch，1，1），dim=（1，2，3））：这条线通过将归一化的相关张量与loss_mask _Num张量逐元素相乘来计算损失函数的分子，loss_mask-Num张量可能包含不同相关值的一些权重。然后在最后三个维度上对得到的张量求和，以获得长度批次的一维张量。视图和重复函数用于确保张量形状与元素乘法相匹配。

new_Den=torc.sum（correlation*（loss_mask_Den）.view（c，1，channel，channel）.reeat（1，batch，1，1），dim=（1，2，3））：这一行以与分子类似的方式计算损失函数的分母，但使用loss_mask _Den张量。

ret_loss=-thorp.sum（new_Num/（new_Den+1e-5））：这一行通过将分子除以分母并在批次维度上求和来计算最终损失值。负号用于将优化问题从最大化转换为最小化。1e-5被加到分母上以避免被零除。

总体而言，该函数似乎是基于两组特征之间的相关性值来计算定制的损失函数，其中相关性值由两个不同的掩码加权。labels参数用于过滤出负样本，并返回最终的损失值。
"""
def update(self, fmap, loss_mask_num, label):
    B, C, _, _ = fmap.shape
    center, _, _ = loss_mask_num.shape
    fmap = fmap.view(1, B, C, -1)
    fmap = fmap.repeat(center, 1, 1, 1)
    mean_activate = torch.mean(torch.matmul(loss_mask_num.view(center, 1, C, C).repeat(1, B, 1, 1), fmap),
                               dim=(2, 3))
    # cosine
    mean_activate = torch.div(mean_activate, torch.norm(mean_activate, p=2, dim=0, keepdim=True) + 1e-5)
    inner_dot = torch.matmul(mean_activate.permute(1, 0), mean_activate).view(-1, B, B).repeat(self.class_num, 1, 1)
    label_mask, label_mask_intra = self.get_label_mask(label)

    new_Num = torch.mean(inner_dot * label_mask, dim=(1, 2))
    new_Den = torch.mean(inner_dot * label_mask_intra, dim=(1, 2))
    ret_loss = -torch.sum(new_Num / (new_Den + 1e-5))
    return ret_loss
"""
此代码定义了一个名为 Multiclass_loss 的类，用于计算多类分类问题的损失。损失基于类的平均激活之间的余弦相似性。以下是代码的工作原理：

init方法接受一个名为 class_num 的参数，它指定分类问题中的类数。此参数存储为实例变量。

get_label_mask 方法将标签张量作为输入并返回两个二进制掩码张量。第一个掩码张量 (label_mask_num) 具有形状 (class_num, batch_size, channel_num, channel_num) 并用于计算损失的分子。第二个掩码张量 (label_mask_den) 具有形状 (class_num, batch_size, channel_num, channel_num)，用于计算损失的分母。掩码张量是通过迭代类并将掩码张量的元素设置为 1 来计算的，其中存在相应的类标签。

更新方法需要三个输入：特征图张量（fmap）、分子掩码张量（loss_mask_num）和标签张量（label）。特征图张量的形状为 (batch_size, channel_num, height, width)，用于计算类的平均激活。分子掩码张量的形状为 (class_num, batch_size, channel_num, channel_num)，用于计算损失的分子。标签张量的形状为 (batch_size,)，用于计算损失的分母。该方法首先通过获取分子掩码张量和特征图张量的逐元素乘积，然后沿通道维度取平均值来计算类的平均激活。然后使用 L2 范数将平均激活归一化为具有单位范数。计算所有类对的平均激活值之间的余弦相似度，并将其存储在形状为 (class_num, batch_size, batch_size) 的 inner_dot 张量中。label_mask 和 label_mask_intra 张量是使用 get_label_mask 方法计算的。最终损失计算为所有类别损失的分子和分母之比的负和。

请注意，此代码假定特征图张量已经标准化为沿通道维度具有单位范数。另请注意，此代码假定标签张量包含从 0 开始的类索引
"""
def Cal_Center(fmap, gt):
    f_1map = fmap.detach().cpu().numpy()
    matrix = gt.detach().cpu().numpy()
    B, C, H, W = f_1map.shape
    cluster = []
    visited = np.zeros(C)
    for i in range(matrix.shape[0]):
        tmp = []
        if(visited[i]==0):
            for j in range(matrix.shape[1]):
                if(matrix[i][j]==1 ):
                    tmp.append(j)
                    visited[j]=1;
            cluster.append(tmp)
    center = []
    for i in range(len(cluster)):
        cur_clustet_fmap = f_1map[:,cluster[i],...]
        cluster_center = np.mean(cur_clustet_fmap,axis=1)
        center.append(cluster_center)
    center = np.transpose(np.array(center),[1,0,2,3])
    center = torch.from_numpy(center).float()
    return center
"""
此函数Cal_Center接受两个输入，一个特征图fmap和一个地面实况gt，并返回地面实况中每个集群的中心。

特征图fmap是一个形状为 (B, C, H, W) 的张量，其中 B 是批量大小，C 是通道数，H 和 W 是特征图的高度和宽度。

ground truthgt是一个二元张量，形状为 (B, K, H, W)，其中 K 是聚类的数量。中的每个元素gt要么为0要么为1，表示对应的像素是否属于该簇。

该函数首先将fmapand转换gt为 numpy 数组并循环遍历基本事实中的每个集群。对于每个簇，它找到属于该簇的像素的索引并提取相应的特征图张量。然后它计算特征图张量沿通道维度的均值以获得聚类的中心。所有集群的中心都沿着通道维度连接起来，并作为张量返回。

请注意，该函数将特征图和地面实况张量从计算图中分离出来，并在使用 numpy 处理它们之前将它们移动到 CPU。生成的中心在返回之前被转换回 PyTorch 张量。
"""

"""
    可追踪的排序方法，即排序后可以
    array: 需要排序的数组
    type: 排序方式 0：正序，1倒序
"""
def traceable_sort(array, type = 0, axis=None):
    if type == 0:
        trace_index = np.argsort(array)
        sorted_arr = np.sort(array, axis=axis)
    else:
        trace_index = np.argsort(-array)
        sorted_arr = -np.sort(-array, axis=axis)

    trace_map = {}
    for source_idx, sorted_idx in enumerate(trace_index):
        trace_map[source_idx] = sorted_idx
    return sorted_arr, trace_map


if __name__ == '__main__':
    source = np.array([3, 2, 6, 4, 6, 8])
    print(source)
    sorted_arr, trace_map = traceable_sort(source, 1)

    print(sorted_arr)
    print(trace_map)

    find_idx = 4
    print('排序后的第' + str(find_idx) + '个元素在原始矩阵中的位置是：' + str(trace_map[find_idx]))
