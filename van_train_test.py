import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import resnet_iccnn_multi_train
from resnet_iccnn_multi_train import ResNet
from resnet_iccnn_multi_train import get_Data
import time

'''
定义运行的设备类型
如果有多张显卡，需要通过cuda:n指定具体某张显卡
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 利用DataLoader来加载数据集
train_dataloader, test_dataloader = get_Data(True, 'voc_multi', 16)

# 创建网络模型
myTrainModel = ResNet(resnet_iccnn_multi_train.BasicBlock,[2,2,2,2])
myTrainModel = myTrainModel.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 创建优化器
# 1e-2 == 1 * (10)^(-2) = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(myTrainModel.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加writer
writer = SummaryWriter("logs/van_train")

# 循环开始时间
start_time = time.time()

for i in range(epoch):

    # 训练步骤开始
    print("----------------第{}轮训练开始----------------".format(i + 1))
    # 进入训练模式(实际只针对部分模型需要，非必须)
    myTrainModel.train()
    for images, targets in train_dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs, f_map, corre = myTrainModel(images, eval=False)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 清零梯度
        optimizer.zero_grad()
        # 计算每个参数对应的梯度
        loss.backward()
        # 执行优化
        optimizer.step()

        # 打印结果
        total_train_step += 1
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        # 避免打印过多，100次打印一次
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))

    # 验证步骤开始
    # 进入验证模式(实际只针对部分模型需要，非必须)
    myTrainModel.eval()
    # 没一个轮次的总偏差
    total_test_loss = 0.0
    # 总体正确率
    total_accuracy = 0
    # 没有梯度，即验证过程不进行调优
    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = myTrainModel(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            '''
                outputs.argmax(1) 每一组横向比较大小，返回最大的一个数的下标
                outputs.argmax(1) == targets 与目标逐项进行比较判断，得到一个布尔值组成的矩阵
                (outputs.argmax(1) == targets).sum() 计算矩阵的True值个数
            '''
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            # total_test_step += 1
            # print("验证次数：{}，Loss：{}".format(total_test_step, loss.item()))
    print("整体测试集上的Loss：{}".format(total_test_loss))
    #print("整体测试集上的正确率：{}".format(total_accuracy/test_data_length))
    writer.add_scalar("test_total_loss", total_test_loss, total_test_step)
    #writer.add_scalar("test_accuracy", total_accuracy/test_data_length, total_test_step)
    total_test_step += 1

    # 模型保存方式1
    # torch.save(myTrainModel, "./model/my_train_model_{}.pth".format(i+1))
    # 模型保存方式2，只保存状态值（官方推荐）
    torch.save(myTrainModel.state_dict(), "./model/my_train_model2_{}.pth".format(i+1))

print("{}轮训练总耗时：{}分钟".format(epoch, (time.time() - start_time)/60))
writer.close()
