import torch.nn as nn
from newPad2d import newPad2d

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)  # new padding


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MySequential(nn.Sequential):
    def forward(self, input, array = None):
        #input_np = input.detach().cpu().numpy()
        #print(input)
        for module in self:
            # TODO 如果目标是MySequential，则传入array
            if isinstance(module, MySequential):
                input = module(input, array)
            # TODO 如果是几种能处理的处理类型（Conv，ZeroPad2D、MaxPool、AvgPool），则正常处理，将结果append进数组
            elif isinstance(module, nn.Conv2d) or isinstance(module, newPad2d):
                input = module(input)
                if array is not None:
                    array.append(input.detach().cpu().numpy())
            # TODO 如果是Bottleneck？
            elif isinstance(module, Bottleneck):
                input = module(input, array)
            else:
                input = module(input)
            # output_np = input.detach().cpu().numpy()
            # print(output_np)
        return input


class Bottleneck(nn.Module):
    # class attribute
    expansion = 4
    num_layers = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1)  # new paddig

    def forward(self, x, array = []):
        identity = x
        out = self.conv1(x)
        array.append(out.detach().cpu().numpy())

        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out)  # new padding
        array.append(out.detach().cpu().numpy())

        out = self.conv2(out)
        array.append(out.detach().cpu().numpy())

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        array.append(out.detach().cpu().numpy())

        out = self.bn3(out)

        if self.downsample is not None:
            # 这一部分不在纳入array数组
            # if isinstance(self.downsample, MySequential):
            #     identity = self.downsample(x, array)
            # else:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [1, 3, 1]
        block_strides = [1, self.stride, 1]
        block_paddings = [0, 1, 0]

        return block_kernel_sizes, block_strides, block_paddings