import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CAAttention import *
from models.cbam import *
from models.SimAM import *

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

#SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Bottle2neck(nn.Module):
    expansion = 4  #残差块的输出通道数=输入通道数*expansion
    def __init__(self, inplanes, planes, stride=1,downsample=None,  scales=4, groups=1, se=True,  norm_layer=True):
        #scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Bottle2neck, self).__init__()

        if planes % scales != 0: #输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  #BN层
            norm_layer = nn.BatchNorm2d

        bottleneck_planes = groups * planes
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        #1*1的卷积层,在第二个layer时缩小图片尺寸
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_planes)
        #3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        #1*1的卷积层，经过这个卷积层之后输出的通道数变成
        self.conv3 = nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #SE模块
        self.se = SimAM(planes * self.expansion) if se else None

    def forward(self, x):
        identity = x

        #1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, 1) #将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        #1*1的卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        #加入SE模块
        if self.se is not None:
            out = self.se(out)
        #下采样
        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.inplanes = 64
        self.baseWidth = 32
        self.scale = 4

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.layer1 = self._make_layer(Bottle2neck, 64, 3)
        self.layer2 = self._make_layer(Bottle2neck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottle2neck, 256, 9, stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)

        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.layer1(x)

        features.append(x)

        x = self.layer2(x)

        features.append(x)

        x = self.layer3(x)
        return x, features[::-1]