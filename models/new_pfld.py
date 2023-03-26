#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import math


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


def conv_bn(inp, oup, kernel, stride, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        layers = [ConvBNReLU(inp, inp * expand_ratio, 1, 1),
                  ConvBNReLU(inp * expand_ratio, inp * expand_ratio, 3, stride=stride, groups=inp * expand_ratio)]
        layers.extend([nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0),
                       nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = ConvBNReLU(3,
                                64,
                                kernel_size=3,
                                stride=2)

        self.conv2 = ConvBNReLU(64,
                                64,
                                kernel_size=3,
                                groups=64,
                                stride=1)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = ConvBNReLU(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112
        x = self.conv1(x)  # [64, 56, 56]
        x = self.conv2(x)  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)

        x = self.conv4_1(x)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)  # 16 14 14
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)  # 32 7 7
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))  # 128 1 1
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# if __name__ == '__main__':
#     input = torch.randn(1, 3, 112, 112)
#     pfld_backbone = PFLDInference()
#     landmarks = pfld_backbone(input)
#
#     print("landmarks.shape: {0:}".format(
#         landmarks.shape))
