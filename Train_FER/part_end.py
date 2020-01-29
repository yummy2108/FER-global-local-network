#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from network import Conv2d
from torch_deform_conv.layers import ConvOffset2D

class back_end_deform(nn.Module):
    def __init__(self, n_classes):
        super(back_end_deform, self).__init__()
        self.conv1 = nn.Sequential( ConvOffset2D(512),
                                      Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1,  NL='relu'))
        self.conv2 = nn.Sequential( ConvOffset2D(256),
                                      Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1,  NL='relu'))
        self.conv3 = nn.Sequential( ConvOffset2D(128),
                                      Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1,  NL='relu'))
        self.conv4 = Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv5 = Conv2d(32, 16, 1, stride=1, padding=0)
        self.conv6 = Conv2d(16, n_classes, 1, stride=1, padding=0)

        self.conv7 = Conv2d(3, 32, 1, stride=1, padding=0)
        self.conv8 = Conv2d(32, 16, 1, stride=1, padding=0)
        self.conv9 = Conv2d(16, n_classes, 1, stride=1, padding=0)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.OutputActivation = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, face, local):
        out1_1 = self.conv1(face)
        out1_2 = self.conv2(out1_1)
        out1_3 = self.conv3(out1_2)
        out1_4 = self.conv4(out1_3)
        out1_5 = self.conv5(out1_4)
        out1_6 = self.conv6(out1_5)
        
        out2_1 = self.conv7(local)
        out2_2 = self.conv8(out2_1)
        out2_3 = self.conv9(out2_2)

        out1 = self.agp(out1_6)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.agp(out2_3)
        out2 = out2.view(out2.size(0), -1)

        return out1, out2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()