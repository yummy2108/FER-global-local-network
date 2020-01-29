#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import front_end
from network import Conv2d
import front_end
from LSTM import Rnn

class mouth_end(nn.Module):
    def __init__(self):
        super(mouth_end, self).__init__()

        self.conv1 = Conv2d(512, 256, 1, stride=1, padding=0)
        self.conv2 = Conv2d(256, 128, 1, stride=1, padding=0)
        self.conv3 = Conv2d(128, 1, 1, stride=1, padding=0)
        self.Rnn = Rnn(32,256,3)
        for param in self.Rnn.parameters():
            param.requires_grad = True


    def forward(self, m1,m2,m3):
        w = m1.shape[2]
        h = m1.shape[3]
        out1 = self.conv1(m1)
        out1_1 = self.conv2(out1)
        out1_2 = self.conv3(out1_1)
        out1_3 = out1_2.view(-1,1,w*h)

        out2 = self.conv1(m2)
        out2_1 = self.conv2(out2)
        out2_2 = self.conv3(out2_1)
        out2_3 = out2_2.view(-1,1,w*h)

        out3 = self.conv1(m3)
        out3_1 = self.conv2(out3)
        out3_2 = self.conv3(out3_1)
        out3_3 = out3_2.view(-1,1,w*h)

        feature = torch.cat((out1_3, out2_3, out3_3), 1)
        out = self.Rnn(feature)
        out = out.view(-1,1,16,16)

        return out