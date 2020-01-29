#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from network import Conv2d
import front_end
from part_end import back_end_deform
from eye_end import eye_end
from nose_end import nose_end
from mouth_end import mouth_end
import cv2
import numpy as np

class GLENet(nn.Module):
    def __init__(self, n_classes):
        super(GLENet, self).__init__()
        self.front_end = front_end.fornt_end()
        for param in self.front_end.parameters():
            param.requires_grad = True


        self.eye_end = eye_end()
        for param in self.eye_end.parameters():
            param.requires_grad = True

        self.nose_end = nose_end()
        for param in self.nose_end.parameters():
            param.requires_grad = True

        self.mouth_end = mouth_end()
        for param in self.mouth_end.parameters():
            param.requires_grad = True

        self.back_end = back_end_deform(n_classes)
        for param in self.back_end.parameters():
            param.requires_grad = True

        #self.ip1 = nn.Linear(n_classes,2)

        self.n_classes = n_classes

    def forward(self, face1, face2, face3, eye1, eye2, eye3, nose1, nose2, nose3, mouth1, mouth2, mouth3):
        #f1 = self.front_end(face1)
        #f2 = self.front_end(face2)
        f3 = self.front_end(face3)

        e1 = self.front_end(eye1)
        e2 = self.front_end(eye2)
        e3 = self.front_end(eye3)

        n1 = self.front_end(nose1)
        n2 = self.front_end(nose2)
        n3 = self.front_end(nose3)

        m1 = self.front_end(mouth1)
        m2 = self.front_end(mouth2)
        m3 = self.front_end(mouth3)

        e_feature = self.eye_end(e1,e2,e3)
        n_feature = self.nose_end(n1,n2,n3)
        m_feature = self.mouth_end(m1,m2,m3)
        
        local_feature = torch.cat((e_feature, n_feature, m_feature), 1)

        out1, out2 = self.back_end(f3, local_feature)
        #ip1 = self.ip1(out)

        return out1, out2, f3
    
