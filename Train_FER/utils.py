import os
import sys
import math
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
import torch
def mixup_data(x, y, alpha=1.0):
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1
	batch_size = x.size()[0]
	index = torch.randperm(batch_size).cuda()
	mixed_x = lam*x + (1-lam) *x[index,:]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
	return lambda criterion, pred: lam* criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)

def init_params(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.kaiming_normal(m.weight, mode='fan_out')
			if m.bias is not None:
				init.constant(m.bias,0)
		elif isinstance(m, nn.BatchNorm2d):
			init.constant(m.weight,1)
			init.consatnt(m.bias, 0)