#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import h5py
import network
import numpy as np
from GLENet import GLENet
import torch.nn.functional as F
from skimage import transform
from PreProcessing import pickle_2_img
import os
from dataset import Emotion_Loder
from dataset import CK_Emotion_Loder
from dataset import test_Emotion_Loder
from torch.nn import init
from utils import mixup_data
from utils import mixup_criterion
from utils import init_params
import time
import matplotlib.pyplot as plt
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"

root = 'ckplus_with_img_geometry_3frame.pkl'
n_classes = 7

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# training configuration
def train(fold, train_dataset,path, test_dataset):
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    start_step = 0
    end_step = 150
    lr = 0.0001
    disp_interval = 10
    device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
    net = GLENet(n_classes)
    for param in net.parameters():
        param.requires_grad = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    net = net.cuda()
    #net.apply(init_params)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_fun = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    train_loss_graph = []
    test_loss_graph = []
    for epoch in range(start_step, end_step+1):
        net.train()
        step = 0
        train_loss = 0
        test_loss = 0
        running_acc = 0
        total = 0
        
        if epoch == 80:
            lr = lr*0.1
            print('learning rate, ',lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
        
        for index, data in enumerate(train_loader):
            #print(index)
            step = step + 1
            label, x1, x2, x3, e1, e2, e3, n1, n2, n3, m1, m2, m3 = data

            label = label.type(torch.LongTensor)
            label = network.tensor_to_variable(label, is_cuda=True, is_training=True)

            x1 = x1.type(torch.FloatTensor)
            x1 = network.tensor_to_variable(x1, is_cuda=True, is_training=True)
            x2 = x2.type(torch.FloatTensor)
            x2 = network.tensor_to_variable(x2, is_cuda=True, is_training=True)
            x3 = x3.type(torch.FloatTensor)
            x3 = network.tensor_to_variable(x3, is_cuda=True, is_training=True)

            e1 = e1.type(torch.FloatTensor)
            e1 = network.tensor_to_variable(e1, is_cuda=True, is_training=True)
            e2 = e2.type(torch.FloatTensor)
            e2 = network.tensor_to_variable(e2, is_cuda=True, is_training=True)
            e3 = e3.type(torch.FloatTensor)
            e3 = network.tensor_to_variable(e3, is_cuda=True, is_training=True)


            n1 = n1.type(torch.FloatTensor)
            n1 = network.tensor_to_variable(n1, is_cuda=True, is_training=True)
            n2 = n2.type(torch.FloatTensor)
            n2 = network.tensor_to_variable(n2, is_cuda=True, is_training=True)
            n3 = n3.type(torch.FloatTensor)
            n3 = network.tensor_to_variable(n3, is_cuda=True, is_training=True)

            m1 = m1.type(torch.FloatTensor)
            m1 = network.tensor_to_variable(m1, is_cuda=True, is_training=True)
            m2 = m2.type(torch.FloatTensor)
            m2 = network.tensor_to_variable(m2, is_cuda=True, is_training=True)
            m3 = m3.type(torch.FloatTensor)
            m3 = network.tensor_to_variable(m3, is_cuda=True, is_training=True)

            mixup, targets_a, targets_b, lam = mixup_data(x3,label,1)

            out1, out2 = net(x1,x2,mixup,e1,e2,e3,n1,n2,n3,m1,m2,m3)
            #print(out.shape, label.shape)

            loss_f = mixup_criterion(targets_a.reshape(-1),targets_b.reshape(-1),lam)
            loss1 = loss_f(loss_fun, out1)
            loss2 = loss_fun(out2, label.reshape(-1))
            loss = loss1 + loss2
            #loss = loss_fun(out, label.reshape(-1))
            train_loss += loss

            out = out1 + out2
            
            _,pred = torch.max(out,1)
            
            num_correct = 0

            for j in range(label.shape[0]):
                if pred[i] == label[i]:
                    num_correct += 1
            
            #num_correct = torch.sum(pred == label.data)
            running_acc += num_correct
            total += label.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            """
            if step % disp_interval == 0:
                ave_loss = train_loss / (batch_size*(index+1))
                acc = running_acc / (batch_size*(index+1))
                log_text = 'epoch: %4d, step: %4d, loss: %4.6f, Acc: %4.6f' % (epoch+1, step, ave_loss, acc)
                print(log_text)
            """
        acc = float(running_acc) / total
        print('epoch: {}, Loss: {}, Acc:{} , acc:{}'.format(epoch+1, train_loss/(index+1), running_acc, acc))
        train_loss_graph.append(train_loss/(index+1))
        net.eval()
        eval_acc = 0
        for tindex,test_data in enumerate(test_loader):
            tlabel, tx1, tx2, tx3, te1, te2, te3, tn1, tn2, tn3, tm1, tm2, tm3 = test_data
            tlabel = tlabel.type(torch.LongTensor)
            tlabel = network.tensor_to_variable(tlabel, is_cuda=True, is_training=False)

            tx1 = tx1.type(torch.FloatTensor)
            tx1 = network.tensor_to_variable(tx1, is_cuda=True, is_training=False)
            tx2 = tx2.type(torch.FloatTensor)
            tx2 = network.tensor_to_variable(tx2, is_cuda=True, is_training=False)
            tx3 = tx3.type(torch.FloatTensor)
            tx3 = network.tensor_to_variable(tx3, is_cuda=True, is_training=False)

            te1 = te1.type(torch.FloatTensor)
            te1 = network.tensor_to_variable(te1, is_cuda=True, is_training=False)
            te2 = te2.type(torch.FloatTensor)
            te2 = network.tensor_to_variable(te2, is_cuda=True, is_training=False)
            te3 = te3.type(torch.FloatTensor)
            te3 = network.tensor_to_variable(te3, is_cuda=True, is_training=False)


            tn1 = tn1.type(torch.FloatTensor)
            tn1 = network.tensor_to_variable(tn1, is_cuda=True, is_training=False)
            tn2 = tn2.type(torch.FloatTensor)
            tn2 = network.tensor_to_variable(tn2, is_cuda=True, is_training=False)
            tn3 = tn3.type(torch.FloatTensor)
            tn3 = network.tensor_to_variable(tn3, is_cuda=True, is_training=False)

            tm1 = tm1.type(torch.FloatTensor)
            tm1 = network.tensor_to_variable(tm1, is_cuda=True, is_training=False)
            tm2 = tm2.type(torch.FloatTensor)
            tm2 = network.tensor_to_variable(tm2, is_cuda=True, is_training=False)
            tm3 = tm3.type(torch.FloatTensor)
            tm3 = network.tensor_to_variable(tm3, is_cuda=True, is_training=False)

            tout1, tout2 = net(tx3,tx3,tx3,te3,te3,te3,tn3,tn3,tn3,tm3,tm3,tm3)

            tout = tout1 + tout2

            tloss = loss_fun(tout, tlabel.reshape(-1))
            test_loss += float(tloss)
            
            _,tpred = torch.max(tout,1)
            correct = (tpred == tlabel).sum()
            eval_acc += correct
        print('test_acc : {} test_loss: {}'.format(eval_acc, test_loss/ tindex+1))
        test_loss_graph.append(test_loss/ tindex+1)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            torch.save(net.state_dict(), path+'{}_best.pth'.format(best_epoch))

    return best_epoch,best_acc


if __name__ == '__main__':

strat_time = time.time()
test_ep = []
test_res = []
for i in range(10):

    #----------train CK+---------------#
    train_dataset = CK_Emotion_Loder(txt=root,fold=i, transform = transform)

    test_dataset = test_Emotion_Loder(txt=root, fold=i, transform = transform)
    path = './CK_log/{}/'.format(i)
    if not os.path.exists(path):
        os.mkdir(path)

    ep, acc = train(i,train_dataset,path,test_dataset)

    test_ep.append(ep)
    test_res.append(acc)
end_time = time.time()
duration = end_time - strat_time
for i in range(10):
    print(test_ep[i], test_res[i])

print(duration)
