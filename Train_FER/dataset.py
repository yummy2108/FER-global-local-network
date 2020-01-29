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

def toOne(im):
    """
    min_num = im.min()
    im = im - min_num
    max_num = im.max()
    im = im / max_num
    """
    im = im[:,:,np.newaxis]
    im = np.tile(im,(1,1,3))
    return im

class Emotion_Loder(Dataset):
    def __init__(self, txt, fold,transform=None, target_transform=None):
        total_x1, total_x2, total_x3, total_gx, total_y,total_mouth_1,total_mouth_2,total_mouth_3,total_nose_1,total_nose_2,total_nose_3,total_eye_1,total_eye_2,total_eye_3 = pickle_2_img(txt)
        
        self.label = np.delete(total_y,fold,axis=0)
        self.label = np.reshape(self.label,(self.label.shape[0]*self.label.shape[1],1))

        self.x_1 = np.delete(total_x1,fold,axis=0)        
        self.x_2 = np.delete(total_x2,fold,axis=0)
        self.x_3 = np.delete(total_x3,fold,axis=0)

        self.x_1 = np.reshape(self.x_1,(self.x_1.shape[0]*self.x_1.shape[1],128,128))
        self.x_2 = np.reshape(self.x_2,(self.x_2.shape[0]*self.x_2.shape[1],128,128))
        self.x_3 = np.reshape(self.x_3,(self.x_3.shape[0]*self.x_3.shape[1],128,128))

        self.eye_1 = np.delete(total_eye_1,fold,axis=0)        
        self.eye_2 = np.delete(total_eye_2,fold,axis=0)
        self.eye_3 = np.delete(total_eye_3,fold,axis=0)

        self.eye_1 = np.reshape(self.eye_1,(self.eye_1.shape[0]*self.eye_1.shape[1],128,48))
        self.eye_2 = np.reshape(self.eye_2,(self.eye_2.shape[0]*self.eye_2.shape[1],128,48))
        self.eye_3 = np.reshape(self.eye_3,(self.eye_3.shape[0]*self.eye_3.shape[1],128,48))

        self.mouth_1 = np.delete(total_mouth_1,fold,axis=0)
        self.mouth_2 = np.delete(total_mouth_2,fold,axis=0)
        self.mouth_3 = np.delete(total_mouth_3,fold,axis=0)

        self.mouth_1 = np.reshape(self.mouth_1,(self.mouth_1.shape[0]*self.mouth_1.shape[1],64,32))
        self.mouth_2 = np.reshape(self.mouth_2,(self.mouth_2.shape[0]*self.mouth_2.shape[1],64,32))
        self.mouth_3 = np.reshape(self.mouth_3,(self.mouth_3.shape[0]*self.mouth_3.shape[1],64,32))

        self.nose_1 = np.delete(total_nose_1,fold,axis=0)
        self.nose_2 = np.delete(total_nose_2,fold,axis=0)
        self.nose_3 = np.delete(total_nose_3,fold,axis=0)

        self.nose_1 = np.reshape(self.nose_1,(self.nose_1.shape[0]*self.nose_1.shape[1],32,64))
        self.nose_2 = np.reshape(self.nose_2,(self.nose_2.shape[0]*self.nose_2.shape[1],32,64))
        self.nose_3 = np.reshape(self.nose_3,(self.nose_3.shape[0]*self.nose_3.shape[1],32,64))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        l = self.label[index]

        x1 = self.x_1[index]
        x2 = self.x_2[index]
        x3 = self.x_3[index]

        x1 = toOne(x1)
        x2 = toOne(x2)
        x3 = toOne(x3)

        e1 = self.eye_1[index]
        e2 = self.eye_2[index]
        e3 = self.eye_3[index]

        e1 = toOne(e1)
        e2 = toOne(e2)
        e3 = toOne(e3)

        m1 = self.mouth_1[index]
        m2 = self.mouth_2[index]
        m3 = self.mouth_3[index]

        m1 = toOne(m1)
        m2 = toOne(m2)
        m3 = toOne(m3)

        n1 = self.nose_1[index]
        n2 = self.nose_2[index]
        n3 = self.nose_3[index] 

        n1 = toOne(n1)
        n2 = toOne(n2)
        n3 = toOne(n3)  

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            x3 = self.transform(x3)

            e1 = self.transform(e1)
            e2 = self.transform(e2)
            e3 = self.transform(e3)

            n1 = self.transform(n1)
            n2 = self.transform(n2)
            n3 = self.transform(n3)

            m1 = self.transform(m1)
            m2 = self.transform(m2)
            m3 = self.transform(m3)

        return l, x1, x2, x3, e1, e2, e3, n1, n2, n3, m1, m2, m3

    def __len__(self):
        return len(self.label)

class test_Emotion_Loder(Dataset):
    def __init__(self, txt, fold,transform=None, target_transform=None):
        total_x1, total_x2, total_x3, total_gx, total_y,total_mouth_1,total_mouth_2,total_mouth_3,total_nose_1,total_nose_2,total_nose_3,total_eye_1,total_eye_2,total_eye_3 = pickle_2_img(txt)
        
        self.label = total_y[fold]

        self.x_1 = total_x1[fold]       
        self.x_2 = total_x2[fold] 
        self.x_3 = total_x3[fold] 
        
        self.x_1 = np.asarray(self.x_1)
        self.x_2 = np.asarray(self.x_2)
        self.x_3 = np.asarray(self.x_3)

        self.x_1 = np.reshape(self.x_1,(self.x_1.shape[0],128,128))
        self.x_2 = np.reshape(self.x_2,(self.x_2.shape[0],128,128))
        self.x_3 = np.reshape(self.x_3,(self.x_3.shape[0],128,128))

        self.eye_1 = total_eye_1[fold]       
        self.eye_2 = total_eye_2[fold]
        self.eye_3 = total_eye_3[fold]
        
        self.eye_1 = np.asarray(self.eye_1)
        self.eye_2 = np.asarray(self.eye_2)
        self.eye_3 = np.asarray(self.eye_3)

        self.eye_1 = np.reshape(self.eye_1,(self.eye_1.shape[0],128,48))
        self.eye_2 = np.reshape(self.eye_2,(self.eye_2.shape[0],128,48))
        self.eye_3 = np.reshape(self.eye_3,(self.eye_3.shape[0],128,48))

        self.mouth_1 = total_mouth_1[fold]
        self.mouth_2 = total_mouth_2[fold]
        self.mouth_3 = total_mouth_3[fold]
        
        self.mouth_1 = np.asarray(self.mouth_1)
        self.mouth_2 = np.asarray(self.mouth_2)
        self.mouth_3 = np.asarray(self.mouth_3)

        self.mouth_1 = np.reshape(self.mouth_1,(self.mouth_1.shape[0],64,32))
        self.mouth_2 = np.reshape(self.mouth_2,(self.mouth_2.shape[0],64,32))
        self.mouth_3 = np.reshape(self.mouth_3,(self.mouth_3.shape[0],64,32))

        self.nose_1 = total_nose_1[fold]
        self.nose_2 = total_nose_2[fold]
        self.nose_3 = total_nose_3[fold]
        
        self.nose_1 = np.asarray(self.nose_1)
        self.nose_2 = np.asarray(self.nose_2)
        self.nose_3 = np.asarray(self.nose_3)

        self.nose_1 = np.reshape(self.nose_1,(self.nose_1.shape[0],32,64))
        self.nose_2 = np.reshape(self.nose_2,(self.nose_2.shape[0],32,64))
        self.nose_3 = np.reshape(self.nose_3,(self.nose_3.shape[0],32,64))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        l = self.label[index]

        x1 = self.x_1[index]
        x2 = self.x_2[index]
        x3 = self.x_3[index]

        x1 = toOne(x1)
        x2 = toOne(x2)
        x3 = toOne(x3)

        e1 = self.eye_1[index]
        e2 = self.eye_2[index]
        e3 = self.eye_3[index]

        e1 = toOne(e1)
        e2 = toOne(e2)
        e3 = toOne(e3)

        m1 = self.mouth_1[index]
        m2 = self.mouth_2[index]
        m3 = self.mouth_3[index]

        m1 = toOne(m1)
        m2 = toOne(m2)
        m3 = toOne(m3)

        n1 = self.nose_1[index]
        n2 = self.nose_2[index]
        n3 = self.nose_3[index] 

        n1 = toOne(n1)
        n2 = toOne(n2)
        n3 = toOne(n3)  

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            x3 = self.transform(x3)

            e1 = self.transform(e1)
            e2 = self.transform(e2)
            e3 = self.transform(e3)

            n1 = self.transform(n1)
            n2 = self.transform(n2)
            n3 = self.transform(n3)

            m1 = self.transform(m1)
            m2 = self.transform(m2)
            m3 = self.transform(m3)

        return l, x1, x2, x3, e1, e2, e3, n1, n2, n3, m1, m2, m3

    def __len__(self):
        return len(self.label)

def pre(data):
    res = []
    for i in range(len(data)):
        res += data[i]
    res = np.asarray(res)
    return res
class CK_Emotion_Loder(Dataset):
    def __init__(self, txt, fold,transform=None, target_transform=None):
        total_x1, total_x2, total_x3, total_gx, total_y,total_mouth_1,total_mouth_2,total_mouth_3,total_nose_1,total_nose_2,total_nose_3,total_eye_1,total_eye_2,total_eye_3 = pickle_2_img(txt)
        
        label = np.delete(total_y,fold,axis=0)
        self.label = pre(label)
        self.label = np.reshape(self.label,(self.label.shape[0],1))

        x_1 = np.delete(total_x1,fold,axis=0)        
        x_2 = np.delete(total_x2,fold,axis=0)
        x_3 = np.delete(total_x3,fold,axis=0)
        
        self.x_1 = pre(x_1)        
        self.x_2 = pre(x_2)
        self.x_3 = pre(x_3) 

        self.x_1 = np.reshape(self.x_1,(self.x_1.shape[0],128,128))
        self.x_2 = np.reshape(self.x_2,(self.x_2.shape[0],128,128))
        self.x_3 = np.reshape(self.x_3,(self.x_3.shape[0],128,128))

        eye_1 = np.delete(total_eye_1,fold,axis=0)        
        eye_2 = np.delete(total_eye_2,fold,axis=0)
        eye_3 = np.delete(total_eye_3,fold,axis=0)
        
        self.eye_1 = pre(eye_1)        
        self.eye_2 = pre(eye_2)
        self.eye_3 = pre(eye_3)

        self.eye_1 = np.reshape(self.eye_1,(self.eye_1.shape[0],128,48))
        self.eye_2 = np.reshape(self.eye_2,(self.eye_2.shape[0],128,48))
        self.eye_3 = np.reshape(self.eye_3,(self.eye_3.shape[0],128,48))

        mouth_1 = np.delete(total_mouth_1,fold,axis=0)
        mouth_2 = np.delete(total_mouth_2,fold,axis=0)
        mouth_3 = np.delete(total_mouth_3,fold,axis=0)
        
        self.mouth_1 = pre(mouth_1)
        self.mouth_2 = pre(mouth_2)
        self.mouth_3 = pre(mouth_3)
        
        self.mouth_1 = np.reshape(self.mouth_1,(self.mouth_1.shape[0],64,32))
        self.mouth_2 = np.reshape(self.mouth_2,(self.mouth_2.shape[0],64,32))
        self.mouth_3 = np.reshape(self.mouth_3,(self.mouth_3.shape[0],64,32))

        nose_1 = np.delete(total_nose_1,fold,axis=0)
        nose_2 = np.delete(total_nose_2,fold,axis=0)
        nose_3 = np.delete(total_nose_3,fold,axis=0)

        self.nose_1 = pre(nose_1)
        self.nose_2 = pre(nose_2)
        self.nose_3 = pre(nose_3)
        
        self.nose_1 = np.reshape(self.nose_1,(self.nose_1.shape[0],64,32))
        self.nose_2 = np.reshape(self.nose_2,(self.nose_2.shape[0],64,32))
        self.nose_3 = np.reshape(self.nose_3,(self.nose_3.shape[0],64,32))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        l = self.label[index]

        x1 = self.x_1[index]
        x2 = self.x_2[index]
        x3 = self.x_3[index]

        x1 = toOne(x1)
        x2 = toOne(x2)
        x3 = toOne(x3)

        e1 = self.eye_1[index]
        e2 = self.eye_2[index]
        e3 = self.eye_3[index]

        e1 = toOne(e1)
        e2 = toOne(e2)
        e3 = toOne(e3)

        m1 = self.mouth_1[index]
        m2 = self.mouth_2[index]
        m3 = self.mouth_3[index]

        m1 = toOne(m1)
        m2 = toOne(m2)
        m3 = toOne(m3)

        n1 = self.nose_1[index]
        n2 = self.nose_2[index]
        n3 = self.nose_3[index] 

        n1 = toOne(n1)
        n2 = toOne(n2)
        n3 = toOne(n3)  

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            x3 = self.transform(x3)

            e1 = self.transform(e1)
            e2 = self.transform(e2)
            e3 = self.transform(e3)

            n1 = self.transform(n1)
            n2 = self.transform(n2)
            n3 = self.transform(n3)

            m1 = self.transform(m1)
            m2 = self.transform(m2)
            m3 = self.transform(m3)

        return l, x1, x2, x3, e1, e2, e3, n1, n2, n3, m1, m2, m3

    def __len__(self):
        return len(self.label) 