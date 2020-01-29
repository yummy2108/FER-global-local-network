# -*- coding: utf-8 -*-
# @Author: Yan Tang
# @Date:   2018-06-27 

'''
----------------------------------------------------------------
Load data for '.pkl' file. 
Generate differential geometric data.
Convert label to one hot label. 
For example: 
3->[0,0,1,0,0,0] and 5->[0,0,0,0,1,0].
Load data for model training for cnn_for_fera_ten_fold_ten.py
----------------------------------------------------------------
'''

import os
import pickle 
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
def pickle_2_img(data_file):
     if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
     with open(data_file, 'rb') as f:
        data = pickle.load(f)
     total_x1, total_x2, total_x3, total_gx, total_y = [], [], [], [], []
     total_mouse_1,total_mouse_2,total_mouse_3 = [],[],[]
     total_nose_1,total_nose_2,total_nose_3 = [],[],[]
     total_eye_1,total_eye_2,total_eye_3 = [],[],[]     
     #ten groups data for ten-fold cross validation
     for i in range(len(data)):         
         x1 = []
         x2 = []
         x3 = []
         x4 = []
         m1 = []
         m2 = []
         m3 = []
         n1 = []
         n2 = []
         n3 = []
         e1 = []
         e2 = []
         e3 = []         
         yl = []
         for j in range(len(data[i]['labels'])):
             geo_array = data[i]['geometry'][j]
             v0 = geo_array[0]
             v1 = geo_array[1]
             v2 = geo_array[2]
             img_array = data[i]['img'][j]
             
             #the first image
             img1 = img_array[0]
             img1 = img1.flatten()          
             #the middle image
             img2 = img_array[1]
             img2 = img2.flatten()
             #the last image
             img3 = img_array[2]
             img3 = img3.flatten()

             mouse_array = data[i]['mouse'][j]
             mouse1 = mouse_array[0]
             mouse1 = mouse1.flatten()
             mouse2 = mouse_array[1]
             mouse2 = mouse2.flatten()
             mouse3 = mouse_array[2]
             mouse3 = mouse3.flatten()
             m1.append(mouse1)
             m2.append(mouse2)
             m3.append(mouse3)             

             nose_array = data[i]['nose'][j]
             nose1 = nose_array[0]
             nose1 = nose1.flatten()
             nose2 = nose_array[1]
             nose2 = nose2.flatten()
             nose3 = nose_array[2]
             nose3 = nose3.flatten()
             n1.append(nose1)
             n2.append(nose2)
             n3.append(nose3) 

             eye_array = data[i]['eye'][j]
             eye1 = eye_array[0]
             eye1 = eye1.flatten()
             eye2 = eye_array[1]
             eye2 = eye2.flatten()
             eye3 = eye_array[2]
             eye3 = eye3.flatten()
             e1.append(eye1)
             e2.append(eye2)
             e3.append(eye3)            
             #final difference
             v = list(map(lambda x: x[0]-x[1], zip(v2, v0))) 
             #dynamicn geometric feature
             gx = v2+v

             label = int(data[i]['labels'][j][2])
             
             #label mapping
             if label==7:
                 label = 2
             label = label-1
             #print(label)                 
             #label = dense_to_one_hot(label,7)
                          
             x1.append(img1)
             x2.append(img2)
             x3.append(img3)
             x4.append(gx)
             yl.append(label)
         """
         x1 = np.asarray(x1)
         x2 = np.asarray(x2)
         x3 = np.asarray(x3)
         x4 = np.asarray(x4)
         yl = np.asarray(yl)

         m1 = np.asarray(m1)
         m2 = np.asarray(m2)
         m3 = np.asarray(m3)

         n1 = np.asarray(n1)
         n2 = np.asarray(n2)
         n3 = np.asarray(n3)

         e1 = np.asarray(e1)
         e2 = np.asarray(e2)
         e3 = np.asarray(e3)
         """
         total_x1.append(x1)
         total_x2.append(x2)
         total_x3.append(x3)
         total_gx.append(x4)
         total_y.append(yl) 
         total_mouse_1.append(m1)
         total_mouse_2.append(m2)
         total_mouse_3.append(m3)
         total_nose_1.append(n1)
         total_nose_2.append(n2)
         total_nose_3.append(n3)
         total_eye_1.append(e1)
         total_eye_2.append(e2)
         total_eye_3.append(e3) 

     total_x1 = np.asarray(total_x1)  
     total_x2 = np.asarray(total_x2) 
     total_x3 = np.asarray(total_x3) 

     total_mouse_1 = np.asarray(total_mouse_1)  
     total_mouse_2 = np.asarray(total_mouse_2) 
     total_mouse_3 = np.asarray(total_mouse_3) 

     total_nose_1 = np.asarray(total_nose_1)  
     total_nose_2 = np.asarray(total_nose_2) 
     total_nose_3 = np.asarray(total_nose_3) 

     total_eye_1 = np.asarray(total_eye_1)  
     total_eye_2 = np.asarray(total_eye_2) 
     total_eye_3 = np.asarray(total_eye_3) 

     total_gx = np.asarray(total_gx)
     total_y = np.asarray(total_y)


           
     return total_x1, total_x2, total_x3, total_gx, total_y,total_mouse_1,total_mouse_2,total_mouse_3,total_nose_1,total_nose_2,total_nose_3,total_eye_1,total_eye_2,total_eye_3

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = []
    for i in range(num_classes):
        if i==labels_dense-1:
            labels_one_hot.append(1)
        else:
            labels_one_hot.append(0)
    return labels_one_hot
"""
total_x1, total_x2, total_x3, total_gx, total_y,total_mouse_1,total_mouse_2,total_mouse_3,total_nose_1,total_nose_2,total_nose_3,total_eye_1,total_eye_2,total_eye_3 = pickle_2_img('./CK/ckplus_with_img_geometry_3frame.pkl')

print(total_x1.shape, total_mouse_3.shape, total_y.shape)
temp = np.delete(total_x1,0,axis=0)
temp = np.asarray(temp)
print(temp.shape,temp[0].shape)
temp = np.reshape(temp,(temp.shape[0]*temp[0].shape[0], 128,128))
print(temp.shape)
for i in range(10):
    print(total_mouse_3[i].shape)
print(total_x1[0][0][:])
"""