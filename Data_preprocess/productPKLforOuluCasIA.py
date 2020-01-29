import cv2
import time
import ntpath
import numpy as np
import sys, os
import pickle
import FaceProcessUtil as fpu
import crop
import matplotlib.pyplot as plt

emotion_key = {'Sadness':6,'Surprise':7,'Anger':1,'Fear':4,'Happiness':5,'Disgust':3}
sampe_size = 3 # sample size, number of images

imgPath = './data_oulu/Oulu_CASIA_Ten_group.txt'

#change to your data root according the image path from txt file
data_root_path = '/Oulu_casia/'

#total list of path and label
imglist = [[],[],[],[],[],[],[],[],[],[]] #ten fold
labellist = [[],[],[],[],[],[],[],[],[],[]] #ten fold


#-----------------prepare imglist for  3 frame [[f1,f2,f3],[f1,f2,f3].......,]
list_txt = open(imgPath,'r')
content = list_txt.readlines()

#append img to one list
for i, line in enumerate(content):
    line = line.replace('\n','')
    line = line.split('\t')
    group = int(line[0])
    path = line[1]
    label = line[2]
    imglist[group-1].append(data_root_path+path)
    labellist[group-1].append(label)

# resize the list: three images as one sample [f1,f2,f3]
for i in range(len(imglist)):
    length = len(imglist[i])
    num = int(length/sampe_size)
    imglist[i] = np.reshape(imglist[i], (num, sampe_size))
    labellist[i] = np.reshape(labellist[i], (num, sampe_size))


total = 0

for i ,e in enumerate(labellist):
    print(len(e))
    total = total+len(e)
print("Total training images:%d"%(total*3))
#-----------------prepare imglist end -----------------------------

count=0

gc = 0

feature_group_of_subject=[]

tm1=time.time()

for i in range(len(imglist)):
    ckplus={}
    ckplus_gf=[]
    ckplus_label=[]
    ckplus_img = []
    ckplus_mouse = []
    ckplus_nose = []
    ckplus_eye = []
    imagelist = imglist[i]
    lablist = labellist[i]
    for j,v in enumerate(imagelist):
        print(v)
        count = count+1
        label = lablist[j][0]
        print("\n> Prepare image                                                                                %f%%"%(count*100/total))

        groupframe_gf = []
        groupframe_gl = []
        groupframe_img = []
        groupframe_mouse = []
        groupframe_nose = []
        groupframe_eye = []
        for k,m in enumerate(v):
            image_path = m
            flag, img=fpu.calibrateImge(image_path)
            mouse,nose,eye = crop.process(image_path)
            groupframe_mouse.append(mouse)
            groupframe_nose.append(nose)
            groupframe_eye.append(eye)
            print('get mouse eye nose')
            if flag:
                 imgr = fpu.getLandMarkFeatures_and_ImgPatches(img)
            else:
                print('Unexpected case while calibrating for:'+str(image_path))
                exit(1)

            if imgr[1]:
                gc = gc+1
                img=imgr[0]
                img2 = img
                print("Get Geometry>>>>>>>>>>>>>>")
                groupframe_gf.append(imgr[2])
                groupframe_gl.append(label)
                groupframe_img.append(img2)
            else:
                print('No feature detected:'+image_path)
                exit(1)
        ckplus_gf.append(groupframe_gf)
        ckplus_label.append(groupframe_gl)
        ckplus_img.append(groupframe_img)
        ckplus_mouse.append(groupframe_mouse)
        ckplus_nose.append(groupframe_nose)
        ckplus_eye.append(groupframe_eye)
    ckplus['labels']=ckplus_label
    ckplus['geometry']=ckplus_gf
    ckplus['img'] = ckplus_img
    ckplus['mouse'] = ckplus_mouse
    ckplus['nose'] = ckplus_nose
    ckplus['eye'] = ckplus_eye
    feature_group_of_subject.append(ckplus)

filenametosave='./data_oulu/oulus_casia_de_1.pkl'

with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject,fin,4)
 
tm2=time.time()
dtm = tm2-tm1
print('Total images: %d\tGet: %d'%(count*3, gc))
print("Total time comsuming: %fs for %d images"%(dtm, count*3))


