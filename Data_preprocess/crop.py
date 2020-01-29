from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import dlib
from skimage import io
from PIL import Image
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import heapq
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  
def generate_crop_margin(x,y,bias,img):
    margin = []
    top = int(max(np.min(y)-bias,0))
    bottom = int(np.max(y)+bias)
    right = int(np.max(x)+bias)
    left = int(max(np.min(x)-bias,0))
    margin.append(top)
    margin.append(bottom)
    margin.append(right)
    margin.append(left)
    crop_part = img[margin[0]:margin[1],margin[3]:margin[2]]
    return crop_part
def get_mouse(x,y,img):
    mouse = generate_crop_margin(x[48:67],y[48:67],5,img)
    mouse = cv2.resize(mouse,(64, 32))
    return mouse
def get_nose(x,y,img):
    nose = generate_crop_margin(x[27:36],y[27:36],5,img)
    nose = cv2.resize(nose,(32, 64))
    return nose 
def get_eye(x,y,img):
    x1 = np.array(x[17:27])
    x2 = np.array(x[37:48])
    x_ = []
    for x in x1:
        x_.append(x)
    for x in x2:
        x_.append(x)
    y1 = np.array(y[17:27])
    y2 = np.array(y[37:48])
    y_ = []
    for y in y1:
        y_.append(y)
    for y in y2:
        y_.append(y)    
    eye = generate_crop_margin(x_,y_,5,img)
    eye = cv2.resize(eye,(128, 48))
    return eye        
def __shape_to_np(shape):
    '''Transform the shape points into numpy array of 68*2'''
    nLM = shape.num_parts
    x = np.asarray([shape.part(i).x for i in range(0,nLM)])
    y = np.asarray([shape.part(i).y for i in range(0,nLM)])
    return x,y
def getLandmark(file):
    imgcv_gray=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if imgcv_gray is None:
        print('Unexpected ERROR: The value read from the imagepath is None. No image was loaded',file)
        exit(-1)
    dets = detector(imgcv_gray,1)
    if len(dets)==0:
        dets = detector(imgcv_gray,0)
        if len(dets)==0:
               print("No face was detected^^^^^^^^^^^^^^",file)
               return False, imgcv_gray
    for id, det in enumerate(dets):
        if id > 0:
            print("ONLY process the first face>>>>>>>>>")
            break
        shape = predictor(imgcv_gray, det)
        x, y = __shape_to_np(shape) 
    return x,y,imgcv_gray
def process(file):
    x,y,imgcv_gray = getLandmark(file)
    mouse = get_mouse(x,y,imgcv_gray)
    nose = get_nose(x,y,imgcv_gray)
    eye = get_eye(x,y,imgcv_gray)
    return mouse,nose,eye