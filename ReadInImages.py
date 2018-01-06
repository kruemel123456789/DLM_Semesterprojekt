#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:08:37 2018

@author: root
"""

import numpy as np
import os
from scipy import ndimage
from skimage.transform import resize
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import scipy



def getAveragePics():
    path = "train_res/avgPics/"
    dirs = os.listdir(path)
    #numpy_int_array = np.array(dirs)
    #for i in numpy_int_array:
    #    input_im = cv2.imread(numpy_int_array[i], 1)
    #    print(input_im.shape)
    
    imList = []
    imList2 = np.zeros(shape=(5, 3, 512,512))

    counter = 0
    for imageName in sorted(dirs):
        imList.append(ndimage.imread(path + os.sep + imageName).transpose((2,0,1)))
        imList2[counter,:,:,:] = ndimage.imread(path + os.sep + imageName).transpose((2,0,1))
        counter = counter +1
        
    imList2 *= 1/255.0
    return imList2

def preprocess_img(img):
    #img = scipy.ndimage.interpolation.zoom(img, (224/512, 224/512, 1.0))
    img = preprocess_input(img)
    return img
    
    