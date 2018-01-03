#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:23:27 2017
calculate mean and std of images

@author: root
"""
import numpy as np
from PIL import Image
import os

#access all files
img_dir = os.getcwd()
os.chdir(os.getcwd()+"/test_res/images")
img_dir = os.getcwd()
print (img_dir)
allfiles = os.listdir(img_dir)
imlist = [filename for filename in allfiles if filename[-5:] in [".tiff",]]

#get the size
w,h = Image.open(imlist[0]).size
N=len(imlist)

#array to store the average of rgb
arr = np.zeros((h,w,3),np.float)

#build the average
for im in imlist:
    imarr = np.array(Image.open(im),dtype = np.float)
    arr = arr+imarr/N

mean = np.mean(arr)
print ("mean: " , mean)

std = np.std(arr)
print ("std: " , std)