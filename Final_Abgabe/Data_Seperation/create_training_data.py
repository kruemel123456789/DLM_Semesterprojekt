#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:40:20 2017

@author: root
"""

import os
import shutil

#set the sizes for the data
TRAINING_SIZE = 571
VALI_SIZE = 200

#set the path
fpath_training = './train_res/training'
fpath_vali = './train_res/vali'
fpath_categories = './train_res/categories/'

#remove old folders
if os.path.exists(fpath_training):
    shutil.rmtree(fpath_training)
if os.path.exists(fpath_vali):
    shutil.rmtree(fpath_vali)

#(re-) create folders
os.mkdir(fpath_training)
os.mkdir(fpath_vali)

#loop over the categories 0-4
for items in range(5):
    print(items)    #print the categories to get indo about progress
    savelist = os.listdir(fpath_categories + str(items))    #save all files in the folder
    
    #loop over the files in the ITEM folder from "0" to "Trainingsize" an copy them into a Training-Folder
    for nums in range(TRAINING_SIZE-1):
        if not os.path.exists(fpath_training + "/" + str(items)):
            os.mkdir(fpath_training + "/" + str(items))
        shutil.copy(fpath_categories + "/" + str(items) + "/" + str(savelist[nums]),fpath_training + "/" + str(items) + "/" + str(savelist[nums]))
    
    #loop over the files in the ITEM folder from "Trainingsize" to "Trainingsize+Valisize" an copy them into a Validation-Folder
    for nums in range(TRAINING_SIZE,(VALI_SIZE+TRAINING_SIZE-1)):
        if not os.path.exists(fpath_vali + "/" + str(items)):
            os.mkdir(fpath_vali + "/" + str(items))
        shutil.copy(fpath_categories + "/" + str(items) + "/" + str(savelist[nums]),fpath_vali + "/" + str(items) + "/" + str(savelist[nums]))
    