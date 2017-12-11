#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:40:20 2017

@author: root
"""
import pandas as pd
import os
import shutil #import copyfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fname = './train_res/trainLabelsDroppedResVal.csv'
fpath_raw = './train_res/images/'
fpath_categories = './train_res/categories/'

df = pd.read_csv(os.path.expanduser(fname))

for index,row in  df.iterrows():
    #print(row["level"])
    print(row["image"])
    if not os.path.exists(fpath_categories + str(row["level"])):
        os.mkdir(fpath_categories + str(row["level"]))
    shutil.copy(fpath_raw + row["image"] + ".tiff", fpath_categories + str(row["level"]) + "/" + row["image"] + ".tif")