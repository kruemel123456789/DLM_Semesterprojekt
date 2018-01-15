# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:34:23 2017

@author: root
"""
import os
import numpy as np
import pandas as pd

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from keras import backend as K
K.set_session(tf.Session(config=config))

from glob import glob
from keras.models import load_model


from PIL import Image
import scipy

from sklearn.externals import joblib

import ReadInImages

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model




from keras.preprocessing.image import ImageDataGenerator


def create_submission(preds, names, group, file_description, output_folder):
    '''
       Creates a Submission as a DataFrame with the required format and 
       saves it as csv under the given arguments
    '''
    # create directory if it doesnt exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # images names
    image_column = pd.Series(names, name='image')    
    # images predictions
    level_column =  pd.Series(np.array(preds).astype(np.int64), name='level')
    predictions = pd.concat([image_column, level_column], axis=1)
    
    # Save submission to csv
    filename = output_folder+'/submission_'+group+'_'+file_description+'.csv'
    predictions.to_csv(filename, index=False)


class DataGeneratorVGG19(object):

    def __init__(self, batch_size=16, flip = False):
        self.batch_size = batch_size
        self.flip = flip
        
    def generate(self, data_list):
        '''Generates batches of samples'''
        while True:
            indexes = np.arange(0, len(data_list))
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Get the data from a and b
                idxs = indexes[i*self.batch_size:(i+1)*self.batch_size]
                # load images
                batch_x = []
                for idx in idxs:
                    im = Image.open(data_list[idx])
                    
                    im = im.resize((224,224))
                    
                    # Preprocessing
                    im = np.array(im)
                    if(self.flip):
                        np.flip(im, 0)
                    #im= ReadInImages.preprocess_img(im.astype(np.float64))
                                       
                    
                    #im = scipy.ndimage.interpolation.zoom(im, (224/512, 224/512, 1.0))
                    batch_x.append(im.astype(np.float32))
                
                yield np.array(batch_x)
                #yield img
                
class DataGeneratorVGG16(object):

    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        
    def generate(self, data_list):
        '''Generates batches of samples'''
        while True:
            indexes = np.arange(0, len(data_list))
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Get the data from a and b
                idxs = indexes[i*self.batch_size:(i+1)*self.batch_size]
                # load images
                batch_x = []
                for idx in idxs:
                    im = Image.open(data_list[idx])
                    
                    im = im.resize((224,224))
                    
                    # Preprocessing
                    im = np.array(im)
                    #im= preprocess_input(im.astype(np.float64))
                                       
                    
                    #im = scipy.ndimage.interpolation.zoom(im, (224/512, 224/512, 1.0))
                    batch_x.append(im.astype(np.float32))
                
                yield np.array(batch_x)
                #yield img                
        
def main():        
    # load model

    #Felix
    #models_dir = '../models/'
    #model = load_model('models/model019-31_12-10_06')

    #Linus
    models_dir = 'models/'
    model1 = load_model(models_dir + 'F_MVGG_8-sgd-moreEpochs.hdf5')
    model2 = load_model(models_dir + 'F_MVGG_10.hdf5')
    model3 = load_model(models_dir + 'L_MVGG16_1-e72-val_acc0.46.hdf5')


    batch_size = 29
    
    # image paths
    test_data_path = 'test_res/images'

    image_paths = glob(test_data_path+'/*.tiff')
    
    # image names
    image_names = [im.split('/')[-1].split('.')[0] for im in image_paths]
    
    # data generator
    test_generator1 = DataGeneratorVGG19(batch_size=batch_size).generate(image_paths)
    test_generator1_flip = DataGeneratorVGG19(batch_size=batch_size, flip = True).generate(image_paths)
    test_generator2 = DataGeneratorVGG19(batch_size=batch_size).generate(image_paths)
    test_generator3 = DataGeneratorVGG16(batch_size=batch_size).generate(image_paths)
        
    # predict
    y_preds1 = model1.predict_generator(test_generator1,
                                      verbose=1,
                                      workers=1,
                                      use_multiprocessing=False,
                                      steps = len(image_paths)//batch_size)
    
    y_preds1_flip = model1.predict_generator(test_generator1_flip,
                                      verbose=1,
                                      workers=16,
                                      use_multiprocessing=True,
                                      steps = len(image_paths)//batch_size)
    
    y_preds2 = model2.predict_generator(test_generator2,
                                      verbose=1,
                                      workers=1,
                                      use_multiprocessing=False,
                                      steps = len(image_paths)//batch_size)
    y_preds3 = model3.predict_generator(test_generator3,
                                      verbose=1,
                                      workers=1,
                                      use_multiprocessing=False,
                                      steps = len(image_paths)//batch_size)
    
    
    
    y_preds =  1.0*y_preds1 + 1*y_preds1_flip +  y_preds2 + y_preds3
    
    X = np.hstack((y_preds1, y_preds2, y_preds3))
    X2 = np.hstack((y_preds1, y_preds2))
    X3 = np.hstack((y_preds2, y_preds3))
    X4 = np.hstack((y_preds1, y_preds3))
    
    
    clf1 = joblib.load('clf1.pkl')
    clf2 = joblib.load('clf2.pkl')
    clf3 = joblib.load('clf3.pkl')
    clf4 = joblib.load('clf4.pkl')
        
    y_out1 = clf1.predict(X)
    y_out2 = clf2.predict(X2)
    y_out3 = clf3.predict(X3)
    y_out4 = clf4.predict(X4)
    
    y_out_summe = np.argmax(y_preds, axis=1)
    y_out_summe = np.rint(y_out_summe)
    
    
    y_out_1_flip = np.argmax(y_preds1_flip, axis = 1)
    y_out_1_flip = np.rint(y_out_1_flip)
    
    y_out_allein_1 = np.argmax(y_preds1,axis = 1) 
    y_out_allein_1= np.rint(y_out_allein_1)
    
    y_out_allein_2 = np.argmax(y_preds2,axis = 1) 
    y_out_allein_2 = np.rint(y_out_allein_2)
    
    y_out_allein_3 = np.argmax(y_preds3,axis = 1) 
    y_out_allein_3 = np.rint(y_out_allein_3)
    
    # create and save submission to csv
    create_submission(preds=y_out1, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='Multi_alle',

                      output_folder='./submissions')
    
    create_submission(preds=y_out2, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='Multi_1_2',

                      output_folder='./submissions')
    
    create_submission(preds=y_out3, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='Multi_2_3',

                      output_folder='./submissions')
    create_submission(preds=y_out4, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='Multi_1_3',
        
                      output_folder='./submissions')
      
    create_submission(preds=y_out_allein_1, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='nur_1',

                      output_folder='./submissions')
       
    create_submission(preds=y_out_allein_2, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='nur_2',

                      output_folder='./submissions')
       
    create_submission(preds=y_out_allein_3, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='nur_3',

                      output_folder='./submissions')
    
    create_submission(preds=y_out_summe, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='Summe_Flip_11',

                      output_folder='./submissions')
    
    create_submission(preds=y_out_1_flip, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='1_Flip_1',

                      output_folder='./submissions')

if __name__=="__main__":
    main()

