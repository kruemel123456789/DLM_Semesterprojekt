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

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.externals import joblib


from PIL import Image
import scipy




from keras.preprocessing.image import ImageDataGenerator

import ReadInImages

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


class DataGenerator(object):

    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        
    def generate(self, data_list):
        '''Generates batches of samples'''
        while True:
            indexes = np.arange(0, len(data_list))
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size) + 1
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
                    #im = scipy.ndimage.interpolation.zoom(im, (224/512, 224/512, 1.0))
                    batch_x.append(np.array(im))
                
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
    
    


    batch_size = 33
    
    # image paths
    test_data_path0 = 'train_res/internal/0'
    test_data_path1 = 'train_res/internal/1'
    test_data_path2 = 'train_res/internal/2'
    test_data_path3 = 'train_res/internal/3'
    test_data_path4 = 'train_res/internal/4'

    image_paths0 = glob(test_data_path0+'/*.tiff')
    image_paths1 = glob(test_data_path1+'/*.tiff')
    image_paths2 = glob(test_data_path2+'/*.tiff')
    image_paths3 = glob(test_data_path3+'/*.tiff')
    image_paths4 = glob(test_data_path4+'/*.tiff')
    
    # image names
    image_names0 = [im.split('/')[-1].split('.')[0] for im in image_paths0]
    image_labels0 = 0 * np.ones(len(image_names0))
    image_names1 = [im.split('/')[-1].split('.')[0] for im in image_paths1]
    image_labels1 = np.ones(len(image_names1))
    image_names2 = [im.split('/')[-1].split('.')[0] for im in image_paths2]
    image_labels2 = 2 * np.ones(len(image_names2))
    image_names3 = [im.split('/')[-1].split('.')[0] for im in image_paths3]
    image_labels3 = 3 * np.ones(len(image_names3))
    image_names4 = [im.split('/')[-1].split('.')[0] for im in image_paths4]
    image_labels4 = 4 * np.ones(len(image_names4))

    
    image_paths = image_paths0
    image_paths = np.hstack((image_paths,image_paths1))
    image_paths = np.hstack((image_paths,image_paths2))
    image_paths = np.hstack((image_paths,image_paths3))
    image_paths = np.hstack((image_paths,image_paths4))
    
    image_names = image_names0
    image_names = np.hstack((image_names,image_names1))
    image_names = np.hstack((image_names,image_names2))
    image_names = np.hstack((image_names,image_names3))
    image_names = np.hstack((image_names,image_names4))
    
    image_labels = image_labels0
    image_labels = np.hstack((image_labels,image_labels1))
    image_labels = np.hstack((image_labels,image_labels2))
    image_labels = np.hstack((image_labels,image_labels3))
    image_labels = np.hstack((image_labels,image_labels4))
    
    # data generator
    test_generator1 = DataGenerator(batch_size=batch_size).generate(image_paths)
    test_generator2 = DataGenerator(batch_size=batch_size).generate(image_paths)
    test_generator3 = DataGenerator(batch_size=batch_size).generate(image_paths)
    
    #test_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        #samplewise_center = True,
        #samplewise_std_normalization = True)
    #    preprocessing_function = ReadInImages.preprocess_img,
    #    )


    #test_generator= test_datagen.flow_from_directory(
    #        test_data_path,
    #        target_size=(224, 224),
    #        shuffle=False,
    #        batch_size=batch_size,
    #        class_mode='categorical')
    

    
    # predict
    y_preds1 = model1.predict_generator(test_generator1,
                                      verbose=1,
                                      workers=16,
                                      use_multiprocessing=True,
                                      steps = len(image_paths)//batch_size)
    
    y_preds2 = model2.predict_generator(test_generator2,
                                      verbose=1,
                                      workers=16,
                                      use_multiprocessing=True,
                                      steps = len(image_paths)//batch_size)
    y_preds3 = model3.predict_generator(test_generator3,
                                      verbose=1,
                                      workers=16,
                                      use_multiprocessing=True,
                                      steps = len(image_paths)//batch_size)
    
    #X =  y_preds1
    X = np.hstack((y_preds1, y_preds2, y_preds3))
    
    X2 = np.hstack((y_preds1, y_preds2))
    X3 = np.hstack((y_preds2, y_preds3))
    X4 = np.hstack((y_preds1, y_preds3))
    #X = np.hstack(y_preds3)
    y = image_labels
    
    clf1 = RandomForestClassifier(n_estimators=100)
    clf1.fit(X,y)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf2.fit(X2,y)
    clf3 = RandomForestClassifier(n_estimators=100)
    clf3.fit(X3,y)
    clf4 = RandomForestClassifier(n_estimators=100)
    clf4.fit(X4,y)
    
    joblib.dump(clf1, 'clf1.pkl')
    joblib.dump(clf2, 'clf2.pkl')
    joblib.dump(clf3, 'clf3.pkl')
    joblib.dump(clf4, 'clf4.pkl')

    
    #y_out = np.argmax(y_preds,axis = 1) 
    #y_out = np.rint(y_out)
    
    # create and save submission to csv
    #create_submission(preds=y_out, 
    #                  names=image_names,
    #                  group='Gruppe2',
    #                  file_description='Multi_1',

#                      output_folder='./submissions')

if __name__=="__main__":
    main()

