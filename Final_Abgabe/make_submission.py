# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:34:23 2017

@author: root
"""
import os
import numpy as np
import pandas as pd

from glob import glob
from keras.models import load_model

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
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Get the data from a and b
                idxs = indexes[i*self.batch_size:(i+1)*self.batch_size]
                # load images
                batch_x = []
                for idx in idxs:
                    im = Image.open(data_list[idx])
                    #print(data_list[idx])
                    #batch_x.append(data_list[idx])
                    # Preprocessing
                    im = im.resize((224, 224))
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
    model = load_model(models_dir + 'F_MVGG_8-sgd-moreEpochs.hdf5')


    batch_size = 64
    
    # image paths
    test_data_path = 'test_res/images'

    image_paths = glob(test_data_path+'/*.tiff')
    
    # image names
    image_names = [im.split('/')[-1].split('.')[0] for im in image_paths]
    
    # data generator
    test_generator = DataGenerator(batch_size=batch_size).generate(image_paths)


        
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
    y_preds = model.predict_generator(test_generator,
                                      verbose=1,
                                      steps = len(image_paths)//batch_size)#
    
    
    y_out = np.argmax(y_preds,axis = 1) 
    y_out = np.rint(y_out)
    
    # create and save submission to csv
    create_submission(preds=y_out, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='testout',

                      output_folder='./submissions')

if __name__=="__main__":
    main()

