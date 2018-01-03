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

    def __init__(self, batch_size=32):
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
                    
                    # Preprocessing
                    
                    batch_x.append(np.array(im))
                
                yield np.array(batch_x)
        
def main():        
    # load model
    models_dir = '../models/'
    model = load_model(models_dir + 'L_M4/L_M4-e50-val_acc0.43.hdf5')

    
    # image paths
    test_data_path = '../test_res/images'
    image_paths = glob(test_data_path+'/*.tiff')
    
    # image names
    image_names = [im.split('/')[-1].split('.')[0] for im in image_paths]
    
    # data generator
    test_generator = DataGenerator(batch_size=32).generate(image_paths)
    
    # predict
    y_preds = model.predict_generator(test_generator,
                                      verbose=1,
                                      steps = len(image_paths)//32)
    
    y_out = np.argmax(y_preds,axis = 1) 
    y_out = np.rint(y_out)
    
    # create and save submission to csv
    create_submission(preds=y_out, 
                      names=image_names,
                      group='Gruppe2',
                      file_description='model1_L',
                      output_folder='./submissions')

if __name__=="__main__":
    main()
