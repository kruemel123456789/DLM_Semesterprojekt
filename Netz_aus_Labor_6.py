#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:28:44 2017

@author: root
"""


import numpy as np

SEED = 4645
np.random.seed(SEED)
import keras

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal, glorot_uniform

from sklearn.model_selection import train_test_split

num_classes = 10

# Hyperparameters
batch_size = 64
epochs = 30
lr = 1e-4


def load_data(grayscale, scale):
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # convert to grayscale
    if grayscale:
        X_train = rgb2gray(X_train)
        X_test = rgb2gray(X_test)
        # Add single channel axis
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
    
    # split in train val
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=SEED)
    
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    if scale:
        # Scale data
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_train /= 255.
        X_val /= 255.
        X_test /= 255.
    
    return X_train, y_train, X_val, y_val, X_test, y_test 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def main():
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(grayscale=False,
                                                               scale=False)
    
    # weight initialization
    weight_init = glorot_uniform(seed=SEED)
    
    # activation function
    activation_function = 'relu'
    
    # Create model
    model = Sequential()
    model.add(Conv2D(filters=16,
                     kernel_size=5,
                     kernel_initializer=weight_init,
                     activation=activation_function,
                     input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32,
                     kernel_size=5,
                     kernel_initializer=weight_init,
                     activation=activation_function,
                     input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=64,
                     kernel_size=5,
                     kernel_initializer=weight_init,
                     activation=activation_function))
    model.add(Conv2D(filters=128,
                     kernel_size=5,
                     kernel_initializer=weight_init,
                     activation=activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128,
                    kernel_initializer=weight_init,
                    activation=activation_function))
    model.add(Dense(units=64,
                    kernel_initializer=weight_init,
                    activation=activation_function))
    model.add(Dense(units=32,
                    kernel_initializer=weight_init,
                    activation=activation_function))
    model.add(Dense(units=num_classes,
                    kernel_initializer=weight_init,
                    activation='softmax'))
    
    # compile
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    tensorboard = TensorBoard(log_dir='./logs/7:kernel:5-5-5-5')
    
    # train
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              callbacks=[tensorboard])
    
    # test
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test loss: {} \nTest accuracy: {}%".format(score[0], score[1]*100))
    
    print(model.summary())

if __name__ == "__main__":
    main()
