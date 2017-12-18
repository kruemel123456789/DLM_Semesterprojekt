'''https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal, glorot_uniform
from keras import backend as K
import keras
import numpy as np

SEED = 4645
np.random.seed(SEED)

#LETZTER DURCHLAUF: CA. 20%, Bis epoch 10 nicht konvergiert


# dimensions of our images.
img_width, img_height = 512, 512
num_classes = 5
lr = 0.5


train_data_dir = 'train_res/training'
validation_data_dir = 'train_res/vali'
nb_train_samples = 2850
nb_validation_samples = 995
epochs = 50
batch_size = 16

dr = lr/epochs

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
# weight initialization
weight_init = glorot_uniform(seed=SEED)

# activation function
activation_function = 'relu'    

#greate model
model = Sequential()
model.add(Conv2D(filters=40,
                 kernel_size=7,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(Conv2D(filters=40,
                 kernel_size=7,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#model.add(Dropout(0.5))
model.add(Conv2D(filters=40,
                 kernel_size=5,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same'))
model.add(Conv2D(filters=40,
                 kernel_size=5,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(filters=40,
                 kernel_size=5,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same'))
model.add(Conv2D(filters=40,
                 kernel_size=5,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(filters=40,
                 kernel_size=5,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same'))
model.add(Conv2D(filters=40,
                 kernel_size=5,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256,
                kernel_initializer=weight_init,
                activation=activation_function))
#model.add(Dropout(0.5))
model.add(Dense(units=256,
                kernel_initializer=weight_init,
                activation=activation_function))
#model.add(Activation('sigmoid'))
model.add(Dense(units=num_classes,
                kernel_initializer=weight_init,
                activation='softmax'))

# compile
sgd = SGD(lr=lr, momentum=0.9, nesterov=True, decay=dr)
Adamax = keras.optimizers.Adamax(lr = lr, decay = dr)
model.compile(optimizer=Adamax,
              loss='MSE',
              metrics=['accuracy'])
    
# Callbacks
tensorboard = TensorBoard(log_dir='./logs/003_more_layers')
    

#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical')

model.summary()

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=1,
    callbacks=[tensorboard])

model.save_weights('first_try.h5')