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
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal, glorot_uniform, glorot_normal
from keras import backend as K
from keras import losses
import keras
import numpy as np

SEED = 4645
np.random.seed(SEED)

# dimensions of our images.
img_width, img_height = 512, 512
num_classes = 5
lr = 0.0001
batch_size = 16
pool_size = (2,2)

train_data_dir = 'train_res/training'
validation_data_dir = 'train_res/vali'
models_dir = 'models'
nb_train_samples = 2850
nb_validation_samples = 995
epochs = 200
sgd_momentum = 0.5

lr_decay = lr/epochs
#loss_function = losses.mean_squared_logarithmic_error
loss_function = losses.categorical_crossentropy



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
# weight initialization
weight_init = glorot_normal(seed=SEED)

# activation function
activation_function = 'relu'    

#greate model
model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(2,2),
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 strides = 1,
                 input_shape=input_shape))
#model.add(BatchNormalization())
model.add(Conv2D(filters=16,
                 kernel_size=(2,2),
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 strides = 1,
                 input_shape=input_shape))
model.add(Conv2D(filters=16,
                 kernel_size=(2,2),
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 strides = 1))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))

model.add(BatchNormalization())
model.add(Conv2D(filters=32,
                 kernel_size=(2,2),
                 strides = 1,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(Conv2D(filters=32,
                 kernel_size=(2,2),
                 strides = 1,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,
                 kernel_size=(2,2),
                 strides = 1,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(Conv2D(filters=64,
                 kernel_size=(2,2),
                 strides = 1,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))

model.add(Conv2D(filters=96,
                 kernel_size=(2,2),
                 strides = 1,
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128,
                 kernel_size=(2,2),
                 kernel_initializer=weight_init,
                 activation=activation_function,
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2)))
model.add(BatchNormalization())
#model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(units=192,
                kernel_initializer=weight_init,
                activation=activation_function))
model.add(Dropout(0.5))
model.add(Dense(units96,
                kernel_initializer=weight_init,
                activation=activation_function))
model.add(Dense(units=num_classes,
                kernel_initializer=weight_init,
                activation='softmax'))

# compile

#sgd = SGD(lr=lr, momentum=sgd_momentum, nesterov=False, decay=lr_decay)
rms = keras.optimizers.RMSprop(lr=0.00001)

model.compile(optimizer=rms,
              loss=loss_function,
              metrics=['accuracy'])

    
# Callbacks
tensorboard = TensorBoard(log_dir='./logs/L_M5')
filename="L_M40-e{epoch:02d}-val_acc{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(models_dir + '/' + filename, monitor='val_acc', save_best_only = True)

#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    samplewise_center = True,
    samplewise_std_normalization = True,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range = 180,
    horizontal_flip=True,
    vertical_flip = True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center = True,
        samplewise_std_normalization = True)

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
    callbacks=[tensorboard,checkpoint])

model.save_weights(models_dir + '/L_weights40.h5')
architecture = model.to_json()
with open (models_dir+'/L_architecture40.txt', 'w') as txt:
    txt.write(architecture)
model.save(models_dir + '/L_model40.h5')
