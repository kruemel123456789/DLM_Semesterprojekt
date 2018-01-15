from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal, glorot_uniform, glorot_normal
from keras import backend as K
from keras import losses
import keras
import numpy as np
import ReadInImages
from keras.models import load_model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input


from datetime import datetime
now = datetime.now()

#Vergabe der Dateinamen
N_OF_RUN = "021" + now.strftime("-%d_%m-%H_%M")
DESCRIPTION_OF_RUN = "_removed_batch_norm"

#Einstellbare Parameter (OFT)
lr = 1e-5
batch_size = 16
pool_size = (2,2)
epochs = 300
sgd_momentum = 0.9
lr_decay = 1e-6
loss_function = losses.categorical_crossentropy
activation_function = 'relu'
weight_init = glorot_normal(seed=SEED)

#Random SEED
SEED = 4645
np.random.seed(SEED)

#Dimensionen von Eingang (Bild) und Audgabe (Kategorien)
img_width, img_height = 224, 224
num_classes = 5

#Verzeichnisse
train_data_dir = 'train_res/training'
debug_dir = 'debugPics'
validation_data_dir = 'train_res/vali'
models_dir = 'models/'

#Anzahl der Dateien
nb_train_samples = 2850
nb_validation_samples = 500



#Mit aktuellem PreProcessing muss es fuer eigenes Modell first sein!
#K.set_image_data_format("channels_first")

#VGG16/19 nutzen channels_last
K.set_image_data_format("channels_last")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

#Optimizers
sgd = SGD(lr=lr, momentum=sgd_momentum, nesterov=False, decay=lr_decay)
rms = keras.optimizers.RMSprop(lr=lr)

#VGG laden
base_model = VGG19(weights = 'imagenet', input_shape=input_shape,include_top = False)

#Untere Layer definieren und zur VGG-Architektur hinzufuegen
x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, kernel_initializer = weight_init, activation = activation_function)(x)
x = Dense(192, kernel_initializer = weight_init, activation = activation_function)(x)
x = Dense(num_classes, kernel_initializer=weight_init,
                activation='softmax')(x)
model = keras.models.Model(inputs = base_model.input, outputs = x)

#Modell zum weiterlernen laden (anderes Modell wird ueberschrieben)
#model = load_model(models_dir + 'F_MVGG_10.hdf5')

#Ausgabe der Modellarchitektur
model.summary()

#Modell compilen
model.compile(optimizer=sgd,
              loss=loss_function,
              metrics=['accuracy'])

    
# Callbacks fuer Tensorboard und Checkpoints
tensorboard = TensorBoard(log_dir=('./logs/' + N_OF_RUN + DESCRIPTION_OF_RUN))

filename=N_OF_RUN +"ModCheck_E{epoch:02d}_valACC{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(models_dir + filename, monitor='val_acc', save_best_only = True)


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    #featurewise_center = True,
    #featurewise_std_normalization = True,
    #rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 360,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode = "constant",   #Raender schwarz auffuellen
    cval = 0.,
    preprocessing_function = ReadInImages.preprocess_img, #Preprocessing falls vorgelerntes Modell geladen
    )

#Nur falls featurewise_center oder std
#fitData = ReadInImages.getAveragePics()

#train_datagen.fit(fitData)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        preprocessing_function = ReadInImages.preprocess_img,
        )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical', save_to_dir = None)#debug_dir)


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    shuffle=False,
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=1,
    callbacks=[tensorboard,checkpoint])

#Ausgabe des Fertigen Modells mit Architektur und Gewichten
model.save_weights(models_dir + 'weights' + N_OF_RUN + '.h5')
architecture = model.to_json()
with open (models_dir + 'architecture' + N_OF_RUN + '.txt', 'w') as txt:
    txt.write(architecture)
model.save(models_dir + 'model' + N_OF_RUN + '.h5')
