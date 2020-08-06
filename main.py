import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.applications import imagenet_utils
from keras.preprocessing import image
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import utils

num_train = 110
num_valid = 11
num_test = 11
num_classes = 7
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


# ---------------------------------------------------------
# UNCOMMENT THIS BLOCK TO REPLACE ALL IMAGES
# ---------------------------------------------------------
# os.chdir('data/skin_lesion_images')
# utils.makeDirectories() # THIS WILL REMOVE ALL IMAGES
# utils.copyImagesToSetsDirs(num_train, num_valid, num_test)
# os.chdir('../../')
# ----------------------------------------------------------


# # Generate a batch of images and labels from the training set
# imgs, labels = next(train_batches)

# # Visualise one batch of training data
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 10, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     #plt.show()
# plotImages(imgs)

# # Print matrix of labels
# print(labels)

# ----- VGG16 MODEL -------

if False: # Change this to retrain CNN
    train_path = 'data/skin_lesion_images/train'
    valid_path = 'data/skin_lesion_images/valid'
    test_path = 'data/skin_lesion_images/test'

    batch_size = 10
    target_size = (600, 450) # resize pixel size of images to this. (600,450) is the normal size
    input_shape = (target_size[0], target_size[1], 3)

    # Creates batches of data 
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=target_size, classes=classes, batch_size=batch_size)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=valid_path, target_size=target_size, classes=classes, batch_size=batch_size)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=test_path, target_size=target_size, classes=classes, batch_size=batch_size, shuffle=False)

    # Do some nice assertions
    assert train_batches.n == num_train * num_classes
    assert valid_batches.n == num_valid * num_classes
    assert test_batches.n == num_test * num_classes
    assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == num_classes

    # Create the CNN
    # padding = same means dimensionality isn't reduced after convolution
    # MaxPool2D strides = 2 cuts dimensions in half
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=input_shape),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(), # flatten to 1D before passing to final layer
        Dense(units=num_classes, activation='softmax')
    ])

    utils.trainAndSaveModel(model,
        'first_CNN', 
        x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches), 
        overwrite=True)

if False:
    utils.loadMakePredictionsAndPlotCM('models/first_CNN.h5', 
        x=test_batches, 
        steps=len(test_batches),
        y_true=test_batches.classes,
        classLabels=classes,
        showCM=True
        )

# Fine-tune the VGG16 model
# VGG16 won ImageNet competition in 2014
# From the VGG16 paper, the only preprocessing that is done is subtracting
# the mean RGB pixel value from each pixel
# unfortunately the images have to be resized to (224,224,3) for pretraining with VGG16

if False:
    # import VGG16 model
    vgg16_model = tf.keras.applications.VGG16(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax'
    )

    # create a new model of type Sequential (this is what we have worked with previously)
    # and copy the layers from the original vgg16 model (of type Functional API)
    model = tf.keras.Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False # don't update these layers
    # add last layer with 7 nodes
    model.add(Dense(units=num_classes, activation='softmax'))

    utils.trainAndSaveModel(model, 
        'VGG16_pretrained_5_epochs', 
        train_batches, len(train_batches), 
        valid_batches, len(valid_batches), 
        epochs=5, 
        overwrite=True)
if False:
    utils.loadMakePredictionsAndPlotCM('models/VGG16_pretrained_5_epochs.h5', 
            x=test_batches, 
            steps=len(test_batches),
            y_true=test_batches.classes,
            classLabels=classes,
            showCM=True
            )

# -------- MOBILENET MODEL ----------
# see paper!

train_path = 'data/skin_lesion_images/train'
valid_path = 'data/skin_lesion_images/valid'
test_path = 'data/skin_lesion_images/test'

batch_size = 10
target_size = (600, 450) # resize pixel size of images to this. (600,450) is the normal size
# (600,450) seems to give better test acc than (224,224)
input_shape = (target_size[0], target_size[1], 3)

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=target_size, batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=target_size, batch_size=batch_size)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=target_size, batch_size=batch_size, shuffle=False)

if False:
    # Do some nice assertions
    assert train_batches.n == num_train * num_classes
    assert valid_batches.n == num_valid * num_classes
    assert test_batches.n == num_test * num_classes
    assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == num_classes
    
    mobile = tf.keras.applications.mobilenet.MobileNet()
    # don't include last 23 layers of original mobileNet (found through experimentation)
    x = mobile.layers[-6].output
    output = Dense(units=7, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-23]:
        layer.trainable = False
    model.summary()

    utils.trainAndSaveModel(model, 
        'mobile_10epochs_3_224x224', 
        train_batches, len(train_batches), 
        valid_batches, len(valid_batches), 
        epochs=10, 
        overwrite=True)

if True:
    utils.loadMakePredictionsAndPlotCM('models/mobile_10epochs_3_224x224.h5', 
            x=test_batches, 
            steps=len(test_batches),
            y_true=test_batches.classes,
            classLabels=classes,
            showCM=True
            )



