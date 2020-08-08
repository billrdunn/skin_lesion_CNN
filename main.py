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
import utils
import VGG16

warnings.simplefilter(action='ignore', category=FutureWarning)

num_train = 110
num_valid = 11
num_test = 11
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
num_classes = len(classes)
paths = ['data/skin_lesion_images/train', 'data/skin_lesion_images/valid', 'data/skin_lesion_images/test']
batch_size = 10
target_size = (600, 450)  # resize pixel size of images to this. (600,450) is the normal size
input_shape = (target_size[0], target_size[1], 3)

# Uncomment below to replace images in training, validation and test sets
# copyImagesToDirs(num_train, num_valid, num_test)

# # Generate a batch of images and labels from the training set and visualise one batch
# imgs, labels = next(train_batches)
# utils.plotImages(imgs, True)
# # Print matrix of labels
# print(labels)

# ----- VGG16 MODEL -------
train_batches, valid_batches, test_batches = VGG16.createBatches(num_train, num_valid, num_test, num_classes,
                                                                 paths, target_size, classes, batch_size)
my_vgg16 = VGG16.define(input_shape, num_classes)
utils.trainAndSaveModel(my_vgg16,
                        'my_vgg16',
                        train_batches, len(train_batches),
                        valid_batches, len(valid_batches),
                        epochs=10,
                        overwrite=True)
VGG16.predict('my_vgg16', test_batches, classes, showCM=True)

# Fine-tune the VGG16 model
# VGG16 won ImageNet competition in 2014
# From the VGG16 paper, the only preprocessing that is done is subtracting
# the mean RGB pixel value from each pixel
# unfortunately the images have to be resized to (224,224,3) for pretraining with VGG16

# TODO uncomment everything below here and refactor
# if False:
#     # import VGG16 model
#     vgg16_model = tf.keras.applications.VGG16(
#         include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
#         pooling=None, classes=1000, classifier_activation='softmax'
#     )
#
#     # create a new model of type Sequential (this is what we have worked with previously)
#     # and copy the layers from the original vgg16 model (of type Functional API)
#     model = tf.keras.Sequential()
#     for layer in vgg16_model.layers[:-1]:
#         model.add(layer)
#     for layer in model.layers:
#         layer.trainable = False  # don't update these layers
#     # add last layer with 7 nodes
#     model.add(Dense(units=num_classes, activation='softmax'))
#
#     utils.trainAndSaveModel(model,
#                             'VGG16_pretrained_5_epochs',
#                             train_batches, len(train_batches),
#                             valid_batches, len(valid_batches),
#                             epochs=5,
#                             overwrite=True)
# if False:
#     utils.loadMakePredictionsAndPlotCM('models/VGG16_pretrained_5_epochs.h5',
#                                        x=test_batches,
#                                        steps=len(test_batches),
#                                        y_true=test_batches.classes,
#                                        classLabels=classes,
#                                        showCM=True
#                                        )
#
# # -------- MOBILENET MODEL ----------
# # see paper!
#
# train_path = 'data/skin_lesion_images/train'
# valid_path = 'data/skin_lesion_images/valid'
# test_path = 'data/skin_lesion_images/test'
#
# batch_size = 10
# target_size = (600, 450)  # resize pixel size of images to this. (600,450) is the normal size
# # (600,450) seems to give better test acc than (224,224)
# input_shape = (target_size[0], target_size[1], 3)
#
# train_batches = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=train_path, target_size=target_size, batch_size=batch_size)
# valid_batches = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=valid_path, target_size=target_size, batch_size=batch_size)
# test_batches = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=test_path, target_size=target_size, batch_size=batch_size, shuffle=False)
#
# if False:
#     # Do some nice assertions
#     assert train_batches.n == num_train * num_classes
#     assert valid_batches.n == num_valid * num_classes
#     assert test_batches.n == num_test * num_classes
#     assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == num_classes
#
#     mobile = tf.keras.applications.mobilenet.MobileNet()
#     # don't include last 23 layers of original mobileNet (found through experimentation)
#     x = mobile.layers[-6].output
#     output = Dense(units=7, activation='softmax')(x)
#     model = Model(inputs=mobile.input, outputs=output)
#     for layer in model.layers[:-23]:
#         layer.trainable = False
#     model.summary()
#
#     utils.trainAndSaveModel(model,
#                             'mobile_10epochs_3_224x224',
#                             train_batches, len(train_batches),
#                             valid_batches, len(valid_batches),
#                             epochs=10,
#                             overwrite=True)
#
# if False:
#     utils.loadMakePredictionsAndPlotCM('models/mobile_10epochs_3_600x450.h5',
#                                        x=test_batches,
#                                        steps=len(test_batches),
#                                        y_true=test_batches.classes,
#                                        classLabels=classes,
#                                        showCM=True
#                                        )
