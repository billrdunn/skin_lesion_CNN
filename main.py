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
import random as rn
import glob
import matplotlib.pyplot as plt
import warnings
import utils
import VGG16
import VGG16_fine_tuned
import mobileNet

warnings.simplefilter(action='ignore', category=FutureWarning)

num_train = 110
num_valid = 15
num_test = 20
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
num_classes = len(classes)
paths = ['data/skin_lesion_images/train', 'data/skin_lesion_images/valid', 'data/skin_lesion_images/test']
batch_size = 10
target_size = (224, 224)  # resize pixel size of images to this. (600,450) is the normal size
input_shape = (target_size[0], target_size[1], 3)

# Uncomment below to replace images in training, validation and test sets
utils.copyImagesToDirs(num_train, num_valid, num_test)

# # Generate a batch of images and labels from the training set and visualise one batch
# imgs, labels = next(train_batches)
# utils.plotImages(imgs, True)
# # Print matrix of labels
# print(labels)

# ----- VGG16 MODEL -------
# model_name = 'my_vgg16'
# # Generate batches
# train_batches, valid_batches, test_batches = VGG16.createBatches(num_train, num_valid, num_test, num_classes,
#                                                                  paths, target_size, classes, batch_size)
# # Define my model
# my_vgg16 = VGG16.define(input_shape, num_classes)
#
# # Train model and save as
# utils.trainAndSaveModel(my_vgg16,
#                         'my_vgg16',
#                         train_batches, len(train_batches),
#                         valid_batches, len(valid_batches),
#                         epochs=10,
#                         overwrite=True)
#
# # Load and predict and plot confusion matrix
# utils.loadMakePredictionsAndPlotCM(model_name, test_batches, len(test_batches), test_batches.classes, classes, showCM=True)

# # ------ Fine-tune existing VGG16 model -----
# # NOTE this only works with target_size = (224,224)
# # Generate batches
# train_batches, valid_batches, test_batches = VGG16.createBatches(num_train, num_valid, num_test, num_classes,
#                                                                  paths, target_size, classes, batch_size)
# model_name = 'my_vgg16_finetuned'
# # Import the original model
# # VGG16 won ImageNet competition in 2014
# vgg16 = VGG16_fine_tuned.download()

# # Create a new model based on VGG16
# my_vgg16_finetuned = VGG16_fine_tuned.define(vgg16, num_classes)
#
# # Train model and save
# utils.trainAndSaveModel(my_vgg16_finetuned,
#                         model_name,
#                         train_batches, len(train_batches),
#                         valid_batches, len(valid_batches),
#                         epochs=10,
#                         overwrite=True)

# # Load and predict and plot confusion matrix
# utils.loadMakePredictionsAndPlotCM(model_name, test_batches, len(test_batches), test_batches.classes, classes, showCM=True)

# TODO uncomment everything below here and refactor


# -------- MOBILENET MODEL ----------

# model_name = 'my_mobilenet'
# # Generate batches
# train_batches, valid_batches, test_batches = mobileNet.createBatches(num_train, num_valid, num_test, num_classes,
#                                                                      paths, target_size, classes, batch_size)
# # # Create model
# # my_mobilenet = mobileNet.define(input_shape, num_classes, layers_to_not_include=23)
# #
# # # Save model
# # utils.trainAndSaveModel(my_mobilenet,
# #                         model_name,
# #                         train_batches, len(train_batches),
# #                         valid_batches, len(valid_batches),
# #                         epochs=10,
# #                         overwrite=True)
#
# # Load model and make predictions and return test_acc
# test_acc = utils.loadMakePredictionsAndPlotCM(model_name,
#                                               x=test_batches,
#                                               steps=len(test_batches),
#                                               y_true=test_batches.classes,
#                                               classLabels=classes,
#                                               showCM=True
#                                               )


# Investigation to see the optimum number of layers to retrain
train_batches, valid_batches, test_batches = mobileNet.createBatches(num_train, num_valid, num_test, num_classes,
                                                                     paths, target_size, classes, batch_size)
test_accs = []
for layers in range(1, 31, 3):
    print("Testing with layers = " + str(layers) + "...")
    model_name = 'mobilenet_224x224_layers=' + str(layers)
    # Create model
    my_mobilenet = mobileNet.define(input_shape, num_classes, layers_to_not_include=layers)
    # Save model
    utils.trainAndSaveModel(my_mobilenet,
                            model_name,
                            train_batches, len(train_batches),
                            valid_batches, len(valid_batches),
                            epochs=10,
                            overwrite=True)
    # Load model and make predictions and return test_acc
    test_accs.append(utils.loadMakePredictionsAndPlotCM(model_name,
                                                        x=test_batches,
                                                        steps=len(test_batches),
                                                        y_true=test_batches.classes,
                                                        classLabels=classes,
                                                        showCM=True
                                                        ))
    print(test_accs)
