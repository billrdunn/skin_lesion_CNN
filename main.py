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

from tensorflow.python.estimator import gc

import utils
import VGG16
import VGG16_fine_tuned
import mobileNet

warnings.simplefilter(action='ignore', category=FutureWarning)
# ----- FORCING DETERMINISTIC (REPRODUCIBLE) MODELS --------
# (see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development)
# note that using GPUs is non-deterministic as many operations are run in parallel and the order is not always the same
# to overcome this we can force the code to run on a CPU using:
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Set a random seed for numpy (choice of int is arbitrary)
np.random.seed(53)
# Set a random seed for Python
rn.seed(32)
# Set a random seed for TensorFlow
tf.random.set_seed(66)

# Set up some global variables
num_train = 105
num_valid = 23
num_test = 22
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
num_classes = len(classes)
paths = ['data/skin_lesion_images/train', 'data/skin_lesion_images/valid', 'data/skin_lesion_images/test']
batch_size = 10

# Uncomment below to replace images in training, validation and test sets
# utils.copyImagesToDirs(num_train, num_valid, num_test)

# # Generate a batch of images and labels from the training set and visualise one batch
# imgs, labels = next(train_batches)
# utils.plotImages(imgs, True)
# # Print matrix of labels
# print(labels)


# -------- MOBILENET MODEL ----------
target_size = (224, 224)  # resize pixel size of images to this. (600,450) is the normal size
input_shape = (target_size[0], target_size[1], 3)

train_batches, valid_batches, test_batches = mobileNet.createBatches(num_train, num_valid, num_test, num_classes,
                                                                     paths, target_size, classes, batch_size)
test_accs = []
for param in range(1, 86, 5):
    model_name = '224x224_layers=(1,86,5)_lr=1-e04' + str(param)
    utils.train_a_model(model_name, num_classes, 'softmax', train_batches, valid_batches, test_batches, param, 10,
                        Adam(learning_rate=1e-4),
                        'categorical_crossentropy', ['accuracy'])

    test_accs.append(utils.loadMakePredictionsAndPlotCM(model_name,
                                                        x=test_batches,
                                                        steps=len(test_batches),
                                                        y_true=test_batches.classes,
                                                        classLabels=classes,
                                                        showCM=False
                                                        ))
    print(test_accs)
np.save('224x224_layers=(1,86,5)_lr=1-e04.npy', test_accs)

# layers_retrained = [1, 11, 21, 31, 41, 51, 61, 71, 81]
# test_accs_224x224 = np.load('test_accs_224x224.npy')
# test_accs_600x450 = np.load('test_accs_600x450.npy')
# plt.plot(layers_retrained, test_accs_224x224, '-')
# plt.plot(layers_retrained, test_accs_600x450, '-')
# plt.legend(['224x224', '600x450'])
# plt.xlabel('Number of layers of MobileNet retrained')
# plt.ylabel('Test accuracy (%)')
# plt.title(
#     'Effect of input dimensions on test accuracy for different numbers of layers retrained in the MobileNet model')
# #plt.show()
# plt.savefig('graphs/test_acc_vs_layers_retrained_for_two_input_dimensions.png')
# plt.close()
