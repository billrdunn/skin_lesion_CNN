import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
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

# ---------------------------------------------------------
# UNCOMMENT THIS BLOCK TO REPLACE ALL IMAGES
# ---------------------------------------------------------
# os.chdir('data/skin_lesion_images')
# utils.makeDirectories() # THIS WILL REMOVE ALL IMAGES
# utils.copyImagesToSetsDirs(num_train, num_valid, num_test)
# os.chdir('../../')
# ----------------------------------------------------------

train_path = 'data/skin_lesion_images/train'
valid_path = 'data/skin_lesion_images/valid'
test_path = 'data/skin_lesion_images/test'

# Creates batches of data 
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
batch_size = 10
target_size = (450, 600) # resize pixel size of images to this. (600,450) is the normal size
# TODO or is the right size (450, 600) ??
input_shape = (450, 600, 3)

# TODO This uses VGG16 preprocessing but maybe this isn't good...
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

if False: # Change this to retrain CNN
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

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epochs=10,
        verbose=2
    )

    # Save model in full
    utils.saveModelFull(model, 'first_CNN', overwrite=True)

# Load a full model
from tensorflow.keras.models import load_model
model = load_model('models/first_CNN.h5')

# Make predictions
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=1)

# Plot a confusion matrix
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
test_batches.class_indices
cm_plot_labels = classes
utils.plotConfusionMatrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix', show=False)