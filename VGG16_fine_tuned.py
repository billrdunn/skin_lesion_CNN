import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.python.keras.models import Sequential

import utils


def download():
    vgg16_model = tf.keras.applications.VGG16(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax'
    )

    return vgg16_model


def define(vgg16, num_classes):
    # create a new model of type Sequential (this is what we have worked with previously)
    # and copy the layers (except the last one) from the original vgg16 model (of type Functional API)
    model = tf.keras.Sequential()
    for layer in vgg16.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False  # don't update these layers
    # add last layer with 7 nodes
    model.add(Dense(units=num_classes, activation='softmax'))

    return model
