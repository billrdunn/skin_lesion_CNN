import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.python.keras.models import Sequential

import utils


def createBatches(num_train, num_valid, num_test, num_classes, paths, target_size, classes, batch_size):
    # Creates batches of data
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=paths[0], target_size=target_size, classes=classes, batch_size=batch_size)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=paths[1], target_size=target_size, classes=classes, batch_size=batch_size)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=paths[2], target_size=target_size, classes=classes, batch_size=batch_size,
                             shuffle=False)
    # Do some nice assertions
    assert train_batches.n == num_train * num_classes
    assert valid_batches.n == num_valid * num_classes
    assert test_batches.n == num_test * num_classes
    assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == num_classes

    return train_batches, valid_batches, test_batches


def define(input_shape, num_classes):
    # Create the CNN
    # padding = same means dimensionality isn't reduced after convolution
    # MaxPool2D strides = 2 cuts dimensions in half
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),  # flatten to 1D before passing to final layer
        Dense(units=num_classes, activation='softmax')
    ])
    return model


def trainAndSave(name, model, num_classes, input_shape, train_batches, valid_batches, overwrite=True):
    utils.trainAndSaveModel(model,
                            name,
                            x=train_batches,
                            steps_per_epoch=len(train_batches),
                            validation_data=valid_batches,
                            validation_steps=len(valid_batches),
                            overwrite=overwrite)


