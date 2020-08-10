import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.python.keras.models import Sequential
from keras.models import Model


import utils


def createBatches(num_train, num_valid, num_test, num_classes, paths, target_size, classes, batch_size):
    train_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=paths[0], target_size=target_size, batch_size=batch_size)
    valid_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=paths[1], target_size=target_size, batch_size=batch_size)
    test_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=paths[2], target_size=target_size, batch_size=batch_size, shuffle=False)
    # Do some nice assertions
    assert train_batches.n == num_train * num_classes
    assert valid_batches.n == num_valid * num_classes
    assert test_batches.n == num_test * num_classes
    assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == num_classes

    return train_batches, valid_batches, test_batches


def define(input_shape, num_classes, layers_to_not_include=23):
    # Create the CNN
    mobile = tf.keras.applications.mobilenet.MobileNet()
    # don't include last 23 layers of original mobileNet (found through experimentation)
    x = mobile.layers[-6].output
    output = Dense(units=7, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-layers_to_not_include]:
        layer.trainable = False

    return model


def trainAndSave(name, model, num_classes, input_shape, train_batches, valid_batches, overwrite=True):
    utils.trainAndSaveModel(model,
                            name,
                            x=train_batches,
                            steps_per_epoch=len(train_batches),
                            validation_data=valid_batches,
                            validation_steps=len(valid_batches),
                            overwrite=overwrite)


