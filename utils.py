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
from keras.models import Model

warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.keras.models import load_model


def copyImagesToDirs(num_train, num_valid, num_test, child_dir='data/skin_lesion_images'):
    """
    Copies images randomly from the root directory to folders corresponding to training, validation, and test sets.
    WARNING: will replace all images currently in these directories.

    Args:
        num_train: number of training images to put in training directory
        num_valid: number of validation images to put in validation directory
        num_test: number of test images to put in test directory

    Returns: None
    """
    os.chdir(child_dir)
    makeDirectories()  # THIS WILL REMOVE ALL IMAGES
    copyImagesToSetsDirs(num_train, num_valid, num_test)
    os.chdir('../../')


def plotImages(images_arr, showPlot=False):
    """
     Plots one image array batch for visualising training procedure.

     Args:
         images_arr: batch of images
         showPlot: if True, display plot

     Returns: None
     """
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    if showPlot:
        plt.show()


def replaceDirTree(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def makeDirectories():
    trees = ['train/akiec', 'train/bcc', 'train/bkl', 'train/df', 'train/mel', 'train/nv', 'train/vasc',
             'test/akiec', 'test/bcc', 'test/bkl', 'test/df', 'test/mel', 'test/nv', 'test/vasc',
             'valid/akiec', 'valid/bcc', 'valid/bkl', 'valid/df', 'valid/mel', 'valid/nv', 'valid/vasc']
    for tree in trees:
        replaceDirTree(tree)


def copyImagesToSetsDirs(num_train, num_valid, num_test):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for c in classes:
        for i in random.sample(glob.glob('*' + c + '*'), num_train):
            shutil.copy(i, 'train/' + c)
        for i in random.sample(glob.glob(('*' + c + '*')), num_valid):
            shutil.copy(i, 'valid/' + c)
        for i in random.sample(glob.glob('*' + c + '*'), num_test):
            shutil.copy(i, 'test/' + c)


def saveModelFull(model, name, overwrite):
    if overwrite:
        print("Saving model in full...")
        try:
            model.save('models/' + name + '.h5')
        except:
            print("Did not save! Error 1")
    else:
        if os.path.isfile('models/' + name + '.h5') is False:
            print("Saving model in full...")
            try:
                model.save('models/' + name + '.h5')
            except:
                print("Did not save! Overwrite = false and file already exists")


def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.get_cmap("Blues"),
                        show=False):
    """
     Plots the confusion matrix for the test set classification on one model.

     Args:
         cm: confusion matrix
         classes: list of class labels
         normalize: if True, normalize matrix
         title: title of plot
         cmap: colour map
         show: if True, display matrix

     Returns: None
     """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if show:
        plt.show()


def loadMakePredictionsAndPlotCM(model_name, x, steps, y_true, classLabels, verbose=1, showCM=False):
    """
     Loads a saved model, makes predictions on it, plots a confusion matrix, and returns the test accuracy.

     Args:
         model_name: name of model in directory with extension '.h5'
         x: test batches
         steps: required for model.predict, equal to len(x) in most applications
         y_true: labels (ground truth) for test set
         classLabels: list of class labels
         verbose: level of information in output
         showCM: if True, display confusion matrix

     Returns: test accuracy for x evaluated on model_name
     """
    modelDir = '../models/' + model_name + '.h5'
    # Load a full model
    model = load_model(modelDir)

    # Make predictions
    predictions = model.predict(x=x, steps=steps, verbose=verbose)

    predictions_list = []
    for prediction in predictions:
        predictions_list.append(np.argmax(prediction))
    predictions_list = np.array(predictions_list)

    temp = sum(predictions_list == y_true)
    test_acc = temp / len(y_true)

    # Plot a confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=np.argmax(predictions, axis=-1))
    x.class_indices
    cm_plot_labels = classLabels
    plotConfusionMatrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix', show=showCM)

    return test_acc


def train_a_model(save_as, num_classes, activation, train_batches, valid_batches, layers_to_retrain,
                  epochs,
                  optimizer, loss, metrics):
    """
     Retrains a number of layers of an existing model and saves it in full with the '.h5' extension.

     Args:
         save_as: name to save model as in directory
         num_classes: number of possible classifications
         activation: type of activation in output layer, eg. 'softmax', 'relu'
         train_batches: training batches used to train model
         valid_batches: validation batches used to monitor validation accuracy during training
         layers_to_retrain: number of layers of the existing model to retrain
         epochs: number of training epochs to use
         optimizer: type of optimizer to use in training - usual is Adam
         loss: type of loss to use - usual is 'categorical_crossentropy'
         metrics: list of metrics on which to evaluate, eg. ['accuracy]


     Returns: None
     """

    print(f"Training a model with {layers_to_retrain} layers retrained...")
    # Create the CNN
    mobile = tf.keras.applications.mobilenet.MobileNet()
    # don't include last layers of original mobileNet (found through experimentation)
    x = mobile.layers[-6].output
    output = Dense(units=num_classes, activation=activation)(x)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-layers_to_retrain]:
        layer.trainable = False
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x=train_batches,
              steps_per_epoch=len(train_batches),
              validation_data=valid_batches,
              validation_steps=len(valid_batches),
              epochs=epochs,
              verbose=2
              )
    model.save('../models/' + save_as + '.h5')
