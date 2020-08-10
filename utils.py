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
from tensorflow.keras.models import load_model


def copyImagesToDirs(num_train, num_valid, num_test):
    os.chdir('data/skin_lesion_images')
    makeDirectories()  # THIS WILL REMOVE ALL IMAGES
    copyImagesToSetsDirs(num_train, num_valid, num_test)
    os.chdir('../../')


def plotImages(images_arr, showPlot=False):
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
    for i in random.sample(glob.glob('*akiec*'), num_train):
        shutil.copy(i, 'train/akiec')
    for i in random.sample(glob.glob('*bcc*'), num_train):
        shutil.copy(i, 'train/bcc')
    for i in random.sample(glob.glob('*bkl*'), num_train):
        shutil.copy(i, 'train/bkl')
    for i in random.sample(glob.glob('*df*'), num_train):
        shutil.copy(i, 'train/df')
    for i in random.sample(glob.glob('*mel*'), num_train):
        shutil.copy(i, 'train/mel')
    for i in random.sample(glob.glob('*nv*'), num_train):
        shutil.copy(i, 'train/nv')
    for i in random.sample(glob.glob('*vasc*'), num_train):
        shutil.copy(i, 'train/vasc')
    for i in random.sample(glob.glob('*akiec*'), num_valid):
        shutil.copy(i, 'valid/akiec')
    for i in random.sample(glob.glob('*bcc*'), num_valid):
        shutil.copy(i, 'valid/bcc')
    for i in random.sample(glob.glob('*bkl*'), num_valid):
        shutil.copy(i, 'valid/bkl')
    for i in random.sample(glob.glob('*df*'), num_valid):
        shutil.copy(i, 'valid/df')
    for i in random.sample(glob.glob('*mel*'), num_valid):
        shutil.copy(i, 'valid/mel')
    for i in random.sample(glob.glob('*nv*'), num_valid):
        shutil.copy(i, 'valid/nv')
    for i in random.sample(glob.glob('*vasc*'), num_valid):
        shutil.copy(i, 'valid/vasc')
    for i in random.sample(glob.glob('*akiec*'), num_test):
        shutil.copy(i, 'test/akiec')
    for i in random.sample(glob.glob('*bcc*'), num_test):
        shutil.copy(i, 'test/bcc')
    for i in random.sample(glob.glob('*bkl*'), num_test):
        shutil.copy(i, 'test/bkl')
    for i in random.sample(glob.glob('*df*'), num_test):
        shutil.copy(i, 'test/df')
    for i in random.sample(glob.glob('*mel*'), num_test):
        shutil.copy(i, 'test/mel')
    for i in random.sample(glob.glob('*nv*'), num_test):
        shutil.copy(i, 'test/nv')
    for i in random.sample(glob.glob('*vasc*'), num_test):
        shutil.copy(i, 'test/vasc')


def saveModelFull(model, name, overwrite=False):
    if overwrite:
        print("Saving model in full...")
        try:
            model.save('models/' + name + '.h5')
        except:
            print("Did not save!")
    else:
        if os.path.isfile('models/' + name + '.h5') is False:
            print("Saving model in full...")
            try:
                model.save('models/' + name + '.h5')
            except:
                print("Did not save!")


def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.get_cmap("Blues"),
                        show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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


def trainAndSaveModel(
        model,
        saveAs,
        x,
        steps_per_epoch,
        validation_data,
        validation_steps,
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        epochs=10,
        verbose=2,
        overwrite=False):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train model
    model.fit(x=x,
              steps_per_epoch=steps_per_epoch,
              validation_data=validation_data,
              validation_steps=validation_steps,
              epochs=epochs,
              verbose=verbose
              )

    # Save model in full
    saveModelFull(model, saveAs, overwrite=True)


def loadMakePredictionsAndPlotCM(model_name, x, steps, y_true, classLabels, verbose=1, showCM=False):
    modelDir = 'models/' + model_name + '.h5'
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
