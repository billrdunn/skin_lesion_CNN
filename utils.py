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

def replaceDirTree(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def makeDirectories():
    replaceDirTree('train/akiec')
    replaceDirTree('train/bcc')
    replaceDirTree('train/bkl')
    replaceDirTree('train/df')
    replaceDirTree('train/mel')
    replaceDirTree('train/nv')
    replaceDirTree('train/vasc')
    replaceDirTree('test/akiec')
    replaceDirTree('test/bcc')
    replaceDirTree('test/bkl')
    replaceDirTree('test/df')
    replaceDirTree('test/mel')
    replaceDirTree('test/nv')
    replaceDirTree('test/vasc')
    replaceDirTree('valid/akiec')
    replaceDirTree('valid/bcc')
    replaceDirTree('valid/bkl')
    replaceDirTree('valid/df')
    replaceDirTree('valid/mel')
    replaceDirTree('valid/nv')
    replaceDirTree('valid/vasc')

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
        try: model.save('models/' + name + '.h5')
        except: print("Did not save!")
    else:
        if os.path.isfile('models/' + name + '.h5') is False:
            print("Saving model in full...")
            try: model.save('models/' + name + '.h5')
            except: print("Did not save!")


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