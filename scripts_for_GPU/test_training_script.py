import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import random as rn
import matplotlib.pyplot as plt
import warnings
import sys

sys.path.insert(0, '/home/cc19563/skin_lesion_CNN')
print(sys.path)
import utils
import mobileNet

warnings.simplefilter(action='ignore', category=FutureWarning)
# ----- FORCING DETERMINISTIC (REPRODUCIBLE) MODELS --------
# (see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development)
# note that using GPUs is non-deterministic as many operations are run in parallel and the order is not always the same
# to overcome this we can force the code to run on a CPU using:
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
paths = ['/home/cc19563/skin_lesion_CNN/data/skin_lesion_images/train', '/home/cc19563/skin_lesion_CNN/data'
                                                                        '/skin_lesion_images/valid',
         '/home/cc19563/skin_lesion_CNN/data/skin_lesion_images/test']
batch_size = 10

# -------- MOBILENET MODEL ----------
target_size = (224, 224)  # resize pixel size of images to this. (600,450) is the original size
input_shape = (target_size[0], target_size[1], 3)

train_batches, valid_batches, test_batches = mobileNet.createBatches(num_train, num_valid, num_test, num_classes,
                                                                     paths, target_size, classes, batch_size)
test_accs = []
lrs = np.linspace(1e-4, 1e-6, 1)
save_name = 'test_lr=(1e-4,1e-6,3)'
print(lrs)
for lr in lrs:
    model_name = save_name + '_' + str(lr)
    utils.train_a_model(model_name, num_classes, 'relu', train_batches, valid_batches, 10, 1,
                        Adam(learning_rate=lr),
                        'categorical_crossentropy', ['accuracy'])

    test_accs.append(utils.loadMakePredictionsAndPlotCM(model_name,
                                                        x=test_batches,
                                                        steps=len(test_batches),
                                                        y_true=test_batches.classes,
                                                        classLabels=classes,
                                                        showCM=False
                                                        ))
    print(test_accs)
np.save('../generated_data/' + save_name + '.npy', test_accs)

# layers_retrained = [i for i in lrs]
test_accs = np.load('../generated_data/' + save_name + '.npy')
plt.plot(lrs, test_accs, '-')
# plt.legend(['224x224', '600x450'])
plt.xlabel('Learning rate')
plt.ylabel('Test accuracy (%)')
plt.title(
    'Test model graph')
# plt.show()
plt.savefig('../graphs/' + save_name + '.png')
plt.close()
