import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import random as rn
import matplotlib.pyplot as plt
import warnings
import sys
import time
start_time = time.time()

from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

sys.path.insert(0, '/home/cc19563/skin_lesion_CNN')
# print(sys.path)
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
train_dir = '/home/cc19563/skin_lesion_CNN/data/skin_lesion_images/train'
valid_dir = '/home/cc19563/skin_lesion_CNN/data/skin_lesion_images/valid'
test_dir = '/home/cc19563/skin_lesion_CNN/data/skin_lesion_images/test'

BATCH_SIZE = 10
IMG_SIZE = (224, 224)


train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='categorical'
                                             )

validation_dataset = image_dataset_from_directory(valid_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  label_mode='categorical')

test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            label_mode='categorical')

class_names = train_dataset.class_names
print(class_names)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
# plt.show()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

# Let's take a look at the base model architecture
# base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(224,224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# model.summary()

len(model.trainable_variables)

initial_epochs = 2

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

for fine_tune_at in range(1, 200):
    save_name = 'MN_initial=2_final=10_blr=0.0001_layers=' + str(fine_tune_at)

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  metrics=['accuracy'])

    #model.summary()

    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy for ' + str(fine_tune_at) + ' layers :', accuracy)

    


print("--- %s seconds ---" % (time.time() - start_time))


