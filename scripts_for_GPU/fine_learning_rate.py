# import itertools

# import sklearn
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import utils
# from utils import plotConfusionMatrix

assert tf.__version__.startswith('2')

# import os
import numpy as np

# import matplotlib.pyplot as plt

# Fixed variables
SEED = 53
RESCALE = 1. / 255  # Force all data to range [0,1]
IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
LOSS = 'categorical_crossentropy'
BATCH_SIZE = 20  # see here: https://stackoverflow.com/questions/49922252/choosing-number-of-steps-per-epoch
DATASET_SIZE = 10000  # Total number of images = 10000
NUM_BATCHES = DATASET_SIZE / BATCH_SIZE
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.14
TEST_FRACTION = 0.16
TRAIN_SIZE = TRAIN_FRACTION * DATASET_SIZE
VAL_SIZE = VAL_FRACTION * DATASET_SIZE
TEST_SIZE = TEST_FRACTION * DATASET_SIZE
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUMBER_OF_CLASSES = len(CLASSES)

# Preprossing parameters
SAMPLEWISE_STD_NORMALISATION = True
SAMPLEWISE_CENTRE = True

# Augmentation parameters
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# Base network training parameters
INITIAL_LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 2  # number of epochs to train the model with no fine-tuning
CLASSIFIER_DROPOUT = 0.2
CLASSIFIER_FILTERS = 32
CLASSIFIER_KERNEL_SIZE = 3

# Fine-tuned network training parameters
EPOCHS = 10  # number of epochs when fine-tuning
# LEARNING_RATE = 0.00001
LEARNING_RATE_RANGE = np.linspace(1e-6, 1e-5, 10)
print(LEARNING_RATE_RANGE)
FINE_TUNE_AT = 60  # number of layers to fine-tune from

ADD_TO_DIR = ''
ON_LOCAL = False  # Change this when copying script to bluepebble
if ON_LOCAL:
    ADD_TO_DIR = '../'

DATA_DIR = ADD_TO_DIR + 'data/images_sorted/'
MODELS_DIR = ADD_TO_DIR + 'models/'
GRAPHS_DIR = ADD_TO_DIR + 'graphs/'
GENERATED_DATA_DIR = ADD_TO_DIR + 'generated_data/'
INVESTIGATION_NAME = 'fine_learning_rate/'

# Set seeds for numpy and TensorFlow for deterministic results
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load the dataset
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='categorical',
    color_mode='rgb', batch_size=BATCH_SIZE, image_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=SEED,
    subset=None, interpolation='bilinear', follow_links=False
)

# Create training, validation, and testing datasets
train_dataset = full_dataset.take(int(NUM_BATCHES * TRAIN_FRACTION))
val_dataset = full_dataset.skip(int(NUM_BATCHES * TRAIN_FRACTION))
test_dataset = val_dataset.take(int(NUM_BATCHES * TEST_FRACTION))
val_dataset = val_dataset.skip(int(NUM_BATCHES * TEST_FRACTION))


for LEARNING_RATE in LEARNING_RATE_RANGE:
    MODEL_NAME = 'LR=' + str(LEARNING_RATE)  # Name to save model
    # Use Mobilenet v2 as the base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # Don't allow fine-tuning of the base model for now
    base_model.trainable = False

    # Add classifier to base model
    # TODO investigate changing some of these parameters
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(filters=CLASSIFIER_FILTERS, kernel_size=CLASSIFIER_KERNEL_SIZE, activation='relu'),
        tf.keras.layers.Dropout(CLASSIFIER_DROPOUT),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])

    # Create the model using these settings
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
                  loss=LOSS,
                  metrics=['accuracy'])

    # Train the model (with no fine-tuning for now)
    history = model.fit(train_dataset,
                        steps_per_epoch=int(TRAIN_SIZE/BATCH_SIZE),
                        epochs=INITIAL_EPOCHS,
                        validation_data=val_dataset,
                        validation_steps=int(VAL_SIZE/BATCH_SIZE))

    # Now allow the base model to be fine-tuned
    base_model.trainable = True

    # # Print number of layers in base model
    # print("Number of layers in the base model: ", len(base_model.layers))

    # Don't train any layers before layer FINE_TUNE_AT
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    # Compile model to be fine-tuned
    model.compile(loss=LOSS,
                  optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy'])

    # Train the fine-tuned model
    history_fine = model.fit(train_dataset,
                             steps_per_epoch=int(TRAIN_SIZE/BATCH_SIZE),
                             epochs=EPOCHS,
                             validation_data=val_dataset,
                             validation_steps=int(VAL_SIZE/BATCH_SIZE))

    # Get the test loss and test accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print('Test accuracy for LR = ' + str(LEARNING_RATE), test_accuracy)
    np.save(GENERATED_DATA_DIR + INVESTIGATION_NAME + 'testAcc_' + MODEL_NAME, test_accuracy)
    np.save(GENERATED_DATA_DIR + INVESTIGATION_NAME + 'testLoss_' + MODEL_NAME, test_loss)

    # Save model as type SavedModel (.pb)
    saved_model_dir = MODELS_DIR + INVESTIGATION_NAME + MODEL_NAME + str(LEARNING_RATE)
    tf.saved_model.save(model, saved_model_dir)



    # # Convert model to tflite for use in app
    # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # tflite_model = converter.convert()
    # with open(saved_model_dir + '.tflite', 'wb') as f:
    #     f.write(tflite_model)
