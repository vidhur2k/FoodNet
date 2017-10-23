"""
Description: A Convolutional Neural Network for food item recognition.
The dataset used for training, validation, and evaluation is called "Food-11",
and it comes from the MMSPG (Multimedia Signal Processing Group)

Author: Vidhur Kumar: kumar289@purdue.edu
Date Started: 10/20/17
Date Published: TBD

"""

import os

# List to save all classes in.
classes = []

with open('./meta/classes.txt') as f:
    classes = f.read().splitlines()

print(classes)

import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread

from PIL import Image

index_to_class = {}

for i in range(len(classes)):
    index_to_class[i] = classes[i]


def load_images_and_classes(root):
    all_images = []
    all_classes = []
    resize_count = 0

    for i in os.listdir(root):
        if i[0:2] == '10':
            all_classes.append(index_to_class[10])
            continue
        else:
            all_classes.append(index_to_class[int(i[0])])

        # print('Reading: ' + i + ' from ' + root)
        img = Image.open(root + i)
        # print(str(img.size[1]) + ' ' + str(img.size[0]))
        all_images.append(imresize(img.getdata(), (512, 384)))
        # current = all_images[len(all_images) - 1]
        # print('Rows: ' + str(len(current)))
        # print('Cols: ' + str(len(current[0])))
        all_images.append(np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3))

    print('Images loaded: ' + str(len(all_images)))
    return np.array(all_images), np.array(all_classes)

# Load the training data.
X_train, Y_train = load_images_and_classes('./training/')

# Load the evaluation data.
X_test, Y_test = load_images_and_classes('./evaluation/')


# for i in range(len(X_train)):
#     print('Pixel Vals: ' + X_train)
#     print('Class: ' + Y_train)

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math


'''

'''

# Initialize hyperparameters.
batch_size = 32
num_classes = len(classes)


K.clear_session()

# Model
model = Sequential()

# First convolution layer.
model.add(
    Convolution2D(32,   # 32 filters.

                  # Kernel size of 3x3.
                  (3, 3),

                  # Define the input shape (3D because its an RGB image.
                  input_shape=(512, 384, 3),

                  #
                  padding='same',

                  activation='relu',

                  kernel_constraint=max(3),

                  )
)

# Dropout regularization at a rate of 0.2.
model.add(
    Dropout(0.2)
)

# Second convolution layer.
model.add(
    Convolution2D(32,

                  (3, 3),

                  activation='relu',

                  padding='same',

                  kernel_constraint=max(3))
)

# First pooling layer.
model.add(
    MaxPooling2D(pool_size=(2, 2))  # Kernel size of 2x2.
)

# Flattening layer.
model.add(

    # Changes the 3d input tensor to a 1d input.
    Flatten()
)


# Dense (Fully connected) layer.
model.add(
    Dense(512,

          activation='relu',

          kernel_constraint=max(3))
)

# Dropout regularization at a rate of 0.5.
model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        num_classes,

        activation='softmax'
    )
)