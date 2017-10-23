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

with open('classes.txt') as f:
    classes = f.read().splitlines()

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

        print('Reading: ' + i + ' from ' + root)
        img = Image.open(root + i)
        print(str(img.size[1]) + ' ' + str(img.size[0]))
        all_images.append(imresize(img.getdata(), (512, 384)))
        current = all_images[len(all_images) - 1]
        # print('Rows: ' + str(len(current)))
        # print('Cols: ' + str(len(current[0])))
        all_images.append(np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3))

    print('Images loaded: ' + str(len(all_images)))
    return np.array(all_images), np.array(all_classes)

X_train, Y_train = load_images_and_classes('./training/')

# img = imread('./training/0_0.jpg')

print(load_images_and_classes('./training/'))
# print(plt.imread('./training/0_0.jpg'))