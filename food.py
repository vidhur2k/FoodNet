import os
from os import listdir
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import h5py
import numpy as np
from scipy.misc import imresize, imread

# Two dictionaries to save the classes and their respective indices.
index_to_classes = {}
classes_to_index = {}

# Adding the k-v pairs to the dicts using the custom text file.
with open('assets/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    classes_to_index = dict(zip(classes, range(len(classes))))
    index_to_classes = dict(zip(range(len(classes)), classes))


TRAINING_DIR = 'assets/training/'
VALIDATION_DIR = 'assets/validation/'
EVALUATION_DIR = 'assets/evaluation/'

def load_images(root, min_side=32):
    print('Loading ' + root[7:-1] + ' data...')
    images = []
    classes = []

    imgs = sorted(os.listdir(root))

    for img in listdir(root):
        # print('Loading Image: ' + str(counter))
        im = imresize(imread(root + img), (min_side, min_side))
        arr = np.array(im)
        images.append(arr)

        if img[0:2] == '10':
            classes.append(10)
        else:
            classes.append(int(img[0:1]))

    return np.array(images), np.array(classes)


# Loading the training, validation, and evaluation data.
X_tr, Y_tr = load_images(TRAINING_DIR)
X_val, Y_val = load_images(VALIDATION_DIR)
X_test, Y_test = load_images(EVALUATION_DIR)

# Set of print statements to check the input pipeline's proper functioning.
# print('X_train shape' + str(X_tr.shape))
# print('Y_train shape' + str(Y_tr.shape))
# print('X_val shape' + str(X_val.shape))
# print('Y_val shape' + str(Y_val.shape))
# print('X_test shape' + str(X_test.shape))
# print('Y_test shape' + str(Y_test.shape))

# Normalizing the inputs from 0-255 to 0.0-1.0.
X_tr = X_tr.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_tr = X_tr / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

print(np.unique(Y_tr))

Y_tr = np_utils.to_categorical(Y_tr)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

#
# print(Y_tr.shape)
#
# # Hyperparameters
n_classes = Y_test.shape[1]
print(n_classes)

# Initializing the sequential model.
model = Sequential()

# Adding the first convolution layer.
model.add(
    Conv2D(32,   # Number of kernels.
           (5, 5),  # Kernel size.
           input_shape=(32, 32, 3),
           padding='same',
           activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    Conv2D(64,
           (5, 5),
           input_shape=(16, 16, 32),
           padding='same',
           activation='relu')
)

model.add(
    Flatten()
)

model.add(
    Dense(
        n_classes,
        activation='softmax'
    )
)

epochs = 200
batch_size = 100
l_rate = 0.001

decay = l_rate / epochs

sgd = SGD(
    lr=l_rate,
    momentum=0.9,
    decay=decay,
    nesterov=False
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

print(model.summary())

model.fit(
    X_tr,
    Y_tr,
    validation_data=(X_val, Y_val),
    epochs=epochs,
    batch_size=batch_size
)

scores = model.evaluate(
    X_test,
    Y_test,
    verbose=1
)

print('\nAccuracy: %.2f%%' % (scores[1] * 100))

# Visualization module provided by Keras.
from keras.utils import plot_model


# # Plotting the final model
# plot_model(
#     model,
#     to_file='model1.png',
# )

model_json = model.to_json()

with open('model1.json', 'w') as json_file:
    json_file.write(model_json)

model.save('my_model1.h5')
model.save_weights('model1_weights.h5')
print('Saved model to disk.')

from keras.models import model_from_json

json_file = open('model1.json', 'r')

loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model1_weights.h5')
print('Loaded model from disk')

loaded_model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

score = loaded_model.evaluate(X_test, Y_test, verbose=1)


print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))