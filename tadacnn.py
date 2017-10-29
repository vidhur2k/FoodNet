food101_dataset_folder_path = 'food-101-batches-py'

food101_location = ''

# TODO: Complete data preprocessing in accordance with the Google Cloud principles.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

n_classes = 11

tf.logging.set_verbosity(tf.logging.INFO)

# TODO: Write the application.

def tada_cnn_model(x_tensor, y_tensor, mode):

    """

    :param x_tensor:
    :param y_tensor:
    :param mode:
    :return:
    """

    """
    Input layer
    NOTE: -1 specifies a dynamic batch size.
    The input consists of images of dimensions 512x384x3 (3 for RGB color channel)
    """
    input_layer = tf.reshape(x_tensor['x'], [-1, 512, 384, 3])

    """
    Convolution Layer 1
    This is the first convolution layer that is applied to the input tensor.
    There are 32 filters with a kernel size of [5, 5].
    We make sure we preserve the dimensions of the input tensor when returning a 
    tensor by specifying the padding to remain the same.
        
    """
    convolution1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    """
    Max Pooling Layer 1
    This is the first max pooling layer that is applied to the output of the first convolution
    layer. The kernel size is [2, 2], and we provide a stride of 2 to prevent any overlap between a certain
    pixel value of the image.
    """
    pooling1 = tf.layers.max_pooling2d(
        inputs=convolution1,
        pool_size=[2, 2],
        strides=2
    )

    """
    Second Convolution Layer
    
    """
    convolution2 = tf.layers.conv2d(
        inputs=pooling1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    """
    Second Pooling Layer
    
    """
    pooling2 = tf.layers.max_pooling2d(
        inputs=convolution2,
        pool_size=[2, 2],
        strides=2
    )

    """
    Flattening Layer
    This layer takes in input from the second pooling layer and flattens it in order
    to provide a 1D tensor to the fully connected layer.
    """
    flattening = tf.reshape(
        pooling2,
        [-1, 128 * 96 * 64])

    """
    Dense (Fully Connected) Layer
    
    """
    dense = tf.layers.dense(
        inputs=flattening,
        units=1024,
        activation=tf.nn.relu
    )

    """
    Dropout Layer
    
    """
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training= mode == tf.estimator.ModeKeys.TRAIN
    )

    """
    Logits Layer
    
    Uses the output from the dropout layer as the input
    """
    logits = tf.layers.dense(
        inputs=dropout,
        units=n_classes
    )

    """
    Saves the various classes and the probability value of each class
    into a dictionary.
    """
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if(mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(y_tensor, tf.int32), depth=n_classes)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )

    if(mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    evaluation_metrics_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=y_tensor, predictions=predictions['classes']
        )}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=evaluation_metrics_ops
    )

# TODO: Initialize hyperparameters.