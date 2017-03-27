from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def martin_net(images):

    input_layer = tf.reshape(images, [-1, 299, 299, 3])

    ## første konvulerende lag.
    conv_layer_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    ##maxpool lag 1
    max_pool_1 = tf.layers.max_pooling2d(
        inputs=conv_layer_1,
        pool_size=[2, 2],
        strides=2,
        padding="same")

    ##andre konvulerende lag
    conv_layer_2 = tf.layers.conv2d(
        inputs=max_pool_1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    ##maxpool lag 2
    max_pool_2 = tf.layers.max_pooling2d(
        inputs=conv_layer_2,
        pool_size=[2, 2],
        padding="same",
        strides=2)

    ##tredje konvulerende lag
    conv_layer_3 = tf.layers.conv2d(
        inputs=max_pool_2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    ##maxpool lag 4
    max_pool_3 = tf.layers.max_pooling2d(
        inputs=conv_layer_3,
        pool_size=[2, 2],
        padding="same",
        strides=2)

    ##tredje konvulerende lag
    conv_layer_4 = tf.layers.conv2d(
        inputs=max_pool_3,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    ##maxpool lag 3
    max_pool_4 = tf.layers.max_pooling2d(
        inputs=conv_layer_4,
        pool_size=[2, 2],
        padding="same",
        strides=2)

    ## reshape av maxpool4
    max_pool_2_flat = tf.reshape(max_pool_4, [-1, 19*19*64])

    ## fullt sammenkoblet lag.
    dense_layer = tf.layers.dense(
        inputs=max_pool_2_flat,
        units=1024,
        activation=tf.nn.relu)

    ## dropoutlag
    dropout_layer = tf.layers.dropout(
        inputs=dense_layer,
        rate=0.4)

    #logits lag
    logits_layer = tf.layers.dense(
        inputs=dropout_layer,
        units=10)

    return logits_layer

