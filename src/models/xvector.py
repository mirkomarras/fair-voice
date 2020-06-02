#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from models.model import Model
from helpers.audio import play_n_rec, get_tf_filterbanks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class XVector(Model):

    """
       Class to represent Speaker Verification (SV) model based on the XVector architecture - Embedding vectors of size 512 are returned
       Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018, April).
       X-vectors: Robust dnn embeddings for speaker recognition.
       In: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5329-5333. IEEE.
    """

    def __init__(self, name='xvector', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)
        self.n_filters = 24

    def build(self, classes=None, loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4, augment=0):
        super().build(classes, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, augment)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input = tf.keras.Input(shape=(None,1,))

        x = tf.keras.layers.Lambda(lambda x: get_tf_filterbanks(x), name='acoustic_layer')(input)

        # Layer parameters
        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]

        # Frame information layer
        prev_dim = self.n_filters
        for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
            x = tf.nn.conv1d(x, tf.random.truncated_normal([kernel_size, prev_dim, layer_size], stddev=0.1), stride=1, padding='SAME')
            x = tf.nn.bias_add(x, tf.constant(0.1, shape=[layer_size]))
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)
            prev_dim = layer_size
            if i != len(kernel_sizes) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)

        # Statistic pooling
        tf_mean, tf_var = tf.nn.moments(x, 1)
        x = tf.concat([tf_mean, tf.sqrt(tf_var + 0.00001)], 1)
        x = tf.math.l2_normalize(x)

        # Embedding layers
        embedding_layers = []
        for i, out_dim in enumerate(embedding_sizes):
            x = tf.keras.layers.Dense(out_dim)(x)
            embedding_layers.append(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)
            if i != len(embedding_sizes) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)

        if loss == 'softmax':
            y = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        elif loss == 'amsoftmax':
            x = keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
            y = keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False, kernel_constraint=tf.keras.constraints.unit_norm(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        else:
            raise NotImplementedError()

        self.model = tf.keras.models.Model(input, y, name='xvector_{}_{}'.format(loss, aggregation))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        print('>', 'built', self.name, 'model on', classes, 'classes')
