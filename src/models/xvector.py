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

def normalize_with_moments(x):
    tf_mean, tf_var = tf.nn.moments(x, 1)
    x = tf.concat([tf_mean, tf.sqrt(tf_var + 0.00001)], 1)
    return x

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

        def g_loss():

            def loss(y_true, y_pred):

                sens_attr = tf.map_fn(lambda g: g == 1, y_true[:, :classes][:, 0], dtype=tf.bool)
                y_true = y_true[:, classes:]

                y_t_male = tf.gather(y_true, tf.reshape(tf.where(sens_attr), [-1]))
                y_p_male = tf.gather(y_pred, tf.reshape(tf.where(sens_attr), [-1]))
                not_sens_attr = tf.math.logical_not(sens_attr)
                y_t_female = tf.gather(y_true, tf.reshape(tf.where(not_sens_attr), [-1]))
                y_p_female = tf.gather(y_pred, tf.reshape(tf.where(not_sens_attr), [-1]))

                cc_male = tf.keras.losses.categorical_crossentropy(y_t_male, y_p_male)
                cc_female = tf.keras.losses.categorical_crossentropy(y_t_female, y_p_female)

                cc_male = tf.keras.backend.mean(cc_male)
                cc_female = tf.keras.backend.mean(cc_female)
                return tf.keras.backend.square(cc_male-cc_female)

            return loss

        input = tf.keras.Input(shape=(None, 24,))

        #g_layer = tf.keras.Input(shape=(), dtype=tf.bool)

        #x = tf.keras.layers.Lambda(lambda x: get_tf_filterbanks(x), name='acoustic_layer')(input)
        x = input
        # Layer parameters
        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]

        # Frame information layer
        for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
            x = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=layer_size, padding='SAME')(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0),
                                                   beta_initializer=tf.constant_initializer(0.0))(x)
            if i != len(kernel_sizes) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)

        # Statistic pooling
        x = tf.keras.layers.Lambda(lambda x: normalize_with_moments(x))(x)

        # Embedding layers
        out_dim = 512

        x = tf.keras.layers.Dense(out_dim)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0),
                                               beta_initializer=tf.constant_initializer(0.0))(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Dense(out_dim, name='fc7')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0),
                                               beta_initializer=tf.constant_initializer(0.0))(x)

        if loss == 'softmax':
            y1 = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                      bias_regularizer=tf.keras.regularizers.l2(weight_decay), name="fair")(x)
            y2 = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       bias_regularizer=tf.keras.regularizers.l2(weight_decay), name="categorical_cross")(x)
        elif loss == 'amsoftmax':
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
            y = tf.keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False,
                                      kernel_constraint=tf.keras.constraints.unit_norm(),
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                      bias_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        else:
            raise NotImplementedError()

        self.model = tf.keras.models.Model([input], [y1, y2], name='xvector_{}_{}'.format(loss, aggregation))
        self.model.compile(optimizer='adam', loss=[g_loss(), 'categorical_crossentropy'], metrics=['acc'], loss_weight=[0.5, 0.5])
        print('>', 'built', self.name, 'model on', classes, 'classes')


