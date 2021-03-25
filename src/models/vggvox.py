#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from models.model import VladPooling
from models.model import Model
from helpers.audio import play_n_rec, get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VggVox(Model):

    """
       Class to represent Speaker Verification (SV) model based on the VGG16 architecture - Embedding vectors of size 1024 are returned
       Nagrani, A., Chung, J. S., & Zisserman, A. (2017).
       VoxCeleb: A Large-Scale Speaker Identification Dataset.
       Proc. Interspeech 2017, 2616-2620.
    """

    def __init__(self, name='vggvox', id=-1, noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)

    def __conv_bn_pool(self, inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, pool='', pool_size=(3, 3), pool_strides=None, conv_layer_prefix='conv', weight_decay=1e-4):
        x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
        x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
        x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
        #"""
        if pool == 'max':
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, name='mpool{}'.format(layer_idx))(x)
        #elif pool == 'avg':
        #    x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, name='apool{}'.format(layer_idx))(x)
        #else:

        return x

    def build(self, classes=None, loss='softmax', aggregation='gvlad', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4, augment=0):
        super().build(classes, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, augment)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        spec = tf.keras.Input(shape=(512, 300,1))


        #x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x), name='acoustic_layer')(spec)

        x = self.__conv_bn_pool(spec, layer_idx=1, conv_filters=96, conv_kernel_size=(7, 7), conv_strides=(2, 2), conv_pad=(1, 1), pool='max', pool_size=(3, 3), pool_strides=(2, 2), weight_decay=weight_decay)
        x = self.__conv_bn_pool(x, layer_idx=2, conv_filters=256, conv_kernel_size=(5, 5), conv_strides=(2, 2), conv_pad=(1, 1), pool='max', pool_size=(3, 3), pool_strides=(2, 2), weight_decay=weight_decay)
        x = self.__conv_bn_pool(x, layer_idx=3, conv_filters=384, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1), weight_decay=weight_decay)
        x = self.__conv_bn_pool(x, layer_idx=4, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1), weight_decay=weight_decay)
        x = self.__conv_bn_pool(x, layer_idx=5, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1), pool='max', pool_size=(5, 3), pool_strides=(3, 2), weight_decay=weight_decay)

        #x = tf.keras.layers.ZeroPadding2D(padding=(0, 0), name='pad{}'.format(6))(x)
        xfc = tf.keras.layers.Conv2D(filters=self.emb_size, kernel_size=(9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='{}{}'.format('fc', 6))(x)
        #xfc=
        #xfc = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(6))(xfc)
        #xfc = tf.keras.layers.Activation('relu', name='relu{}'.format(6))(xfc)

        if aggregation == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1, 1), name='apool{}'.format(6))(xfc)
            x = tf.math.reduce_mean(x, axis=[1, 2], name='rmean{}'.format(6))
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
        elif aggregation == 'vlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters+ghost_clusters, (9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='vlad', name='vlad_pool')([xfc, xkcenter])
        elif aggregation == 'gvlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([xfc, xkcenter])
        else:
            raise NotImplementedError()

        e = tf.keras.layers.Dense(self.emb_size, activation='relu', kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='embedding')(x)

        if loss == 'softmax':
            y = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(e)
        elif loss == 'amsoftmax':
            x = keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
            y = keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False, kernel_constraint=tf.keras.constraints.unit_norm(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(x)
        else:
            raise NotImplementedError()

        self.model = tf.keras.models.Model(spec, y, name='vggvox_{}_{}'.format(loss, aggregation))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
