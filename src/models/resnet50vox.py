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
from .utils import fair_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ResNet50Vox(Model):
    """
       Class to represent Speaker Verification (SV) model based on the ResNet50 architecture - Embedding vectors of size 512 are returned
       Chung, J. S., Nagrani, A., & Zisserman, A. (2018).
       VoxCeleb2: Deep Speaker Recognition.
       Proc. Interspeech 2018, 1086-1090.
    """

    def __init__(self, name='resnet50vox', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000, loss_bal=None):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate, loss_bal=loss_bal)

    def build(self, classes=None, loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4, augment=0):
        super().build(classes, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, augment)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        #input = tf.keras.Input(shape=(None,1,))
        spec = tf.keras.Input(shape=(257, 250,1))
        #x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x), name='acoustic_layer')(input)

        resnet_50 = tf.keras.applications.ResNet50(input_tensor=spec, include_top=False, weights=None)

        x = tf.keras.layers.ZeroPadding2D(padding=(0, 0), name='pad{}'.format(6))(resnet_50.output)
        xfc = tf.keras.layers.Conv2D(filters=self.emb_size, kernel_size=(9, 1), strides=(1, 1), padding='valid', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='{}{}'.format('fc', 6))(x)
        xfc = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(6))(xfc)
        xfc = tf.keras.layers.Activation('relu', name='relu{}'.format(6))(xfc)

        if aggregation == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1, 1), name='apool{}'.format(6))(xfc)
            x = tf.math.reduce_mean(x, axis=[1, 2], name='rmean{}'.format(6))
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
        elif aggregation == 'vlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='vlad', name='vlad_pool')([xfc, xkcenter])
        elif aggregation == 'gvlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([xfc, xkcenter])
        else:
            raise NotImplementedError()

        e = tf.keras.layers.Dense(self.emb_size, activation='relu', kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='embedding')(x)

        if loss == 'softmax':
            y1 = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fair')(e)
            y2 = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='categorical_cross')(e)
        elif loss == 'amsoftmax':
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
            y = tf.keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False, kernel_constraint=tf.keras.constraints.unit_norm(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(x)
        else:
            raise NotImplementedError()

        self.model = tf.keras.models.Model(spec, [y1, y2], name='resnet50vox_{}_{}'.format(loss, aggregation))
        self.model.compile(optimizer='adam', loss=[fair_loss(classes), 'categorical_crossentropy'], metrics=['accuracy'], loss_weight=self.loss_bal)
        print('>', 'built', self.name, 'model on', classes, 'classes')
