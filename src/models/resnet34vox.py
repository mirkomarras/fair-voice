#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from .utils import fair_loss

import os


# from models.model import VladPooling
from models.model import Model, VladPooling
from helpers.audio import play_n_rec, get_tf_spectrum2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ResNet34Vox(Model):

    def __init__(self, name='resnet34vox', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000, loss_bal=None):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate, loss_bal=loss_bal)

    def identity_block_2d(self, input_tensor, kernel_size, filters, stage, block, weight_decay,trainable):

        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
        x = tf.keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='orthogonal', use_bias=False,
                                   trainable=trainable, kernel_regularizer= tf.keras.regularizers.l2(weight_decay), name=conv_name_1)(input_tensor)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
        x = tf.keras.layers.Conv2D(filters2, kernel_size,
                                   padding='same',
                                   kernel_initializer='orthogonal',
                                   use_bias=False,
                                   trainable=trainable,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name=conv_name_2)(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
        x = tf.keras.layers.Conv2D(filters3, (1, 1),
                                   kernel_initializer='orthogonal',
                                   use_bias=False,
                                   trainable=trainable,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name=conv_name_3)(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_3)(x)

        x = tf.keras.layers.Add()([x, input_tensor])
        x = tf.keras.layers.ReLU()(x)
        return x

    def conv_block_2d(self, input_tensor, kernel_size, filters, stage, block, strides, weight_decay,trainable):

        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
        x = tf.keras.layers.Conv2D(filters1, (1, 1),
                                   strides=strides,
                                   kernel_initializer='orthogonal',
                                   use_bias=False,
                                   trainable=trainable,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name=conv_name_1)(input_tensor)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',
                                   kernel_initializer='orthogonal',
                                   use_bias=False,
                                   trainable=trainable,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name=conv_name_2)(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
        x = tf.keras.layers.Conv2D(filters3, (1, 1),
                                   kernel_initializer='orthogonal',
                                   use_bias=False,
                                   trainable=trainable,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name=conv_name_3)(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_3)(x)

        conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
        bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
        shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                                          kernel_initializer='orthogonal',
                                          use_bias=False,
                                          trainable=trainable,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name=conv_name_4)(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(
            axis=bn_axis, trainable=trainable, name=bn_name_4)(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    def build(self, classes=None, loss='softmax', aggregation='gvlad', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4, augment=0):
        super().build(classes, loss, aggregation, vlad_clusters,
                      ghost_clusters, weight_decay, augment)
        print('>', 'building', self.name, 'model on', classes, 'classes')
        bn_axis = 3

        spec = tf.keras.Input(shape=(257, 250,1))

        #spec = tf.keras.layers.Lambda(
        #    lambda x: get_tf_spectrum2(x), name='acoustic_layer')(input)

        x1 = tf.keras.layers.Conv2D(64, (7, 7),
                                    kernel_initializer='orthogonal',
                                    use_bias=False, trainable=True,
                                    kernel_regularizer=tf.keras.regularizers.l2(
                                        weight_decay),
                                    padding='same',
                                    name='conv1_1/3x3_s1')(spec)
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
        x1 = tf.keras.layers.ReLU()(x1)
        x1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x1)

        # Conv 2_x
        x2 = self.conv_block_2d(
            x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1),weight_decay=weight_decay, trainable=True)
        x2 = self.identity_block_2d(
            x2, 3, [48, 48, 96], stage=2, block='b',weight_decay=weight_decay, trainable=True)

        # Conv 3_x
        x3 = self.conv_block_2d(
            x2, 3, [96, 96, 128], stage=3, block='a',strides=(2,2),weight_decay=weight_decay, trainable=True)
        x3 = self.identity_block_2d(
            x3, 3, [96, 96, 128], stage=3, block='b',weight_decay=weight_decay, trainable=True)
        x3 = self.identity_block_2d(
            x3, 3, [96, 96, 128], stage=3, block='c',weight_decay=weight_decay, trainable=True)

        # Conv 4_x
        x4 = self.conv_block_2d(
            x3, 3, [128, 128, 256], stage=4, block='a',strides=(2,2),weight_decay=weight_decay, trainable=True)
        x4 = self.identity_block_2d(
            x4, 3, [128, 128, 256], stage=4, block='b',weight_decay=weight_decay, trainable=True)
        x4 = self.identity_block_2d(
            x4, 3, [128, 128, 256], stage=4, block='c',weight_decay=weight_decay, trainable=True)

        # Conv 5_x
        x5 = self.conv_block_2d(
            x4, 3, [256, 256, 512], stage=5, block='a',strides=(2,2),weight_decay=weight_decay, trainable=True)
        x5 = self.identity_block_2d(
            x5, 3, [256, 256, 512], stage=5, block='b',weight_decay=weight_decay, trainable=True)
        x5 = self.identity_block_2d(
            x5, 3, [256, 256, 512], stage=5, block='c',weight_decay=weight_decay, trainable=True)
        x = tf.keras.layers.MaxPooling2D(
            (3, 1), strides=(2, 1), name='mpool2')(x5)

        # Fc layers
        xfc = tf.keras.layers.Conv2D(self.emb_size, (7, 1), strides=(1, 1), kernel_initializer='orthogonal',
                                     use_bias=False,trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='x_fc')(x)


        if aggregation == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=(
                1, 8), strides=(1, 1), name='apool{}'.format(6))(x)
            x = tf.math.reduce_mean(x, axis=[1, 2], name='rmean{}'.format(6))
            x = tf.keras.layers.Lambda(
                lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
        elif aggregation == 'vlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(
                weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters,
                            mode='vlad', name='vlad_pool')([xfc, xkcenter])
        elif aggregation == 'gvlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters+ghost_clusters, (7, 1),
                                             strides=(1, 1),
                                             kernel_initializer='orthogonal',
                                             use_bias=True, trainable=True,
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([xfc, xkcenter])

        else:
            raise NotImplementedError()

        x = tf.keras.layers.Dense(self.emb_size, activation='relu',
                                   kernel_initializer='orthogonal',
                                   use_bias=True, trainable=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   name='embedding')(x)

        if loss == 'softmax':
                y1 = tf.keras.layers.Dense(classes, activation='softmax',
                                       kernel_initializer='orthogonal',
                                       use_bias=False, trainable=True,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       name='fair')(x)
                y2 = tf.keras.layers.Dense(classes, activation='softmax',
                                          kernel_initializer='orthogonal',
                                          use_bias=False, trainable=True,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='categorical_cross')(x)

        elif loss == 'amsoftmax':
                x_l2 = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
                y = tf.keras.layers.Dense(classes,
                                       kernel_initializer='orthogonal',
                                       use_bias=False, trainable=True,
                                       kernel_constraint=tf.keras.constraints.unit_norm(),
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       name='prediction')(x_l2)

        self.model = tf.keras.models.Model(spec, [y1, y2], name='resnet34vox_{}_{}'.format(loss, aggregation))
        self.model.compile(optimizer='adam', loss=[fair_loss(classes), 'categorical_crossentropy'], metrics=['accuracy'], loss_weight=self.loss_bal)
