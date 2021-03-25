#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.distance import euclidean, cosine
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import roc_curve, auc
import datetime
from scipy import spatial
import tensorflow as tf
import keras.backend as K
import numpy as np
import random
import pandas as pd
import time
import os
import sys
import librosa
import soundfile as sf

sys.setrecursionlimit(10000)

from helpers.audio import decode_audio_fix_size, play_n_rec, get_tf_spectrum2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SAVE_PATH = '/home/meddameloni/dl-fair-voice/exp/trained_model/'
RESULT_PATH = '/home/meddameloni/dl-fair-voice/exp/results/'
RESULT_PATH_TRAIN3 = os.path.join(RESULT_PATH, 'deep_res_EN-SP_train3/')
RESULT_PATH_TRAIN_NO_PRIOR = os.path.join(RESULT_PATH, 'trainSP-testEN/')
RESULT_PATH_VOXCELEB = os.path.join(RESULT_PATH, 'voxceleb_res/')
RESULT_PATH_RESNET50 = os.path.join(RESULT_PATH, 'resnet50vox_EN-SP/')
RESULT_SECOND_STAGE = os.path.join(RESULT_PATH, 'second_stage/')
RESULT_SECOND_STAGE_SN_RESNET = os.path.join(RESULT_PATH, 'second_stage/siamese_net/resnet34vox/first_stage/')
RESULT_SECOND_STAGE_SN_XVECTOR = os.path.join(RESULT_PATH, 'second_stage/siamese_net/xvector/first_stage/')


class StepDecay():
    def __init__(self, init_alpha=0.01, decay_factor=0.25, decay_step=10):
        self.init_alpha = init_alpha
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.decay_step)
        alpha = self.init_alpha * (self.decay_factor ** exp)
        print('Learning rate for next epoch', float(alpha))
        return float(alpha)


class VladPooling(tf.keras.layers.Layer):
    """
    This layer follows the NetVlad, GhostVlad
    """
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers+self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers*input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims = True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)    # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)    # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster    # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)     # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs

    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['k_centers'] = self.k_centers
        config['g_centers'] = self.g_centers
        config['mode'] = self.mode
        return config


class Model(object):
    """
       Class to represent Speaker Verification (SV) models with model saving / loading and playback & recording capabilitie
    """
    def __init__(self, name='', id=-1, noises=None, cache=None, n_seconds=3, sample_rate=16000, emb_size=512, loss_bal=[0.5,0.5]):
        """
        Method to initialize a speaker verification model that will be saved in 'data/pt_models/{name}'
        :param name:        String id for this model
        :param id:          Version id for this model - default: auto-increment value along the folder 'data/pt_models/{name}'
        :param noises:      Dictionary of paths to noise audio samples, e.g., noises['room'] = ['xyz.wav', ...]
        :param cache:       Dictionary of noise audio samples, e.g., cache['xyz.wav'] = [0.1, .54, ...]
        :param n_seconds:   Maximum number of seconds of an audio sample to be processed
        :param sample_rate: Sample rate of an audio sample to be processed
        """
        self.noises = noises
        self.cache = cache

        self.sample_rate = sample_rate
        self.n_seconds = n_seconds
        self.emb_size = emb_size

        self.loss_bal = loss_bal if loss_bal is not None else [0.0, 1.0]
        self.name = name
        self.dir = os.path.join('.', 'data', 'pt_models', self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.id = len(os.listdir(self.dir)) if id < 0 else id
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' +
                                     str('{:03d}'.format(self.id))))

    def get_model(self):
        return self.inference_model

    def build(self, classes=None, loss='softmax', aggregation='gvlad', vlad_clusters=2, ghost_clusters=8, weight_decay=1e-4, augment=0):
        """
        Method to build a speaker verification model that takes audio samples of shape (None, 1) and impulse flags (None, 3)
        :param classes:         Number of classes that this model should manage during training
        :param loss:            Type of loss
        :param aggregation:     Type of aggregation function
        :param vlad_clusters:   Number of vlad clusters in vlad and gvlad
        :param ghost_clusters:  Number of ghost clusters in vlad and gvlad
        :param weight_decay:    Decay of weights in convolutional layers
        :param augment:         Augmentation flag
        :return:                None
        """
        self.model = None
        self.inference = None
        self.classes = classes

    def save(self):
        """
        Method to save the weights of this model in 'data/pt_models/{name}/v{id}/model.tf'
        :return:            None
        """
        print('>', 'saving', self.name, 'model')
        if not os.path.exists(os.path.join(self.dir)):
            os.makedirs(os.path.join(self.dir))
        self.model.save(os.path.join(
            self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.tf'))
        print('>', 'saved', self.name, 'model in', os.path.join(
            self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.tf'))

    def load(self):
        """
        Method to load weights for this model from 'data/pt_models/{name}/v{id}/model.tf'
        :return:            None
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.tf')):
                self.model = tf.keras.models.load_model(os.path.join(
                    self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.tf'))
                print('>', 'loaded model from', os.path.join(
                    self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.tf'))
            else:
                print('>', 'no pre-trained model for', self.name, 'model from',
                      os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.tf'))
        else:
            print('>', 'no directory for', self.name, 'model at',
                  os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def embed(self, signal):
        """
        Method to compute the embedding vector extracted by this model from signal with no playback & recording
        :param signal:      The audio signal from which the embedding vector will be extracted - shape (None,1)
        :return:            None
        """
        return self.model.predict(signal)

    def train(self, train_data, test_data, steps_per_epoch=1, epochs=1, learning_rate=1e-1, patience=20, decay_factor=0.1, decay_step=10, optimizer='adam' ,info=""):
        """
        Method to train and validate this model
        :param train_data:      Training data pipeline - shape ({'input_1': (batch, None, 1), 'input_2': (batch, 3)}), (batch, classes)
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :param steps_per_epoch: Number of steps per epoch
        :param epochs:          Number of training epochs
        :param learning_rate:   Learning rate
        :param patience:        Number of epochs with non-improving EER willing to wait
        """

        """
        for i, y_true in train_data:
            layer = tf.map_fn(lambda x: x == 1, y_true[0][:][:, :620][:,0], dtype=tf.bool)
            # print(tf.where(layer))
            print(y_true[0].shape)
            print(y_true[0][:, 620:], y_true[0][:, 620:].shape)
            y_true = y_true[0][:][:, 620:]
            print(y_true, y_true.shape)
            #print("y_true", y_true.shape)
            #print(tf.where(layer))
            #print(tf.math.logical_not(layer))
            y_t_male = tf.gather(y_true, tf.reshape(tf.where(layer), [-1]))
            #print(y_true)
            print(y_t_male)
            not_layer = tf.math.logical_not(layer)
            y_t_female = tf.gather(y_true, tf.reshape(tf.where(not_layer), [-1]))
            print(y_t_female)
            exit()
        """

        self.model.summary()
        print('>', 'training', self.name, 'model')
        schedule = StepDecay(init_alpha=learning_rate,
                             decay_factor=decay_factor, decay_step=decay_step)
        learning_rate = LearningRateScheduler(schedule)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='categorical_cross_accuracy', baseline=0.95, patience=2, mode='auto')

        # current date to use for the folder name composition
        current_date = datetime.datetime.now().strftime('%d%m%Y_%H%M')
        # name of the CSV train file used
        # train 1  --> balanced by user
        # train 2  --> not balanced
        # train 3  --> balanced by user and samples
        train_file_name = info.split('/')[-1][:-4]
        # composition of the string containing loss function balancing details
        loss_balancing = ''
        for index in range(len(self.loss_bal)):
            if index != (len(self.loss_bal)-1):
                loss_balancing += str(self.loss_bal[index]).replace('.','') + '_'
            else:
                loss_balancing += str(self.loss_bal[index]).replace('.','')

        # Composition of the destination folder for the weight saved at every epoch
        folder_name = self.name + '_' + \
                      current_date + '_' + \
                      train_file_name + '#' + \
                      loss_balancing
        save_path = os.path.join(SAVE_PATH, folder_name)

        if not(os.path.exists(save_path)):
            os.makedirs(save_path)

        save_weight = ModelCheckpoint(os.path.join(save_path, 'weights-{epoch:02d}-{categorical_cross_accuracy:.3f}.h5'),
                                      monitor='categorical_cross_accuracy',
                                      mode='max', save_best_only=True)

        callbacks = [learning_rate, save_weight, early_stopping]
        # num_nonimproving_steps, last_eer = 0, 1.0

        for epoch in range(epochs):
            self.model.fit(train_data, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, epochs=epoch+1, callbacks=callbacks)
            """
            eer, _, _ = self.test(test_data)
            if eer < last_eer:
                print('>', 'eer improved from', round(
                    last_eer, 2), 'to', round(eer, 2))
                num_nonimproving_steps = 0
                 last_eer = eer
            else:
                print('>', 'eer NOT improve from', round(last_eer, 2))
                num_nonimproving_steps += 1

            if num_nonimproving_steps == patience:
                print('>', 'early stopping training after',
                      num_nonimproving_steps, 'non-improving steps')
                break
             """
        print('>', 'trained', self.name, 'model')

    def test(self, test_data):
        """
        Method to test this model against verification attempts
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :return:                (Model EER, EER threshold, FAR1% threshold)
        """

        print('>', 'testing', self.name, 'model')
        (x1, x2), y = test_data
        eer, thr_eer, thr_far1 = 0, 0, 0
        similarity_scores = np.zeros(len(x1))
        for pair_id, (f1, f2) in enumerate(zip(x1, x2)):
            similarity_scores[pair_id] = (
                1 - cosine(self.embed(np.array(f1)), self.embed(np.array(f2))) + 1) / 2
            if pair_id > 2:
                far, tpr, thresholds = roc_curve(
                    y[:pair_id+1], similarity_scores[:pair_id+1], pos_label=1)
                frr = 1 - tpr
                id_eer = np.argmin(np.abs(far - frr))
                id_far1 = np.argmin(np.abs(far - 0.01))
                eer = float(np.mean([far[id_eer], frr[id_eer]]))
                thr_eer = thresholds[id_eer]
                thr_far1 = thresholds[id_far1]
                print('\r> pair %5.0f / %5.0f - eer: %3.5f - thr@eer: %3.5f - thr@far1: %3.1f' % (pair_id+1, len(x1), eer, thr_eer, thr_far1), end='')
        print()
        print('>', 'tested', self.name, 'model')
        return eer, thr_eer, thr_far1

    def test_and_save_on_csv(self, test_file, audio_dir, aggregation, model_path):
        """
        Method to test this model against verification attempts
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :return:                (Model EER, EER threshold, FAR1% threshold)
        """
        normalize_switch = True
        # Loading file

        print("Loading model : ", os.path.exists(model_path))
        if aggregation == 'vlad' or aggregation == 'gvlad':
            model = tf.keras.models.load_model(model_path, custom_objects={'VladPooling': VladPooling})
        else:
            model = tf.keras.models.load_model(model_path,  custom_objects={'loss': lambda x, y: 0.0})

        emb_layer = None
        for layer in model.layers:
            if layer.name == "embedding" or layer.name == "fc7":
                emb_layer = layer
                break

        if emb_layer is None:
            raise ValueError("No layer called embedding")

        inference_model = tf.keras.models.Model(inputs=model.get_input_at(0), outputs=emb_layer.get_output_at(0))

        inference_model.summary()
        print('Start Testing')
        pairs = pd.read_csv(test_file)
        index_len = len(pairs.index)
        for index, row in pairs.iterrows():
            if self.name == 'xvector':
                audio_1 = decode_audio_fix_size(audio_dir+row['audio_1'], input_format='bank')
                audio_2 = decode_audio_fix_size(audio_dir+row['audio_2'], input_format='bank')
            else:
                audio_1 = decode_audio_fix_size(audio_dir + row['audio_1'], input_format='spec')
                audio_2 = decode_audio_fix_size(audio_dir + row['audio_2'], input_format='spec')

            emb1 = inference_model.predict(audio_1)
            emb2 = inference_model.predict(audio_2)

            if normalize_switch:
                emb1 = tf.keras.layers.Lambda(lambda emb1: K.l2_normalize(emb1, 1))(emb1)
                emb2 = tf.keras.layers.Lambda(lambda emb2: K.l2_normalize(emb2, 1))(emb2)

            similarity = 1 - cosine(emb1, emb2)
            if index % (index_len // 10) == 0:
                print('PAIR', index+1, index_len, row['label'], round(similarity, 2))
            pairs.loc[index, 'simlarity'] = float("{0:.2f}".format(similarity))

        """
            Retrieving important information for the experiment:
            - first_segment  -->  (train model name, date)
            - second_segment -->  (epoch, accuracy)
            - third_segment  -->  (loss balancing information)
        """

        first_segment, second_segment, third_segment = model_path.split('/')[-2].split('#')[0], \
                                                       model_path.split('/')[-1], \
                                                       model_path.split('/')[-2].split('#')[1]

        train_model_name = first_segment.split('_')[-1]

        epoch_time = second_segment.split('-')[1]
        accuracy = second_segment.split('-')[-1].split('.')[-2]

        loss_balancing = third_segment

        current_date = datetime.datetime.now().strftime('%d%m%Y')

        test_file_name = test_file.split('/')[-1][:-4]

        path = RESULT_PATH + \
               self.name + '_' + \
               train_model_name + '@' + epoch_time + '_' + \
               accuracy + '_' + \
               current_date + '_' + \
               test_file_name + \
               '#' + loss_balancing + '.csv'
        pairs.to_csv(path, index=False)

    def impersonate(self, impostor_signal, threshold, policy, x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, n_templates=10):
        """
        Method to test this model under impersonation attempts
        :param impostor_signal:     Audio signal against which this model is tested - shape (None, 1)
        :param threshold:           Verification threshold
        :param policy:              Verification policy - choices ['avg', 'any']
        :param x_mv_test:           Testing users' audio samples - shape (users, n_templates, None, 1)
        :param y_mv_test:           Testing users' labels - shape (users, n_templates)
        :param male_x_mv_test:      Male users' ids
        :param female_x_mv_test:    Female users' ids
        :param n_templates:         Number of audio samples to create a user template
        :return:                    {'m': impersonation rate against male users, 'f': impersonation rate against female users}
        """

        print('>', 'impersonating', self.name, 'model')
        mv_emb = self.embed(impostor_signal)
        mv_fac = np.zeros(len(np.unique(y_mv_test)))
        for class_index, class_label in enumerate(np.unique(y_mv_test)):
            template = [self.embed(
                signal) for signal in x_mv_test[class_index*n_templates:(class_index+1)*n_templates]]
            if policy == 'any':
                mv_fac[class_index] = len([1 for template_emb in np.array(
                    template) if 1 - spatial.distance.cosine(template_emb, mv_emb) > threshold])
            elif policy == 'avg':
                mv_fac[class_index] = 1 if 1 - spatial.distance.cosine(
                    mv_emb, np.mean(np.array(template), axis=0)) > threshold else 0
        print('>', 'impersonated', self.name, 'model')
        return {'m': len([index for index, fac in enumerate(mv_fac) if fac > 0 and index in male_x_mv_test]) / len(male_x_mv_test), 'f': len([index for index, fac in enumerate(mv_fac) if fac > 0 and index in female_x_mv_test]) / len(female_x_mv_test)}
