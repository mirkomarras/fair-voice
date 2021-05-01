#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import librosa
import random
import time
import sys
import os


import models.model
from models.vggvox import VggVox
from models.xvector import XVector
from models.resnet50vox import ResNet50Vox
from models.resnet34vox import ResNet34Vox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Parameters for verifier
    parser.add_argument('--net', dest='net', default='resnet34vox', type=str, action='store', help='Network model architecture')

    # Parameters for testing a verifier against eer
    parser.add_argument('--test_base_path', dest='test_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--test_pair_path', dest='test_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--tecst_n_pair', dest='test_n_pair', default=3200, type=int, action='store', help='Number of test pairs')
    parser.add_argument('--test_file', dest='test_file', default='/home/meddameloni/dl-fair-voice/exp/test/English-test2.csv',type=str, action='store', help='CSV file used for testing')
    parser.add_argument('--audio_dir', dest='audio_dir', default='/home/meddameloni/FairVoice/', type=str, action='store', help='audio folder path')
    parser.add_argument('--model_path', dest='model_path', default='/home/meddameloni/dl-fair-voice/exp/trained_model/resnet34vox_19062020_0606_English-Spanish-train3#00_10/weights-15-0.870.h5', type=str, action='store', help='audio folder path')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')

    args = parser.parse_args()

    print('Parameters summary')

    print('>', 'Net: {}'.format(args.net))

    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Test pairs dataset path: {}'.format(args.test_base_path))
    print('>', 'Test pairs path: {}'.format(args.test_pair_path))
    print('>', 'Number of test pairs: {}'.format(args.test_n_pair))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))
    print('>', 'Test file: {}'.format(args.test_file))
    print('>', 'Audio directory {}'.format(args.audio_dir))
    print('>', 'Model path: {}'.format(args.model_path))


    # Create model
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}

    model=available_nets[args.net.split('/')[0]](id=(int(args.net.split('/')[1].replace('v','')) if '/v' in args.net else -1), n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.build(classes=0)

    # Test model
    print('Testing model')
    t1 = time.time()
    model.test_and_save_on_csv(args.test_file, args.audio_dir, aggregation='avg', model_path=args.model_path)
    t2 = time.time()
    print('>', t2-t1, 'seconds for testing')


if __name__ == '__main__':
    main()
