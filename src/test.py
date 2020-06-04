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


import  models.model
from models.vggvox import VggVox
from models.xvector import XVector
from models.resnet50vox import ResNet50Vox
from models.resnet34vox import ResNet34Vox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Parameters for verifier
    parser.add_argument('--net', dest='net', default='resnet34vox', type=str, action='store', help='Network model architecture')

    # Parameters for testing a verifier against eer
    parser.add_argument('--audio_dir', dest='audio_dir', default='/FairVoice2/French/', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--test_pair_path', dest='test_pair_path', default='../meta/6_3_2020_14_12_French/French_test.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--test_n_pair', dest='test_n_pair', default=3200, type=int, action='store', help='Number of test pairs')
    parser.add_argument('--model_path', dest='model_path', default="../trained_model/9_3_2020_14_0_xvector__meta_6_3_2020_14_12_French_French_train/weights-01-0.807.tf", type=str,
    action=store, help="Model selected" )
    parser.add_argument('--aggregation', dest='aggregation', default='avg', type=str, choices=['avg', 'vlad', 'gvlad'], action='store', help='Type of aggregation')
    # Parameters for raw audio
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
    #test_file = '../meta/6_3_2020_14_12_French/French_test.csv'
    #audio_dir = '/home/.../FairVoice2/French/'
    #model_path = '../trained_model/9_3_2020_14_0_xvector__meta_6_3_2020_14_12_French_French_train/weights-01-0.807.tf'

    # Load test data
    #test_data = load_test_data(args.test_base_path, args.test_pair_path, args.test_n_pair, args.sample_rate, args.n_seconds)
    # test_data  = load_test_from_csv(args.test_csv_path,args.audio_dir);
    # Create model
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    #model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')), n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model=available_nets[args.net.split('/')[0]](id=(int(args.net.split('/')[1].replace('v','')) if '/v' in args.net else -1), n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.build(classes=0)
    # Test model

    print('Testing model')
    t1 = time.time()
    model.test_and_save_on_csv(args.test_pair_path,args.audio_dir,aggregation=args.aggregation,model_path = args.model_path);
    t2 = time.time()
    print('>', t2-t1, 'seconds for testing')


if __name__ == '__main__':
    main()
