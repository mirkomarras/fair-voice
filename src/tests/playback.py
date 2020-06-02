#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf
import numpy as np
import argparse
import sys
import os

from helpers.audio import load_noise_paths, cache_noise_data, play_n_rec, decode_audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Playback functionality testing')

    # Parameters
    parser.add_argument('--audio_path', dest='audio_path', default='/beegfs/mm10572/voxceleb1/test/id10281/Yw8v8055uPc/00001.wav', type=str, action='store', help='Audio path')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--speaker_flag', dest='speaker_flag', default=0, type=int, choices=[0,1], action='store', help='Speaker flag')
    parser.add_argument('--room_flag', dest='room_flag', default=0, type=int, choices=[0,1], action='store', help='Room flag')
    parser.add_argument('--microphone_flag', dest='microphone_flag', default=0, type=int, choices=[0,1], action='store', help='Microphone flag')
    
    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Audio path: {}'.format(args.audio_path))
    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Speaker flag: {}'.format(args.speaker_flag))
    print('>', 'Room flag: {}'.format(args.room_flag))
    print('>', 'Microphone flag: {}'.format(args.microphone_flag))

    impulse_flags = [args.speaker_flag, args.room_flag, args.microphone_flag]
    
    print('Load impulse response paths')
    noise_paths = load_noise_paths('./data/vs_noise_data')
    
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)
    
    print('Noise samples')
    print('Speaker', noise_cache[noise_paths['speaker'][0]].shape)
    print('Room', noise_cache[noise_paths['room'][0]].shape)
    print('Microphone', noise_cache[noise_paths['microphone'][0]].shape)
    
    print('Compute playback & recording')
    xt = decode_audio(os.path.join(args.audio_path)).reshape((1, -1, 1)).astype(np.float32)
    xn = np.array(impulse_flags, dtype=np.float32).reshape(1, -1)
    
    print('> signal:', xt.shape)
    print('> impulse_flags:', xn.shape)
    
    @tf.function
    def forward(signal, impulse_flags):
      return play_n_rec((signal, impulse_flags), noises=noise_paths, cache=noise_cache, noise_strength='random')
    
    xf = forward(xt, xn).numpy()
    
    print('> playback signal:', xf.shape)
    
    print('> data flow in the model:')
    print('>>> original audio: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(xt.shape, xt.min(), xt.max(), xt.size / args.sample_rate))
    print('>>> p&rec audio: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(xf.shape, xf.min(), xf.max(), xf.size / args.sample_rate))
    
    print('Saving playback comparison plot')
    
    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches((16, 8))
    
    axes[0].plot(xt.ravel())
    axes[0].set_title('Speech sample')
    
    axes[1].plot(xf.ravel())
    axes[1].set_title('Playback sample')
    
    plt.savefig('./tests/playback_comparison.png')
    
    print('Saving original and playback audio samples')
    sf.write('./tests/original_audio.wav', np.squeeze(xt), args.sample_rate)
    sf.write('./tests/playback_audio.wav', np.squeeze(xf), args.sample_rate)

if __name__ == '__main__':
    main()

