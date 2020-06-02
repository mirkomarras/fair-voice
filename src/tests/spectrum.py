#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf
import numpy as np
import argparse
import decimal
import math
import sys
import os

from helpers.audio import decode_audio, get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win

def normalize_frames(m, epsilon=1e-12):
    frames = []
    means = []
    stds = []
    for v in m:
        means.append(np.mean(v))
        stds.append(np.std(v))
        frames.append((v - np.mean(v)) / max(np.std(v), epsilon))
    return np.array(frames), np.array(means), np.array(stds)

def get_np_spectrum(signal, sample_rate, num_fft=512, frame_size=0.025, frame_stride=0.01):
    assert signal.ndim == 1, 'Only 1-dim signals supported'

    frames = framesig(signal, frame_len=frame_size * sample_rate, frame_step=frame_stride * sample_rate, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=num_fft))
    fft = fft[:, :(num_fft // 2 + 1)]
    fft_norm, fft_mean, fft_std = normalize_frames(fft.T)

    return fft_norm, fft_mean, fft_std

def main():
    parser = argparse.ArgumentParser(description='Spectrum functionality testing')

    # Parameters
    parser.add_argument('--audio_path', dest='audio_path', default='/beegfs/mm10572/voxceleb1/test/id10281/Yw8v8055uPc/00001.wav', type=str, action='store', help='Audio path')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')

    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Audio path: {}'.format(args.audio_path))
    print('>', 'Sample rate: {}'.format(args.sample_rate))

    print('Compute spectrum')
    xt = decode_audio(os.path.join(args.audio_path)).astype(np.float32)
    print('> signal:', xt.shape)

    sp_np, _, _ = get_np_spectrum(xt, args.sample_rate)
    print('> numpy spectrum:', sp_np.shape, np.min(sp_np), np.max(sp_np))

    @tf.function
    def forward(signal):
        return get_tf_spectrum(signal)

    sp_tf = np.squeeze(forward(xt.reshape((1, -1, 1))).numpy())
    print('> tensorflow spectrum:', sp_tf.shape, np.min(sp_tf), np.max(sp_tf))

    print('Saving spectrum comparison plot')

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches((16, 8))

    axes[0].matshow(sp_np, aspect="auto")
    axes[0].set_title('Numpy spectrum')

    axes[1].matshow(sp_tf, aspect="auto")
    axes[1].set_title('Tensorflow spectrum')

    plt.savefig('./tests/spectrum_comparison.png')



if __name__ == '__main__':
    main()

