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

from helpers.audio import decode_audio, get_tf_filterbanks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def normalize_frames(m, epsilon=1e-12):
    frames = []
    means = []
    stds = []
    for v in m:
        means.append(np.mean(v))
        stds.append(np.std(v))
        frames.append((v - np.mean(v)) / max(np.std(v), epsilon))
    return np.array(frames), np.array(means), np.array(stds)

def get_np_filterbanks(signal, sample_rate=16000, n_filters=24, frame_size=0.025, frame_stride=0.01):
    assert signal.ndim == 1, 'Only 1-dim signals supported'

    # Framing computation
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    padded_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.pad(signal, [(0, padded_signal_length - signal_length)], 'constant', constant_values=0)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming window computation
    frames *= np.hamming(frame_length)

    # FFT computation
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter banks computation
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # Normalization
    fft_norm, fft_mean, fft_std = normalize_frames(filter_banks)

    return fft_norm, fft_mean, fft_std

def main():
    parser = argparse.ArgumentParser(description='Filterbanks functionality testing')

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

    sp_np, _, _ = get_np_filterbanks(xt, args.sample_rate)
    print('> numpy spectrum:', sp_np.shape, sp_np.min(), sp_np.max())

    @tf.function
    def forward(signal):
        return get_tf_filterbanks(signal)

    sp_tf = np.squeeze(forward(xt.reshape((1, -1, 1))).numpy())
    print('> tensorflow filterbanks:', sp_tf.shape, sp_tf.min(), sp_tf.max())

    print('Saving filterbanks comparison plot')

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches((16, 8))

    axes[0].matshow(sp_np, aspect="auto")
    axes[0].set_title('Numpy spectrum')

    axes[1].matshow(sp_tf, aspect="auto")
    axes[1].set_title('Tensorflow filterbanks')

    plt.savefig('./tests/filterbanks_comparison.png')



if __name__ == '__main__':
    main()

