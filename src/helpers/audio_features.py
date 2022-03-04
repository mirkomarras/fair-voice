#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import librosa
import pydub
import myprosody as mypr
import parselmouth.praat as parselmouth
import speech_recognition as sr

from src.helpers import audio as audio_helper


class AudioFeatureExtractor(object):
    # https://wiki.aalto.fi/pages/viewpage.action?pageId=149890776
    # F0_MIN = 80
    # F0_MAX = 450

    # https://osf.io/qe9k4/
    F0_MIN = 75
    F0_MAX = 500

    def __init__(self,
                 audio: [str, os.PathLike],
                 sample_rate=16000,
                 n_seconds=3,
                 myprosody_path=None):
        self._audio_path = audio
        self._channels = 1
        self._sample_rate = sample_rate
        self._audio_librosa = audio_helper.decode_audio_fix_size(audio,
                                                                 sample_rate=sample_rate,
                                                                 n_seconds=n_seconds,
                                                                 input_format='raw')
        temp_audio_path = "temp_" + os.path.basename(audio)
        librosa.output.write_wav(temp_audio_path, self._audio_librosa, sample_rate)
        self._audio_pydub = pydub.AudioSegment(
            self._audio_librosa.tobytes(),
            sample_width=4,  # because librosa convert the audio in a float32 numpy array, so 4 according to pydub
            frame_rate=self._sample_rate,
            channels=self._channels
        )
        self._audio_parselmouth = parselmouth.Sound(temp_audio_path)
        self._myprosody_path = myprosody_path or \
                               os.path.join(
                                   os.path.dirname(os.path.dirname(os.path.dirname(audio_helper.__file__))),
                                   'myprosody'
                               )
        # TODO Fix text extraction from audio
        # with sr.AudioFile(temp_audio_path) as source:
        #     audio_data = sr.Recognizer().record(source=source)
        #     self._audio_text = sr.Recognizer().recognize_google(audio_data)

        os.remove(temp_audio_path)

    @propert
    def signaltonoise_dB(self, axis=0, ddof=0):
        # https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python
        m = self._audio_librosa.mean(axis)
        sd = self._audio_librosa.std(axis=axis, ddof=ddof)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))

    @property
    def dBFS(self):
        return self._audio_pydub.dBFS

    @property
    def rms(self):
        return self._audio_pydub.rms

    @property
    def max(self):
        return self._audio_pydub.max

    @property
    def duration_seconds(self):
        return self._audio_pydub.duration_seconds

    @property
    def jitter_localabsolute(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_local(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_rap(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_ppq5(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_ddp(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def shimmer_localdB(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call([self._audio_parselmouth, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02,
                                1.3, 1.6)

    @property
    def shimmer_local(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call([self._audio_parselmouth, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3,
                                1.6)

    @property
    def shimmer_apq3(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call([self._audio_parselmouth, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3,
                                1.6)

    @property
    def shimmer_apq5(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call([self._audio_parselmouth, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3,
                                1.6)

    @property
    def shimmer_apq11(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call([self._audio_parselmouth, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3,
                                1.6)

    @property
    def shimmer_dda(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = parselmouth.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return parselmouth.call([self._audio_parselmouth, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3,
                                1.6)

    @property
    def hnr(self):
        harmonicity = parselmouth.call(self._audio_parselmouth, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        return parselmouth.call(harmonicity, "Get mean", 0, 0)

    @property
    def f0_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        pitch = parselmouth.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
        return parselmouth.call(pitch, "Get mean", 0, 0, unit)

    @property
    def f0_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        pitch = parselmouth.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
        return parselmouth.call(pitch, "Get standard deviation", 0, 0, unit)

    @property
    def number_syllables(self):
        return mypr.myspsyl(self._audio_path, self._myprosody_path)

    # TODO just set some properties for audio text features - to test+
    # @property
    # def avg_words_per_sec(self):
    #     return self._audio_text // self.duration_seconds
    #
    # @property
    # def avg_words_length(self):
    #     return sum([len(word) for word in self._audio_text.split()]) // len(self._audio_text.split())
    #
    # @property
    # def words_per_utterance(self):
    #     return len(self._audio_text.split())
