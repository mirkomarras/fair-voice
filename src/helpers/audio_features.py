#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
from collections import defaultdict
from typing import Union, Sequence, Iterable

import numpy as np
import pandas as pd
import scipy.stats as stats
import librosa
import pydub
import myprosody as mypr
import parselmouth
import parselmouth.praat as pm_praat
# import speech_recognition as sr

import audio as audio_helper


class AudioFeatureExtractor(object):
    # https://wiki.aalto.fi/pages/viewpage.action?pageId=149890776
    # F0_MIN = 80
    # F0_MAX = 450

    # https://osf.io/qe9k4/
    F0_MIN = 75
    F0_MAX = 500

    def __init__(self,
                 audio: Union[str, os.PathLike],
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
        self._temp_audio_path = os.path.join(os.getcwd(), "temp_" + os.path.basename(audio))
        librosa.output.write_wav(self._temp_audio_path, self._audio_librosa, sample_rate)

        self._audio_pydub = pydub.AudioSegment(
            self._audio_librosa.tobytes(),
            sample_width=4,  # because librosa convert the audio in a float32 numpy array, so 4 according to pydub
            frame_rate=self._sample_rate,
            channels=self._channels
        )
        self._audio_parselmouth = parselmouth.Sound(self._temp_audio_path)
        self._myprosody_path = myprosody_path or \
            os.path.join(
               os.path.dirname(os.path.dirname(os.path.dirname(audio_helper.__file__))),
               'myprosody'
            )
        # TODO Fix text extraction from audio
        # with sr.AudioFile(temp_audio_path) as source:
        #     audio_data = sr.Recognizer().record(source=source)
        #     self._audio_text = sr.Recognizer().recognize_google(audio_data)

    def __del__(self):
        if os.path.exists(self._temp_audio_path):
            os.remove(self._temp_audio_path)

        praat_temp_audio = os.path.splitext(self._temp_audio_path)[0] + '.TextGrid'
        if os.path.exists(praat_temp_audio):
            os.remove(praat_temp_audio)

    @property
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
    def intensity_dB(self, f0min=F0_MIN):
        intensity = pm_praat.call(self._audio_parselmouth, "To Intensity", f0min, 0)
        return pm_praat.call(intensity, "Get mean", 0, 0, "dB")

    @property
    def intensity_energy(self, f0min=F0_MIN):
        intensity = pm_praat.call(self._audio_parselmouth, "To Intensity", f0min, 0)
        return pm_praat.call(intensity, "Get mean", 0, 0, "Energy")

    @property
    def intensity_sones(self, f0min=F0_MIN):
        intensity = pm_praat.call(self._audio_parselmouth, "To Intensity", f0min, 0)
        return pm_praat.call(intensity, "Get mean", 0, 0, "Sones")

    def _get_intensity(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        intensity = pm_praat.call(self._audio_parselmouth, "To Intensity", f0min, 0)
        numPoints = pm_praat.call(pointProcess, "Get number of points")

        return [
            pm_praat.call(
                intensity,
                "Get value at time",
                pm_praat.call(pointProcess, "Get time from index", point),
                'Linear'
            ) for point in range(1, numPoints + 1)
        ]

    @property
    def intensity_mean(self, f0min=F0_MIN, f0max=F0_MAX):
        intensity = np.nan_to_num(self._get_intensity(f0min=f0min, f0max=f0max))
        return np.mean(intensity)

    @property
    def intensity_std(self, f0min=F0_MIN, f0max=F0_MAX):
        intensity = np.nan_to_num(self._get_intensity(f0min=f0min, f0max=f0max))
        return np.std(intensity)

    @property
    def intensity_skew(self, f0min=F0_MIN, f0max=F0_MAX):
        intensity = np.nan_to_num(self._get_intensity(f0min=f0min, f0max=f0max))
        return stats.skew(intensity)

    @property
    def intensity_kurt(self, f0min=F0_MIN, f0max=F0_MAX):
        intensity = np.nan_to_num(self._get_intensity(f0min=f0min, f0max=f0max))
        return stats.kurtosis(intensity)

    @property
    def jitter_localabsolute(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_local(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_rap(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_ppq5(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def jitter_ddp(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    @property
    def shimmer_localdB(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call([self._audio_parselmouth, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    @property
    def shimmer_local(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call([self._audio_parselmouth, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    @property
    def shimmer_apq3(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call([self._audio_parselmouth, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    @property
    def shimmer_apq5(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call([self._audio_parselmouth, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    @property
    def shimmer_apq11(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call([self._audio_parselmouth, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    @property
    def shimmer_dda(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        return pm_praat.call([self._audio_parselmouth, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    def _get_hnr(self, f0min=F0_MIN, f0max=F0_MAX):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        harmonicity = pm_praat.call(self._audio_parselmouth, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        numPoints = pm_praat.call(pointProcess, "Get number of points")

        return [
            pm_praat.call(
                harmonicity,
                "Get value at time",
                pm_praat.call(pointProcess, "Get time from index", point),
                'Linear'
            ) for point in range(1, numPoints + 1)
        ]

    @property
    def hnr_mean(self, f0min=F0_MIN, f0max=F0_MAX):
        harmonicity = np.nan_to_num(self._get_hnr(f0min=f0min, f0max=f0max))
        return np.mean(harmonicity)

    @property
    def hnr_std(self, f0min=F0_MIN, f0max=F0_MAX):
        harmonicity = np.nan_to_num(self._get_hnr(f0min=f0min, f0max=f0max))
        return np.std(harmonicity)

    @property
    def hnr_skew(self, f0min=F0_MIN, f0max=F0_MAX):
        harmonicity = np.nan_to_num(self._get_hnr(f0min=f0min, f0max=f0max))
        return stats.skew(harmonicity)

    @property
    def hnr_kurt(self, f0min=F0_MIN, f0max=F0_MAX):
        harmonicity = np.nan_to_num(self._get_hnr(f0min=f0min, f0max=f0max))
        return stats.kurtosis(harmonicity)

    def _get_f0(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        pitch = pm_praat.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
        numPoints = pm_praat.call(pointProcess, "Get number of points")

        return [
            pm_praat.call(
                pitch,
                "Get value at time",
                pm_praat.call(pointProcess, "Get time from index", point),
                unit,
                'Linear'
            ) for point in range(1, numPoints + 1)
        ]

    # @property
    # def f0_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
    #     pitch = pm_praat.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
    #     return pm_praat.call(pitch, "Get mean", 0, 0, unit)

    @property
    def f0_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f0 = np.nan_to_num(self._get_f0(f0min=f0min, f0max=f0max, unit=unit))
        return np.mean(f0)

    @property
    def f0_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f0 = np.nan_to_num(self._get_f0(f0min=f0min, f0max=f0max, unit=unit))
        return np.std(f0)

    @property
    def f0_skew(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f0 = np.nan_to_num(self._get_f0(f0min=f0min, f0max=f0max, unit=unit))
        return stats.skew(f0)

    @property
    def f0_kurt(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f0 = np.nan_to_num(self._get_f0(f0min=f0min, f0max=f0max, unit=unit))
        return stats.kurtosis(f0)

    # @property
    # def f0_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
    #     pitch = pm_praat.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
    #     return pm_praat.call(pitch, "Get standard deviation", 0, 0, unit)

    def _get_formant(self, formant_type, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        pointProcess = pm_praat.call(self._audio_parselmouth, "To PointProcess (periodic, cc)", f0min, f0max)
        formants = pm_praat.call(self._audio_parselmouth, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        numPoints = pm_praat.call(pointProcess, "Get number of points")

        # Measure formants only at glottal pulses
        return [
            pm_praat.call(
                formants,
                "Get value at time",
                formant_type,
                pm_praat.call(pointProcess, "Get time from index", point),
                unit,
                'Linear'
            ) for point in range(1, numPoints + 1)
        ]

    @property
    def f1_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f1 = np.nan_to_num(self._get_formant(1, f0min=f0min, f0max=f0max, unit=unit))
        return np.mean(f1)

    @property
    def f1_median(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f1 = np.nan_to_num(self._get_formant(1, f0min=f0min, f0max=f0max, unit=unit))
        return np.median(f1)

    @property
    def f1_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f1 = np.nan_to_num(self._get_formant(1, f0min=f0min, f0max=f0max, unit=unit))
        return np.std(f1)

    @property
    def f1_skew(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f1 = np.nan_to_num(self._get_formant(1, f0min=f0min, f0max=f0max, unit=unit))
        return stats.skew(f1)

    @property
    def f1_kurt(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f1 = np.nan_to_num(self._get_formant(1, f0min=f0min, f0max=f0max, unit=unit))
        return stats.kurtosis(f1)

    @property
    def f2_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f2 = np.nan_to_num(self._get_formant(2, f0min=f0min, f0max=f0max, unit=unit))
        return np.mean(f2)

    @property
    def f2_median(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f2 = np.nan_to_num(self._get_formant(2, f0min=f0min, f0max=f0max, unit=unit))
        return np.median(f2)

    @property
    def f2_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f2 = np.nan_to_num(self._get_formant(2, f0min=f0min, f0max=f0max, unit=unit))
        return np.std(f2)

    @property
    def f2_skew(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f2 = np.nan_to_num(self._get_formant(2, f0min=f0min, f0max=f0max, unit=unit))
        return stats.skew(f2)

    @property
    def f2_kurt(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f2 = np.nan_to_num(self._get_formant(2, f0min=f0min, f0max=f0max, unit=unit))
        return stats.kurtosis(f2)

    @property
    def f3_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f3 = np.nan_to_num(self._get_formant(3, f0min=f0min, f0max=f0max, unit=unit))
        return np.mean(f3)

    @property
    def f3_median(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f3 = np.nan_to_num(self._get_formant(3, f0min=f0min, f0max=f0max, unit=unit))
        return np.median(f3)

    @property
    def f3_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f3 = np.nan_to_num(self._get_formant(3, f0min=f0min, f0max=f0max, unit=unit))
        return np.std(f3)

    @property
    def f3_skew(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f3 = np.nan_to_num(self._get_formant(3, f0min=f0min, f0max=f0max, unit=unit))
        return stats.skew(f3)

    @property
    def f3_kurt(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f3 = np.nan_to_num(self._get_formant(3, f0min=f0min, f0max=f0max, unit=unit))
        return stats.kurtosis(f3)

    @property
    def f4_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f4 = np.nan_to_num(self._get_formant(4, f0min=f0min, f0max=f0max, unit=unit))
        return np.mean(f4)

    @property
    def f4_median(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f4 = np.nan_to_num(self._get_formant(4, f0min=f0min, f0max=f0max, unit=unit))
        return np.median(f4)

    @property
    def f4_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f4 = np.nan_to_num(self._get_formant(4, f0min=f0min, f0max=f0max, unit=unit))
        return np.std(f4)

    @property
    def f4_skew(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f4 = np.nan_to_num(self._get_formant(4, f0min=f0min, f0max=f0max, unit=unit))
        return stats.skew(f4)

    @property
    def f4_kurt(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        f4 = np.nan_to_num(self._get_formant(4, f0min=f0min, f0max=f0max, unit=unit))
        return stats.kurtosis(f4)

    @staticmethod
    def formant_position(*args: "AudioFeatureExtractor"):
        f1_median = [afe.f1_median for afe in args]
        f2_median = [afe.f2_median for afe in args]
        f3_median = [afe.f3_median for afe in args]
        f4_median = [afe.f4_median for afe in args]

        return (
                   stats.zscore(f1_median) +
                   stats.zscore(f2_median) +
                   stats.zscore(f3_median) +
                   stats.zscore(f4_median)
               ) / 4

    @property
    def formant_dispersion(self):
        return (self.f4_median - self.f1_median) / 3

    @property
    def formant_average(self):
        return np.mean([self.f1_median, self.f2_median, self.f3_median, self.f4_median])

    @property
    def mff(self):
        """
        Smith, D. R., & Patterson, R. D. (2005). The interaction of glottal-pulse rate and vocal-tract length in
        judgements of speaker size, sex, and age. The Journal of the Acoustical Society of America, 118(5), 3177-3186.
        :return:
        """
        return (self.f1_median * self.f2_median * self.f3_median * self.f4_median) ** 0.25

    @property
    def fitch_vtl(self):
        """
        Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus
        macaques. The Journal of the Acoustical Society of America, 102(2), 1213-1222.
        :return:
        """
        return (
                   (1 * (35000 / (4 * self.f1_median))) +
                   (3 * (35000 / (4 * self.f2_median))) +
                   (5 * (35000 / (4 * self.f3_median))) +
                   (7 * (35000 / (4 * self.f4_median)))
               ) / 4

    @property
    def delta_f(self):
        """
        Reby,D.,& McComb,K.(2003). Anatomical constraints generate honesty: acoustic cues to age and weight in the
        roars of red deer stags. Animal Behaviour, 65, 519e-530.
        :return:
        """
        factors = np.array([0.5, 1.5, 2.5, 3.5])
        formants_median = np.array([self.f1_median, self.f2_median, self.f3_median, self.f4_median])
        return (factors * formants_median).sum() / (factors ** 2).sum()

    @property
    def vtl_delta_f(self):
        """
        Reby,D.,& McComb,K.(2003). Anatomical constraints generate honesty: acoustic cues to age and weight in the
        roars of red deer stags. Animal Behaviour, 65, 519e-530.
        :return:
        """
        return 35000 / (2 * self.delta_f)

    @property
    def number_syllables(self):
        return mypr.myspsyl(self._temp_audio_path, self._myprosody_path)

    @property
    def number_pauses(self):
        return mypr.mysppaus(self._temp_audio_path, self._myprosody_path)

    @property
    def rate_of_speech(self):
        # syllables/sec original duration
        return mypr.myspsr(self._temp_audio_path, self._myprosody_path)

    @property
    def articulation_rate(self):
        # syllables/sec speaking duration
        return mypr.myspatc(self._temp_audio_path, self._myprosody_path)

    @property
    def speaking_duration_without_pauses(self):
        return mypr.myspst(self._temp_audio_path, self._myprosody_path)

    @property
    def speaking_duration_with_pauses(self):
        return mypr.myspod(self._temp_audio_path, self._myprosody_path)

    @property
    def balance(self):
        # ratio (speaking duration)/(original duration)
        return mypr.myspbala(self._temp_audio_path, self._myprosody_path)

    @property
    def gender(self):
        """
        if z4>97 and z4<=163:
            Male
        elif z4>163 and z4<=245:
            Female
        else:
            Voice not recognized

        :return: Tuple[z4, p-value/sample_size]
        """
        return mypr.myspgend(self._temp_audio_path, self._myprosody_path)

    @property
    def mood(self):
        """
        if (z4>97 and z4<=114) or (z4>163 and z4<=197):
            Showing no emotion, normal,
        elif (z4>114 and z4<=135) or (z4>197 and z4<=226):
            Reading
        elif (z4>135 and z4<=163) or (z4>226 and z4<=245):
            speaking passionately
        else:
            Voice not recognized

        :return: Tuple[z4, p-value/sample_size]
        """
        return mypr.myspgend(self._temp_audio_path, self._myprosody_path)

    def get_features(self, subset=None):
        subset = subset or set(self.__class__.__dict__.keys())

        return {
            k: getattr(self, k) for k, v in self.__class__.__dict__.items() if isinstance(v, property) and k in subset 
        }

    # TODO just set some properties for audio text features - to test
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


def extract_audio_features(audios: Sequence[Union[str, os.PathLike]],
                           audio_base_path: Union[str, os.PathLike],
                           subset: Iterable = None):
    features = {}
    users_audios = defaultdict(dict)
    for audio in audios:
        afe = AudioFeatureExtractor(os.path.join(audio_base_path, audio))
        features[audio] = afe.get_features(subset=subset)
        users_audios[os.path.dirname(audio)][audio] = afe

    static_funcs = [k for k, v in AudioFeatureExtractor.__dict__.items() if isinstance(v, staticmethod) and k in subset]
    for func in static_funcs:
        for user_lang in users_audios:
            user_vals = getattr(AudioFeatureExtractor, func)(*list(users_audios[user_lang].values()))
            for audio, audio_val in zip(users_audios[user_lang].keys(), user_vals):
                features[audio][func] = audio_val

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Features Extraction')

    parser.add_argument('--fairvoice_path', dest='fairvoice_path', default=r'/home/meddameloni/FairVoice',
                        type=str, action='store', help='Path of the FairVoice dataset')
    parser.add_argument('--test_path', dest='test_path',
                        default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/test_ENG_SPA_75users_6samples_50neg_5pos.csv',
                        type=str, action='store', help='Path of the test set containing the audios to extract')
    parser.add_argument('--metadata_path', dest='metadata_path', default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/metadata_ENG_SPA_75users_6minsample.csv',
                        type=str, action='store', help='Path of the csv containing the metadata of each user')
    parser.add_argument('--features', dest='features', default=[], nargs='+',
                        type=str, action='store', help='List of the features to extract')
    parser.add_argument('--save_path', dest='save_path',
                        default='/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/audio_analysis/test_new_audio_features.pkl', type=str, action='store',
                        help='Save path of the extracted audio_features')

    args = parser.parse_args()

    test_df = pd.read_csv(args.test_path)
    test_audios = np.unique(np.concatenate([test_df['audio_1'].tolist(), test_df['audio_2'].tolist()]))
    
    metadata = pd.read_csv(args.metadata_path).set_index(["language_l1", "id_user"])
    keys = ("gender", "age")

    if not args.features:
        features_subset = [
            'signaltonoise_dB',
            'dBFS',
            'intensity_mean',
            'intensity_std',
            'intensity_skew',
            'intensity_kurt',
            # 'rms',
            # 'max',
            # 'duration_seconds',
            # 'jitter_localabsolute',
            'jitter_local',
            # 'jitter_rap',
            # 'jitter_ppq5',
            # 'jitter_ddp',
            'shimmer_localdB',
            # 'shimmer_local',
            # 'shimmer_apq3',
            # 'shimmer_apq5',
            # 'shimmer_apq11',
            # 'shimmer_dda',
            'hnr_mean',
            'hnr_std',
            'hnr_skew',
            'hnr_kurt',
            'f0_mean',
            'f0_std',
            'f0_skew',
            'f0_kurt',
            'f1_mean',
            'f1_std',
            'f1_skew',
            'f1_kurt',
            'f2_mean',
            'f2_std',
            'f2_skew',
            'f2_kurt',
            'f3_mean',
            'f3_std',
            'f3_skew',
            'f3_kurt',
            'f4_mean',
            'f4_std',
            'f4_skew',
            'f4_kurt',
            'formant_position',
            'formant_dispersion',
            'formant_average',
            'mff',
            'fitch_vtl',
            'delta_f',
            'vtl_delta_f',
            # 'number_syllables',
            # 'number_pauses',
            # 'rate_of_speech',
            # 'articulation_rate',
            'speaking_duration_without_pauses',
            # 'speaking_duration_with_pauses',
            # 'balance',
            # 'gender',
            # 'mood'
        ]
    else:
        features_subset = args.features

    audio_features = extract_audio_features(test_audios, args.fairvoice_path, subset=features_subset)
    for _audio in audio_features:
        lang, user = os.path.split(os.path.dirname(_audio))
        
        gender, age = metadata.loc[(lang, user), keys]
        for k, v in zip(keys, [gender, age]):
            audio_features[_audio][k] = v
        audio_features[_audio]["language"] = lang

    with open(args.save_path, 'wb') as f:
        pickle.dump(audio_features, f)
