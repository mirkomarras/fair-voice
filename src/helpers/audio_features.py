#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from typing import Union, Sequence

import numpy as np
import pandas as pd
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

    @property
    def hnr(self):
        harmonicity = pm_praat.call(self._audio_parselmouth, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        return pm_praat.call(harmonicity, "Get mean", 0, 0)

    @property
    def f0_mean(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        pitch = pm_praat.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
        return pm_praat.call(pitch, "Get mean", 0, 0, unit)

    @property
    def f0_std(self, f0min=F0_MIN, f0max=F0_MAX, unit='Hertz'):
        pitch = pm_praat.call(self._audio_parselmouth, "To Pitch", 0.0, f0min, f0max)
        return pm_praat.call(pitch, "Get standard deviation", 0, 0, unit)

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

    def get_features(self):
        return {k: getattr(self, k) for k, v in self.__class__.__dict__.items() if isinstance(v, property)}

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


def extract_audio_features(audios: Sequence[Union[str, os.PathLike]], audio_base_path: Union[str, os.PathLike]):
    features = {}
    for audio in audios:
        features[audio] = AudioFeatureExtractor(os.path.join(audio_base_path, audio)).get_features()

    return features


if __name__ == "__main__":
    fairvoice_path = r'/home/meddameloni/FairVoice'
    test_path = r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/test_ENG_SPA_75users_6samples_50neg_5pos.csv'

    test_df = pd.read_csv(test_path)
    test_audios = np.unique(np.concatenate([test_df['audio_1'].tolist(), test_df['audio_2'].tolist()]))

    audio_features = extract_audio_features(test_audios, fairvoice_path)

    import pickle
    with open("test_audio_features.pkl", 'wb') as f:
        pickle.dump(audio_features, f)
