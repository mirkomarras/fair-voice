import os
import time
import pandas as pd
import numpy as np
import pydub
import librosa
import librosa.display
import matplotlib.pyplot as plt
from _collections import defaultdict
from dataset_preprocess import ages_vals


def sample_group_info(source,
                      type_file=None,
                      metadata="/home/meddameloni/FairVoice/metadata/metadata.csv",
                      dataset_dir="/home/meddameloni/FairVoice",
                      distinct_group=False,
                      young_cap="fourties"):
    """
    Takes users from different sources and returns a csv file with the average of several audio info of the
    audios of each user

    :param source: file containing data (list of users, train file, test file, dataframe)
    :param type_file: type_file controls which users are considered for audios, if it is not None
    users are taken from "train" or "test" files
    :param metadata: file containing metadata of the users if source is not a dataframe
    :param dataset_dir: base directory of audio files
    :param distinct_group: if true returns info of audios of each user considering the distinct group "female", "male",
    "old", "young", so each user is present 2 times in the output file. if false returns info of audios of each unique
    group, so "female old", "female young", "male old", "male young"
    :param young_cap: this should be the same value used to generate train and test files. It is not re-written in other
    files, so it needs to be given manually as input
    :return:
    """
    if type_file is not None:
        if type_file not in ["test", "train"]:
            raise TypeError("type_file must be None or 'test' or 'train'")

    if not isinstance(source, pd.DataFrame):
        if os.path.splitext(source)[1] == '.csv':
            out_name = 'sample_info_{}'.format(os.path.basename(source))
            data = pd.read_csv(source, sep=',', encoding='utf-8')
        else:
            out_name = 'sample_info_{}.csv'.format(os.path.basename(source))
            with open(source, 'r') as source_file:
                try:
                    data = ast.literal_eval(source_file.read())
                except ValueError:
                    raise ValueError('non-csv files are considered files '
                                     'containing a python array-like object of users')
    else:
        data = source
        out_name = 'sample_info_{}.csv'.format(hash(str(data)))

    if isinstance(data, list):  # python array-like object
        meta = pd.read_csv(metadata, sep=',', encoding='utf-8')
        df = pd.DataFrame()
        for t_lang, t_id in data:
            df = pd.concat([df, meta[(meta["language_l1"] == t_lang) & (meta["id_user"] == t_id)]])
    else:
        if type_file is "test":  # test file have the paths of the audios of the 2 compared users in each row
            users_audios = defaultdict(set)
            for row in data.itertuples():
                user_path_1, audio_1 = os.path.split(row.audio_1)
                user_path_2, audio_2 = os.path.split(row.audio_2)
                users_audios['_'.join(os.path.split(user_path_1))].add(audio_1)
                users_audios['_'.join(os.path.split(user_path_2))].add(audio_2)

            meta = pd.read_csv(metadata, sep=',', encoding='utf-8')
            df = pd.DataFrame()
            for lang_id in users_audios.keys():
                user_lang, user_id = lang_id.split('_')
                df = pd.concat([df, meta[(meta["language_l1"] == user_lang) & (meta["id_user"] == user_id)]])
        elif type_file is "train":  # train file have the path of the audios of only a user in each row
            users_audios = defaultdict(set)
            for row in data.itertuples():
                user_path, audio = os.path.split(row.audio)
                users_audios['_'.join(os.path.split(user_path))].add(audio)

            meta = pd.read_csv(metadata, sep=',', encoding='utf-8')
            df = pd.DataFrame()
            for lang_id in users_audios.keys():
                user_lang, user_id = lang_id.split('_')
                df = pd.concat([df, meta[(meta["language_l1"] == user_lang) & (meta["id_user"] == user_id)]])
        else:  # data is a pandas dataframe
            df = data

    rows = []

    ages = {k: "young" if ages_vals.index(young_cap) > i else "old" for i, k in enumerate(ages_vals)}

    if not distinct_group:
        df_by_gender = df.groupby("gender")

        # age for loop inside gender for loop for unique groups
        for gender in df_by_gender.groups:
            gender_group = df_by_gender.get_group(gender)
            gender_by_age = gender_group.set_index("age").groupby(ages)
            for age in gender_by_age.groups:
                age_group = gender_by_age.get_group(age).reset_index()
                for row in age_group.itertuples():
                    if type_file is not None:
                        audios = [os.path.join(dataset_dir, row.language_l1, row.id_user, audio_file) for audio_file in
                                  users_audios['{}_{}'.format(row.language_l1, row.id_user)]]
                    else:
                        audios = [file.path for file in os.scandir(os.path.join(dataset_dir, row.language_l1, row.id_user))
                                  if file.name.endswith('.wav') or file.name.endswith('mp3')]
                    get_audio_noise(audios[0])
                    audio_segments = [pydub.AudioSegment.from_wav(_audio) for _audio in audios]
                    lens_audio_segments = [audio_s.duration_seconds for audio_s in audio_segments]
                    row_data = [
                        '{}_{}'.format(gender, age),
                        row.language_l1,
                        row.id_user,
                        len(audios),
                        np.mean(lens_audio_segments),
                        min(lens_audio_segments),
                        max(lens_audio_segments),
                        np.mean([audio_s.dBFS for audio_s in audio_segments]),
                        np.mean([audio_s.max_dBFS for audio_s in audio_segments]),
                        np.mean([get_audio_noise(_audio) for _audio in audios])
                    ]
                    rows.append(row_data)
    else:
        df_by_gender = df.groupby("gender")

        # gender for loop (distinct_group)
        for gender in df_by_gender.groups:
            gender_group = df_by_gender.get_group(gender)

            for row in gender_group.itertuples():
                if type_file is not None:
                    audios = [os.path.join(dataset_dir, row.language_l1, row.id_user, audio_file) for audio_file in
                              users_audios['{}_{}'.format(row.language_l1, row.id_user)]]
                else:
                    audios = [file.path for file in
                              os.scandir(os.path.join(dataset_dir, row.language_l1, row.id_user))
                              if file.name.endswith('.wav') or file.name.endswith('mp3')]
                get_audio_noise(audios[0])
                audio_segments = [pydub.AudioSegment.from_wav(_audio) for _audio in audios]
                lens_audio_segments = [audio_s.duration_seconds for audio_s in audio_segments]
                row_data = [
                    '{}'.format(gender),
                    row.language_l1,
                    row.id_user,
                    len(audios),
                    np.mean(lens_audio_segments),
                    min(lens_audio_segments),
                    max(lens_audio_segments),
                    np.mean([audio_s.dBFS for audio_s in audio_segments]),
                    np.mean([audio_s.max_dBFS for audio_s in audio_segments]),
                    np.mean([get_audio_noise(_audio) for _audio in audios])
                ]
                rows.append(row_data)

        df_by_age = df.set_index("age").groupby(ages)
        for age in df_by_age.groups:  # age for loop (distinct_group)
            age_group = df_by_age.get_group(age).reset_index()

            for row in age_group.itertuples():
                if type_file is not None:
                    audios = [os.path.join(dataset_dir, row.language_l1, row.id_user, audio_file) for audio_file in
                              users_audios['{}_{}'.format(row.language_l1, row.id_user)]]
                else:
                    audios = [file.path for file in
                              os.scandir(os.path.join(dataset_dir, row.language_l1, row.id_user))
                              if file.name.endswith('.wav') or file.name.endswith('mp3')]
                get_audio_noise(audios[0])
                audio_segments = [pydub.AudioSegment.from_wav(_audio) for _audio in audios]
                lens_audio_segments = [audio_s.duration_seconds for audio_s in audio_segments]
                row_data = [
                    '{}'.format(age),
                    row.language_l1,
                    row.id_user,
                    len(audios),
                    np.mean(lens_audio_segments),
                    min(lens_audio_segments),
                    max(lens_audio_segments),
                    np.mean([audio_s.dBFS for audio_s in audio_segments]),
                    np.mean([audio_s.max_dBFS for audio_s in audio_segments]),
                    np.mean([get_audio_noise(_audio) for _audio in audios])
                ]
                rows.append(row_data)

    if not os.path.exists(os.path.join(os.path.dirname(metadata), 'sample_info')):
        os.mkdir(os.path.join(os.path.dirname(metadata), 'sample_info'))

    pd.DataFrame(rows, columns=['group', 'language', 'id_user', 'n_sample', 'duration_avg',
                                'min_duration', 'max_duration', 'dBFS_avg', 'max_dBFS_avg', 'noise_avg']).to_csv(
        os.path.join(os.path.dirname(metadata), 'sample_info', ('distinct_' if distinct_group else '') + out_name),
        index=False,
        encoding='utf-8'
    )


def sample_group_info_groupby(sample_info_file):
    """
    It should be used on a "sample_info_file", that is the file generated from the function "sample_group_info". It
    returns a csv file containing the average of the audio info for each group, so a csv file with only 4 rows

    :param sample_info_file: the sample info file given in input containing the average of the audio info of all the
    audios of each user
    :return:
    """
    out_dir = os.path.dirname(sample_info_file)
    file_name = os.path.splitext(sample_info_file)[0]
    distinct_group = True if file_name.startswith('distinct_') else False
    out_file_name = os.path.join(out_dir, '{}_{}group_avg.csv'.format(file_name, 'distinct_' if distinct_group else ''))

    df = pd.read_csv(sample_info_file, sep=',', encoding='utf-8')
    df_group = df.groupby('group')

    groups = [gr for gr in df_group.groups]
    out_df = pd.concat([df_group.get_group(gr).mean() for gr in groups], axis=1).T

    out_df.insert(0, 'group', groups)

    out_df.to_csv(out_file_name, index=False, encoding='utf-8')


"""
================
Vocal separation
================

This notebook demonstrates a simple technique for separating vocals (and
other sporadic foreground signals) from accompanying instrumentation.

This is based on the "REPET-SIM" method of `Rafii and Pardo, 2012
<http://www.cs.northwestern.edu/~zra446/doc/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%20ISMIR%202012.pdf>`_, but includes a couple of modifications and extensions:

    - FFT windows overlap by 1/4, instead of 1/2
    - Non-local filtering is converted into a soft mask by Wiener filtering.
      This is similar in spirit to the soft-masking method used by `Fitzgerald, 2012
      <http://arrow.dit.ie/cgi/viewcontent.cgi?article=1086&context=argcon>`_,
      but is a bit more numerically stable in practice.
"""

# Code source: Brian McFee
# License: ISC


def get_audio_noise(audio_path):
    """
    As stated above this code was taken from a notebook in the website of librosa
    :param audio_path: path of the audio from which extract the noise score
    :return: noise score of the audio
    """
    #############################################
    # Load an example with vocals.
    y, sr = librosa.load(audio_path, sr=None)

    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))

    #######################################
    # Plot a 5-second slice of the spectrum
    """
    idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()
    """

    ###########################################################
    # The wiggly lines above are due to the vocal component.
    # Our goal is to separate them from the accompanying
    # instrumentation.
    #

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    # Giacomo Medda modification: hop_length = 1024
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr, hop_length=2048)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    ##############################################
    # The raw filter output can be used as a mask,
    # but it sounds better if we use soft-masking.

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # Giacomo Medda modification: to avoid memory warning
    plt.close()

    # Giacomo Medda modification: we needed just a metric to represent a noise "score" of the audio
    # as the numpy matrix standard norm (Frobenius norm) of the difference between original and de-noised audio
    return np.linalg.norm(S_full - S_background)

    ##########################################
    # Plot the same slice, but separated into its foreground and background

    # sphinx_gallery_thumbnail_number = 2
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('/home/meddameloni/FairVoice/metadata/sample_info/audio_plot {}.png'.format(time.asctime()))


if __name__ == "__main__":
    print()
