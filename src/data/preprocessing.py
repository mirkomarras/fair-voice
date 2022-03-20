import os
import inspect
import itertools
import random
import pickle
import argparse
from typing import Union, Sequence, Any, Dict, Iterable, Hashable

import numpy as np
import pandas as pd


Language = Hashable
random.seed(32)
np.random.seed(32)


def preprocess_data(metadata_path,
                    n,
                    languages,
                    min_sample=0,
                    last_younger="fourties",
                    user_in: Dict[Language, Iterable[str]] = None):
    """

    :param metadata_path: file of metadata with information about each user
    :param n: number of users for each demographic group
    :param languages: list of languages to use
    :param min_sample: minimum of samples that a user must have
    :param last_younger: define the last age group that must be denoted as "younger"
    :param user_in: a dict that maps each language to a list of users that must be inside the output data. Only n users
                    among these ones will be taken
    :return:
    """
    df = pd.read_csv(metadata_path)

    df = df[df["language_l1"].isin(languages)]
    df = df[~df["gender"].isna() & ~df["age"].isna()]

    ages = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]
    if last_younger in ages:
        split_age_idx = ages.index(last_younger)
    else:
        raise ValueError(f"\"{last_younger}\" is not in {ages}")

    map_ages = {k: ("younger" if i <= split_age_idx else "older") for i, k in enumerate(ages)}
    df["age"] = df["age"].map(map_ages)

    df = df[df["n_sample"] >= min_sample]

    new_df = []
    for x, g_df in df.groupby(["gender", "age"]):
        if user_in:
            new_g_df = []
            for lang in user_in:
                lang_g_df = g_df[g_df["language_l1"] == lang]
                lang_g_df = lang_g_df[lang_g_df["id_user"].isin(user_in[lang])]
                new_g_df.append(lang_g_df.sample(n))
            g_df = pd.concat(new_g_df)

        new_df.append(g_df)

    new_df = pd.concat(new_df)

    return new_df


def test_file_from_df(df: pd.DataFrame,
                      neg_pairs,
                      pos_pairs,
                      fairvoice_path=None,
                      samples_per_user=None):
    """

    :param df: a DataFrame in the same format of the one returned by `preprocess_data`
    :param neg_pairs:
    :param pos_pairs:
    :param fairvoice_path:
    :param sample_per_user: the number of samples that must be used for each user
    :return:
    """

    def random_combination(iterable, r):
        """Random selection from itertools.combinations(iterable, r)"""
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)

    fairvoice_path = fairvoice_path or \
        os.path.join(os.path.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, os.pardir, 'FairVoice')
    samples_per_user = samples_per_user or df["n_sample"].max()

    users_audios = dict.fromkeys(df["language_l1"].unique().tolist())
    users_info = dict.fromkeys(df["language_l1"].unique().tolist())
    all_audios = set()
    for lang in users_audios:
        users_audios[lang] = dict.fromkeys(df.loc[df["language_l1"] == lang, "id_user"].unique().tolist())
        users_info[lang] = dict.fromkeys(df.loc[df["language_l1"] == lang, "id_user"].unique().tolist())

        df_info = df.set_index(["language_l1", "id_user"])
        for user in users_audios[lang]:
            user_samples = np.random.permutation(os.listdir(os.path.join(fairvoice_path, lang, user)))
            users_audios[lang][user] = user_samples[:samples_per_user]
            users_info[lang][user] = dict(zip(
                ["gender", "age"],
                [df_info.loc[(lang, user), "gender"], df_info.loc[(lang, user), "age"]]
            ))
            for sample in users_audios[lang][user]:
                all_audios.add(os.path.join(lang, user, sample))

    print('>', '#Total audios:', len(all_audios))

    test_df = []
    for lang in users_audios:
        for user in users_audios[lang]:
            n_pos = pos_pairs
            for (audio1, audio2) in random_combination(itertools.combinations(users_audios[lang][user], 2), n_pos):
                test_df.append([
                    os.path.join(lang, user, audio1),
                    os.path.join(lang, user, audio2),
                    users_info[lang][user]["age"],
                    "",
                    users_info[lang][user]["gender"],
                    "",
                    1
                ])

            other_users_audios = list(filter(lambda a: user not in a and lang in a, all_audios))
            n_neg = neg_pairs
            while n_neg > 0:
                for audio1 in users_audios[lang][user]:
                    if n_neg == 0:
                        break

                    audio2 = np.random.choice(other_users_audios)

                    test_df.append([
                        os.path.join(lang, user, audio1),
                        audio2,
                        users_info[lang][user]["age"],
                        users_info[lang][os.path.basename(os.path.dirname(audio2))]["age"],
                        users_info[lang][user]["gender"],
                        users_info[lang][os.path.basename(os.path.dirname(audio2))]["gender"],
                        0
                    ])

                    n_neg -= 1

    test_df = pd.DataFrame(test_df, columns=["audio_1", "audio_2", "age_1", "age_2", "gender_1", "gender_2", "label"])

    return test_df


if __name__ == "__main__":
    """
    python3 preprocessing.py --metadata_path /home/meddameloni/FairVoice/metadata.csv --languages English Spanish --n_users 75 --min_sample 6 --needed_users_path /home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/ENG_SPA_1_TRAIN_users_dict.pkl --output_metadata_path /home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/metadata_ENG_SPA_75users_6minsample.csv --neg_pairs 50 --pos_pairs 5 --fairvoice_path /home/meddameloni/FairVoice --samples_per_user 6 --output_test_path /home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/test_ENG_SPA_75users_6samples_50neg_5pos.csv
    """

    parser = argparse.ArgumentParser(description='Tensorflow counterfactual fairness preprocessing')

    # Parameters for testing a verifier against eer
    parser.add_argument('--metadata_path', dest='metadata_path', default='./FairVoice/metadata.csv', type=str, action='store', help='Metadata path')
    parser.add_argument('--languages', dest='languages', default=["English", "Spanish"], type=str, nargs='+', action='store', help='List of languages to keep')
    parser.add_argument('--n_users', dest='n_users', default=130, type=int, action='store', help='Number of users for each demographic group')
    parser.add_argument('--min_sample', dest='min_sample', default=6, type=int, action='store', help='Min sample per user')
    parser.add_argument('--needed_users_path', dest='needed_users_path', default='./data/ENG_SPA_1_TRAIN_users_dict.py', type=str, action='store', help='Path to pickle dict of [Language => Array of users] that are allowed to be in output')
    parser.add_argument('--output_metadata_path', dest='output_metadata_path', default=None, type=str, action='store', help='Output metadata path')
    parser.add_argument('--neg_pairs', dest='neg_pairs', default=50, type=int, action='store', help='Number of negative pairs')
    parser.add_argument('--pos_pairs', dest='pos_pairs', default=5, type=int, action='store', help='Number of positive pairs')
    parser.add_argument('--fairvoice_path', dest='fairvoice_path', default='./FairVoice', type=str, action='store', help='FairVoice path')
    parser.add_argument('--samples_per_user', dest='samples_per_user', default=6, type=int, action='store', help='Number of samples per user. If None all samples are used')
    parser.add_argument('--output_test_path', dest='output_test_path', default=None, type=str, action='store', help='Output test path')

    args = parser.parse_args()

    if args.output_metadata_path is None:
        args.output_metadata_path = f"metadata_{'_'.join(args.languages)}_{args.n_users}users_{args.min_sample}minsample.csv"

    if args.output_test_path is None:
        args.output_test_path = f"test_{'_'.join(args.languages)}_{args.n_users}users_{args.samples_per_user}samples_{args.neg_pairs}neg_{args.pos_pairs}pos.csv"

    print('>', 'Metadata Path: {}'.format(args.metadata_path))
    print('>', 'FairVoice Path: {}'.format(args.fairvoice_path))
    print('>', '#Users Sample for each Demographic Group: {}'.format(args.n_users))
    print('>', 'Languages: {}'.format(args.languages))
    print('>', 'Min Number of Samples per User: {}'.format(args.min_sample))
    print('>', 'Output Metadata Path: {}'.format(args.output_metadata_path))
    print('>', 'Negative Pairs: {}'.format(args.neg_pairs))
    print('>', 'Positive Pairs: {}'.format(args.pos_pairs))
    print('>', 'Samples per User: {}'.format(args.samples_per_user))
    print('>', 'Output Test Path: {}'.format(args.output_test_path))

    if args.needed_users_path:
        with open(args.needed_users_path, 'rb') as f:
            needed_users = pickle.load(f)
    else:
        needed_users = None

    out_df = preprocess_data(args.metadata_path,
                             args.n_users,
                             languages=args.languages,
                             min_sample=args.min_sample,
                             user_in=needed_users)
    out_df.to_csv(args.output_metadata_path, index=False)

    test_dataframe = test_file_from_df(out_df, args.neg_pairs, args.pos_pairs, fairvoice_path=args.fairvoice_path, samples_per_user=args.samples_per_user)
    test_dataframe.to_csv(args.output_test_path, index=False)
