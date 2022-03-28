import os
import pickle
import argparse
from typing import Callable, Union, Dict, Any

import numpy as np
import pandas as pd
import pandas.api.types as pd_types
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as mat_cb
from scipy.stats import ttest_ind


class AudioFeatureAnalyzer:

    PV005 = '*'
    PV001 = '^'

    def __init__(self, path):
        self._audio_f_path = path
        with open(self._audio_f_path, 'rb') as f:
            self._audio_f = pickle.load(f)

        self._df_feat = self._audio_f_to_df(self._audio_f)

    @staticmethod
    def _audio_f_to_df(audio_f):
        df = pd.DataFrame.from_dict(audio_f, orient='index').reset_index()

        df[["lang", "id_user", "audio"]] = df['index'].str.split(os.sep, expand=True)
        del df['index']

        return df.set_index(["lang", "id_user", "audio"])

    def correlation_heatmap(self, subset=None, method='pearson', sns_kw=None):
        cols = subset or self._df_feat.columns
        sns_kw = sns_kw or {}

        corr_df = self._df_feat[cols].corr(method=method)
        rename = {
            'speaking_duration_with_pauses': 'speak_with_pauses',
            'speaking_duration_without_pauses': 'speak_without_pauses'
        }
        corr_df = corr_df.rename(index=rename, columns=rename)

        return sns.heatmap(data=corr_df, xticklabels=True, yticklabels=True, linewidths=.2, **sns_kw)

    def mean_stat_difference(self, grouping, subset=None, return_stat_data=False, sns_kw=None):
        cols = subset or self._df_feat.columns
        sns_kw = sns_kw or {}

        cols = [col for col in cols if pd_types.is_numeric_dtype(self._df_feat[col])]
        df = self._df_feat[cols + grouping]

        out_data = []
        p_value_info = []
        for gr in grouping:
            df_gb = df.groupby(gr)
            if df_gb.ngroups != 2:
                raise ValueError(f"The column `{gr}` does not have 2 values")

            gr_groups = list(df_gb.groups.keys())
            pval_data = []
            p_v_info = []
            for col in cols:
                gr1 = df_gb.get_group(gr_groups[0])[col]
                gr2 = df_gb.get_group(gr_groups[1])[col]
                gr1 = gr1[np.isfinite(gr1)]
                gr2 = gr2[np.isfinite(gr2)]
                stat, pval = ttest_ind(gr1, gr2)
                pval_data.append(pval)
                p_v_info.append(self.PV001 if pval <= 0.01 else (self.PV005 if pval <= 0.05 else ''))
            out_data.append(pval_data)
            p_value_info.append(p_v_info)

        out_df = pd.DataFrame(out_data, columns=cols, index=grouping)
        if return_stat_data:
            return out_df
        else:
            # annot = np.array(out_data).round(3).astype(str).astype(object) + np.array(p_value_info, dtype=object)
            annot = np.array(p_value_info, dtype=object)
            return sns.heatmap(data=out_df, annot=annot, fmt='s', **sns_kw)  # , annot_kws={'rotation': 'vertical'})

    def distribution_boxplot(self,
                             hue,
                             subset=None,
                             sub_feat_axs=None,
                             equalize_stat=False,
                             results_file=None,
                             feat_imp_file=None,
                             n_eq=2,
                             p_value=0.05,
                             sns_kw=None):
        cols = subset or self._df_feat.columns
        sns_kw = sns_kw or {}

        df_imp = pd.read_csv(feat_imp_file)
        imp_list = df_imp.T.sort_values(0, ascending=False).index[:n_eq].tolist()
        cols = [c for c in imp_list if c in cols]

        df = self._df_feat[cols + [hue]]
        df = [pd.melt(df.reset_index(), id_vars=['lang', 'id_user', 'audio', hue])]

        if sub_feat_axs is not None:
            if 'ax' in sns_kw:
                raise ValueError("Cannot use `sub_feat_axs` when `ax` is in `sns_kw`")

            if len(sub_feat_axs.flat) < len(cols):
                raise ValueError("Number of columns should be lower or equal than the number of axes")

            for ax, col in zip(sub_feat_axs.flat, cols):
                print(f"Plotting {col}")
                data = df[0][df[0]['variable'] == col]

                if equalize_stat:
                    hue_gb = dict(data.groupby(hue).__iter__())
                    grs = list(hue_gb.keys())
                    temp_data = pd.concat(list(hue_gb.values()))
                    print("len:", len(temp_data))
                    print("users:", len(temp_data[['lang', 'id_user']].apply(lambda x: (x['lang'], x['id_user']), axis=1).unique()))

                    while True:
                        gr1 = hue_gb[grs[0]]['value']
                        gr2 = hue_gb[grs[1]]['value']
                        gr1 = gr1[np.isfinite(gr1)]
                        gr2 = gr2[np.isfinite(gr2)]
                        if ttest_ind(gr1, gr2)[1] > p_value:
                            break

                        funcs = dict(zip(grs, ["min", "max"] if np.median(gr1) <= np.median(gr2) else ["max", "min"]))
                        for gr, df_gr in hue_gb.items():
                            data_to_del = df_gr.loc[
                                np.isclose(df_gr['value'].astype(float), getattr(df_gr['value'], funcs[gr])())
                            ]
                            lang, user, del_a = data_to_del[['lang', 'id_user','audio']].iloc[0]
                            hue_gb[gr] = df_gr[
                                ~((df_gr['lang'] == lang) & (df_gr['id_user'] == user) & (df_gr['audio'] == del_a))
                            ]
                            df[0] = df[0][
                                ~((df[0]['lang'] == lang) & (df[0]['id_user'] == user) & (df[0]['audio'] == del_a))
                            ]
                    data = pd.concat(list(hue_gb.values()))
                    print("\nlen:", len(data))
                    print("users:", len(data[['lang', 'id_user']].apply(lambda x: (x['lang'], x['id_user']), axis=1).unique()), '\n')

                    # df[0] = pd.concat([df[0][df[0]['variable'] != col], data])

                sns.boxplot(x="variable", y="value", data=data, hue=hue, ax=ax, **sns_kw)

                with open(results_file, 'rb') as f:
                    res_data = pickle.load(f)
                res_rate = os.path.basename(results_file).split('_')[0]

                handles, labels = ax.get_legend_handles_labels()

                rate_labels = {}
                rates_per_group = {}
                for gr, gr_df in data.groupby(hue):
                    gr_df[res_rate] = gr_df[['lang', 'id_user']].apply(lambda x: res_data[x['lang']][x['id_user']][0],
                                                                       axis=1)
                    rates_per_group[gr] = gr_df[res_rate].to_numpy()
                    rate_labels[gr] = str(round(gr_df[res_rate].mean(), 3))

                mean_r = round(np.concatenate(list(rates_per_group.values())).mean(), 3)
                for gr in rate_labels:
                    r = rates_per_group[gr]
                    rate_labels[gr] = f"{rate_labels[gr]} | {len(r[r > mean_r])}/{len(r)} > {res_rate} = {mean_r}"

                labels = list(map(lambda x: f"{x} {rate_labels[x]}", labels))
                ax.legend(handles, labels)
        else:
            return sns.boxplot(x="variable", y="value", data=df, hue=hue, **sns_kw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Features Extraction')

    parser.add_argument('--audio_features_path', dest='audio_features_path',
                        default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/audio_analysis/test_audio_features.pkl',
                        type=str, action='store', help='Path of the audio features extracted by AudioFeatureExtractor')
    parser.add_argument('--features', dest='features', default=[], nargs='+',
                        type=str, action='store', help='List of the features to extract')
    parser.add_argument('--results_file', dest='results_file',
                        default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.pkl',
                        type=str, action='store', help='Path of the pickle file of FAR or FRR for each user')
    parser.add_argument('--feature_importance_file', dest='feature_importance_file',
                        default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10/RF_far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.csv',
                        type=str, action='store', help='File of the csv with feature and its importance.')
    parser.add_argument('--n', dest='n', default=6, type=int, action='store',
                        help='First n features to consider')
    parser.add_argument('--equalize', dest='equalize_stat', default=False, action='store_true',
                        help='If distribution between groups should be equalized for the first `n` features')
    parser.add_argument('--save_path', dest='save_path',
                        default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/audio_analysis',
                        type=str, action='store', help='Path used to save the plots')

    args = parser.parse_args()

    afa = AudioFeatureAnalyzer(args.audio_features_path)

    if not args.features:
        features_subset = [
            'signaltonoise_dB',
            'dBFS',
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
            'hnr',
            'f0_mean',
            'f0_std',
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = afa.correlation_heatmap(subset=features_subset, sns_kw={'ax': ax})
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'audio_features_correlation.png'))
    plt.close()

    stat_diff = afa.mean_stat_difference(["gender", "age"], subset=features_subset)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, 'audio_features_stat_difference.png'))
    plt.close()

    fig, axs = plt.subplots(3, 3, figsize=(12, 14))
    for _ax in axs.flat[args.n:]:
        fig.delaxes(_ax)

    hue = "gender"
    afa.distribution_boxplot(
        hue,
        subset=['signaltonoise_dB', 'dBFS', 'jitter_local', 'shimmer_localdB', 'hnr', 'f0_mean', 'f0_std'],
        sub_feat_axs=axs,
        equalize_stat=args.equalize_stat,
        results_file=args.results_file,
        feat_imp_file=args.feature_importance_file,
        n_eq=args.n,
        p_value=0.05,
        sns_kw={'palette': ['#F5793A', '#A95AA1']}
    )
    for ax in axs.flat:
        ax.set_xlabel('')

    net = args.results_file.split('__')[1].split('_')[0]
    plt.suptitle(net)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(
        args.save_path,
        f'{net}_{hue.upper()}_{"equalized" if args.equalize_stat else "all"}_distribution_barplot_hue.png'
    ))
    plt.close()
