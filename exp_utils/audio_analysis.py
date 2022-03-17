import os
import pickle
import argparse

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
            stat_data = []
            p_v_info = []
            for col in cols:
                gr1 = df_gb.get_group(gr_groups[0])[col]
                gr2 = df_gb.get_group(gr_groups[1])[col]
                gr1 = gr1[np.isfinite(gr1)]
                gr2 = gr2[np.isfinite(gr2)]
                stat, pval = ttest_ind(gr1, gr2)
                stat_data.append(stat)
                p_v_info.append(self.PV001 if pval <= 0.01 else (self.PV005 if pval <= 0.05 else ''))
            out_data.append(stat_data)
            p_value_info.append(p_v_info)

        out_df = pd.DataFrame(out_data, columns=cols, index=grouping)
        if return_stat_data:
            return out_df
        else:
            # annot = np.array(out_data).round(3).astype(str).astype(object) + np.array(p_value_info, dtype=object)
            annot = np.array(p_value_info, dtype=object)
            return sns.heatmap(data=out_df, annot=annot, fmt='s', **sns_kw)  # , annot_kws={'rotation': 'vertical'})

    def distribution_boxplot(self, hue, subset=None, sub_feat_axs=None, lower_gr_eq=None, min_diff_eq=None, sns_kw=None):
        cols = subset or self._df_feat.columns
        sns_kw = sns_kw or {}

        df = self._df_feat[cols + [hue]]
        df = pd.melt(df.reset_index(), id_vars=['lang', 'id_user', 'audio', hue])

        if sub_feat_axs is not None:
            if 'ax' in sns_kw:
                raise ValueError("Cannot use `sub_feat_axs` when `ax` is in `sns_kw`")

            if len(sub_feat_axs.flat) < len(cols):
                raise ValueError("Number of columns should be lower or equal than the number of axes")

            for ax, col in zip(sub_feat_axs.flat, cols):
                data = df[df['variable'] == col]
                if lower_gr_eq is not None and col == "f0_mean":
                    min_diff_eq = min_diff_eq or data['value'].std()
                    hue_gb = dict(data.groupby(hue).__iter__())
                    grs = list(hue_gb.keys())
                    temp_data = pd.concat(list(hue_gb.values()))
                    print("len:", len(temp_data))
                    print("users:", len(temp_data['id_user'].unique()))
                    while abs(hue_gb[grs[0]]['value'].median() - hue_gb[grs[1]]['value'].median()) > min_diff_eq:
                        for gr, df_gr in hue_gb.items():
                            func = "min" if gr == lower_gr_eq else "max"
                            lang, user, del_a = df_gr.loc[np.isclose(df_gr['value'].astype(float), getattr(df_gr['value'], func)())][['lang', 'id_user','audio']].iloc[0]
                            hue_gb[gr] = df_gr[~((df_gr['lang'] == lang) & (df_gr['id_user'] == user) & (df_gr['audio'] == del_a))]
                    data = pd.concat(list(hue_gb.values()))
                    print("len:", len(data))
                    print("users:", len(data['id_user'].unique()))
                # if col == "f0_mean":
                    # bs = mat_cb.boxplot_stats(data["value"].dropna())
                    # data = data[~data['value'].isin(bs.pop(0)['fliers']) & data['g']]
                sns.boxplot(x="variable", y="value", data=data, hue=hue, ax=ax, **sns_kw)
        else:
            return sns.boxplot(x="variable", y="value", data=df, hue=hue, **sns_kw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Features Extraction')

    parser.add_argument('--audio_features_path', dest='audio_features_path',
                        default=r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/audio_analysis/test_audio_features.pkl',
                        type=str, action='store', help='Path of the audio features extracted by AudioFeatureExtractor')
    parser.add_argument('--features', dest='features', default=[], nargs='+',
                        type=str, action='store', help='List of the features to extract')
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

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[2, 2])
    hue = "gender"
    afa.distribution_boxplot(
        hue,
        subset=['signaltonoise_dB', 'dBFS', 'jitter_local', 'shimmer_localdB', 'hnr', 'f0_mean', 'f0_std'],
        sub_feat_axs=axs,
        lower_gr_eq="male",
        # min_diff_eq=1,
        sns_kw={'palette': ['#F5793A', '#A95AA1']}
    )
    for ax in axs.flat:
        ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'{hue.upper()}_distribution_barplot_hue.png'))
    plt.close()
