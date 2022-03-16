import os
import pickle
import argparse

import numpy as np
import pandas as pd
import pandas.api.types as pd_types
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


class AudioFeatureAnalyzer:

    PV005 = '*'
    PV01 = '^'
    
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
                p_v_info.append(self.PV005 if pval <= 0.05 else (self.PV01 if pval <= 0.1 else ''))
            out_data.append(stat_data)
            p_value_info.append(p_v_info)

        out_df = pd.DataFrame(out_data, columns=cols, index=grouping)
        if return_stat_data:
            return out_df
        else:
            # annot = np.array(out_data).round(3).astype(str).astype(object) + np.array(p_value_info, dtype=object)
            annot = np.array(p_value_info, dtype=object)
            return sns.heatmap(data=out_df, annot=annot, fmt='s', **sns_kw)  # , annot_kws={'rotation': 'vertical'})


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
            'jitter_localabsolute',
            # 'jitter_local',
            # 'jitter_rap',
            # 'jitter_ppq5',
            # 'jitter_ddp',
            # 'shimmer_localdB',
            'shimmer_local',
            # 'shimmer_apq3',
            # 'shimmer_apq5',
            # 'shimmer_apq11',
            # 'shimmer_dda',
            'hnr',
            'f0_mean',
            'f0_std',
            'number_syllables',
            'number_pauses',
            'rate_of_speech',
            'articulation_rate',
            'speaking_duration_without_pauses',
            # 'speaking_duration_with_pauses',
            'balance',
            'gender',
            'age'
            # 'mood'
        ]
    else:
        features_subset = args.features

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = afa.correlation_heatmap(subset=features_subset, sns_kw={'ax': ax})
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, 'audio_features_correlation.png'))
    plt.close()

    stat_diff = afa.mean_stat_difference(["gender", "age"], subset=features_subset)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, 'audio_features_stat_difference.png'))
    plt.close()
