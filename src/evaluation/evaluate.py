import os
import copy
import pickle
import functools
import argparse
from collections import namedtuple
from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics


def _check_sensitive_evaluation(func):
    def wrapper(self, **kwargs):
        binary_cols = kwargs.pop("binary_cols") if "binary_cols" in kwargs else None  # ["gender_1", "age_1"]
        binary_attributes = kwargs.pop(
            "binary_attributes") if "binary_attributes" in kwargs else None  # [["male", "female"], ["young", "old"]])

        if binary_cols is not None and binary_attributes is not None:
            compute_func = functools.partial(self._compute_metric_sensitive,
                                             binary_cols=binary_cols,
                                             binary_attributes=binary_attributes)
        else:
            compute_func = self._compute_metric

        metric_func = func(self)
        return compute_func(self.results, metric_func, **kwargs)

    return wrapper


class Evaluator:
    RESULTS_BASE_COLUMNS = ['Architecture', 'Loss_bal', 'Epoch', 'Train File', 'Test File', 'Accuracy']
    ROCMetrics = namedtuple('Metric', ["EER", "FAR", "FRR", "thresholds", "min_index"])

    def __init__(self, results_file):
        self.results_file = results_file
        self.results = pd.read_csv(results_file)

    @_check_sensitive_evaluation
    def ROC(self):
        return self._calculate_ROC

    @staticmethod
    def _calculate_ROC(y, y_score, **kwargs):
        """
        Function to calcolate EER, FAR, FRR and other important measurements for tests
        :param y:               real result
        :param y_score:         predicted result
        :return:                min_index
                                FAR
                                FRR
                                FAR (min index)
                                FRR (min index)
                                EER
                                Threshold
                                Thresholds
        """

        far, tpr, thresholds = sk_metrics.roc_curve(y, y_score, pos_label=1)
        frr = 1 - tpr
        abs_diffs = np.abs(far - frr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((far[min_index], frr[min_index]))
        # return min_index, far, frr, far[min_index], frr[min_index], eer, thresholds[min_index], thresholds
        return Evaluator.ROCMetrics(eer, far, frr, thresholds, min_index)

    @staticmethod
    def _compute_metric_sensitive(data,
                                  func,
                                  binary_cols: Sequence[str] = None,
                                  binary_labels: Sequence[Sequence[str]] = None,
                                  **kwargs):
        """
        Function used to extract from from data
        :param data: contains the results reported in the current .csv file stored from the test
        :param func: metric function to use
        :param binary_cols: list of columns of the csv containing sensitive attributes
        :param binary_labels: list of list of binary sensitives attributes to filter csv rows, each sublist must contain
                              attributes of the respective index column, first sublist filter rows of first 'binary_cols'
                              column
        :return:
        """
        flat_binary_labels = np.array(binary_labels).flatten()

        sensitive_metrics = dict.fromkeys(flat_binary_labels)
        data_labels_dfs = dict.fromkeys(flat_binary_labels)

        # Here are taken the columns containing the expected results (label) and the predicted ones (similarity)
        label = data.loc[:, 'label']
        similarity = data.loc[:, 'simlarity']
        # The data are passed to the function in order to calculate all the results
        total_metrics = func(label, similarity, **kwargs)
        sensitive_metrics["total"] = total_metrics

        # Extraction of similarity scores for each sensitive attribute in `binary_labels`
        for col, labels in zip(binary_cols, binary_labels):
            for lab in labels:
                data_lab = data[data[col] == lab]
                similarity = data_lab.loc[:, "simlarity"]
                cos_sim_labels = data_lab.loc[:, "label"]
                sensitive_metrics[lab] = func(cos_sim_labels, similarity, **kwargs)
                data_labels_dfs[lab] = data_lab

        return sensitive_metrics

    @staticmethod
    def _compute_metric(data,
                        func,
                        **kwargs):
        """
        Function used to extract from from data
        :param data: contains the results reported in the current .csv file stored from the test
        :param func: metric function to use
        :return:
        """

        label = data.loc[:, 'label']
        similarity = data.loc[:, 'simlarity']

        return func(label, similarity, **kwargs)

    def results_info(self):
        info = self.results_file.split('#')[0].split('_')
        arch_name = info[0]
        train_model_name = info[1].split('@')[0]
        epoch = info[1].split('@')[1]
        accuracy = info[2]
        accuracy = (accuracy[:2] + '.' + accuracy[-1]).replace('.', ',')
        test_file_name = info[4]

        balancing_levels = list(map(lambda x: x[:2], self.results_file.split('#')[1].split('_')))
        loss_bal = f"[{' - '.join(['.'.join(bal_level) for bal_level in balancing_levels])}]"

        base_record = [arch_name, loss_bal, epoch, train_model_name, test_file_name, accuracy]

        return dict(zip(self.RESULTS_BASE_COLUMNS, base_record))

    def kde_analysis(self, return_far_frr=False, plots_path='', plot_kwargs=None):
        res_df = self.results.copy()
        res_df[["lang_1", "user_1", "audio_1"]] = res_df["audio_1"].str.split(os.sep, expand=True)
        res_df[["lang_2", "user_2", "audio_2"]] = res_df["audio_2"].str.split(os.sep, expand=True)

        languages = res_df["lang_1"].unique()

        far_data = dict.fromkeys(languages)
        frr_data = dict.fromkeys(languages)
        for i, (lang, lang_df) in enumerate(res_df.groupby("lang_1")):
            far_data[lang] = {}
            frr_data[lang] = {}
            for user, user_df in lang_df.groupby("user_1"):
                sens = ' '.join([user_df["gender_1"].iloc[0], user_df["age_1"].iloc[0]])

                labels = user_df.loc[:, "label"]
                sim = user_df.loc[:, "simlarity"]

                roc_result = self._calculate_ROC(labels, sim)

                far_data[lang][user] = [roc_result.FAR[roc_result.min_index], sens]
                frr_data[lang][user] = [roc_result.FRR[roc_result.min_index], sens]

        if return_far_frr:
            return far_data, frr_data
        else:
            self.plot_far_frr(far_data, frr_data,
                              plot_func=sns.kdeplot, plots_path=plots_path, plot_kwargs=plot_kwargs)
            
    def plot_far_frr(self, far_data, frr_data, plot_func=sns.kdeplot, plots_path='', plot_kwargs=None):
        n_languages = len(far_data)
        plot_kwargs = plot_kwargs or {}
        
        fig_far, axs_far = plt.subplots(1, n_languages, figsize=(10, 10), sharey=True, sharex=True)
        fig_frr, axs_frr = plt.subplots(1, n_languages, figsize=(10, 10), sharey=True, sharex=True)
        
        for i, lang in enumerate(far_data):
            far_data_plot = pd.DataFrame.from_dict(
                far_data[lang], orient='index', columns=["FAR", "Demographic Group"]
            ).reset_index().rename(columns={'index': 'users'})
            frr_data_plot = pd.DataFrame.from_dict(
                frr_data[lang], orient='index', columns=["FRR", "Demographic Group"]
            ).reset_index().rename(columns={'index': 'users'})

            plot_func(x="FAR", data=far_data_plot, hue="Demographic Group",
                      ax=axs_far[i] if n_languages > 1 else axs_far, **plot_kwargs)
            plot_func(x="FRR", data=frr_data_plot, hue="Demographic Group",
                      ax=axs_frr[i] if n_languages > 1 else axs_frr, **plot_kwargs)
            axs_far[i].set_title(lang)
            axs_frr[i].set_title(lang)
        
        fig_far.savefig(plots_path.format(f"FAR") + ('.png' if plots_path[:-4] != '.png' else ''))
        fig_frr.savefig(plots_path.format(f"FRR") + ('.png' if plots_path[:-4] != '.png' else ''))
        
    def hist_analysis_mapped_far_frr(self,
                                     far_data,
                                     frr_data,
                                     far_label_0_le=0.1,
                                     frr_label_0_le=0.1,
                                     plots_path='',
                                     plot_kwargs=None):
        far_data = copy.deepcopy(far_data)
        frr_data = copy.deepcopy(frr_data)

        for lang in far_data:
            for user in far_data[lang]:
                user_far_data = far_data[lang][user]
                if far_label_0_le is not None:
                    far_data[lang][user] = [0 if user_far_data[0] <= far_label_0_le else 1, *user_far_data[1:]]

                user_frr_data = frr_data[lang][user]
                if frr_label_0_le is not None:
                    frr_data[lang][user] = [0 if user_frr_data[0] <= frr_label_0_le else 1, *user_frr_data[1:]]
                
        self.plot_far_frr(far_data, frr_data,
                          plot_func=sns.histplot, plots_path=plots_path, plot_kwargs=plot_kwargs)


if __name__ == "__main__":
    """
    Resnet34
    python3 evaluate.py --results_file /home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/results/resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.csv --plots_kde_filepath "/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_kde__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png" --far_label_0_le 0 --frr_label_0_le 0 --plots_hist_filepath "/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_0_0_hist__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png"
    
    X-Vector
    python3 evaluate.py --results_file /home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/results/xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.csv --plots_kde_filepath "/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_kde__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png" --far_label_0_le 0 --frr_label_0_le 0 --plots_hist_filepath "/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_0_0_hist__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png"
    """

    parser = argparse.ArgumentParser(description='Tensorflow Results Evaluation')

    # Parameters for testing a verifier against eer
    parser.add_argument('--results_file', dest='results_file', default='./results/resnet34vox_English-Spanish-train1@15_920_08032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.csv', type=str, action='store', help='Results file path')
    parser.add_argument('--plots_kde_filepath', dest='plots_kde_path', default='./plots/{}_plot.png', type=str, action='store', help='Path to save kde plots. It must contain a placeholder for formatting "FAR" or "FRR"')
    parser.add_argument('--far_label_0_le', dest='far_label_0_le', default=None, type=float, action='store', help='FAR values lower or equal to this parameter are mapped to 0, otherwise 1')
    parser.add_argument('--frr_label_0_le', dest='frr_label_0_le', default=None, type=float, action='store', help='FRR values lower or equal to this parameter are mapped to 0, otherwise 1')
    parser.add_argument('--plots_hist_filepath', dest='plots_hist_path', default='./plots/{}_plot.png', type=str, action='store', help='Path to save histogram plots. It must contain a placeholder for formatting "FAR" or "FRR"')

    args = parser.parse_args()

    ev = Evaluator(args.results_file)

    if not os.path.exists(os.path.dirname(args.plots_kde_path)):
        os.makedirs(os.path.dirname(args.plots_kde_path))

    ev.kde_analysis(plots_path=args.plots_kde_path)
    far, frr = ev.kde_analysis(return_far_frr=True)

    with open(r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/far_data__' + os.path.splitext(os.path.basename(args.results_file))[0] + '.pkl', 'wb') as f:
        pickle.dump(far, f)

    with open(r'/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/evaluation/frr_data__' + os.path.splitext(os.path.basename(args.results_file))[0] + '.pkl', 'wb') as f:
        pickle.dump(frr, f)

    if not os.path.exists(os.path.dirname(args.plots_hist_path)):
        os.makedirs(os.path.dirname(args.plots_hist_path))

    if args.far_label_0_le is not None or args.frr_label_0_le is not None:
        ev.hist_analysis_mapped_far_frr(
            far,
            frr,
            far_label_0_le=args.far_label_0_le,
            frr_label_0_le=args.frr_label_0_le,
            plots_path=args.plots_hist_path,
            plot_kwargs={'multiple': 'stack'}
        )