import sys
import os
import re
import argparse
import shutil
import inspect
from datetime import datetime as dt
from typing import Sequence, Any
from collections import namedtuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Constants for framework paths

# RESULT_PATH indicates the path where are stored the resulting .csv from models tests
RESULT_PATH = '/home/meddameloni/dl-fair-voice/exp/results/interspeech/'
# DEST_PATH indicates the path where are going to be saved the measurements of the calculation done in the script
DEST_PATH = '/home/meddameloni/fair-voice/exp/metrics'

DEST_PLOTS_RQ1_PATH = '/home/meddameloni/fair-voice/exp/metrics/rq1_plots'
DEST_PLOTS_RQ2_PATH = '/home/meddameloni/fair-voice/exp/metrics/rq2_plots'

RESULTS_BASE_COLUMNS = ['Architecture', 'Loss_bal', 'Epoch', 'Train File', 'Test File', 'Accuracy']
METRICS_NAMES = ["EER", "FAR", "FRR"]


ROCMetrics = namedtuple('Metric', ["EER", "FAR", "FRR", "thresholds", "min_index"])

map_test = {'1': "Age", '2': "Gender", '3': "Random"}
map_train = {'1': "UB", '2': "NB", '3': "SB"}


def _fair_plot(func):
    """
    def wrapper(df, color_fair='green', color_unfair='red', eps=1e-2, **kwargs):
        fairness_colors = {}
        (gr1_name, gr1), (gr2_name, gr2) = [(name, gr) for name, gr in df.groupby("group")]

        for th in df["thresholds"].unique():
            # these 2 lines are used only to verify that the threshold `th` exists in each group df
            gr1_frr = gr1[gr1["thresholds"] == th]["FRR"]
            gr2_frr = gr2[gr2["thresholds"] == th]["FRR"]

            if not gr1_frr.empty and not gr2_frr.empty:
                gr_dist: Iterable = func(th, gr1_name, gr1, gr2_name, gr2, **kwargs)

                if all(d < eps for d in gr_dist):
                    fairness_colors[th] = color_fair
                else:
                    fairness_colors[th] = color_unfair
            else:
                fairness_colors[th] = None

        return fairness_colors
    """

    def wrapper(df, **kwargs):
        fairness_scores = pd.DataFrame(df["thresholds"].unique(), columns=["thresholds"])
        (gr1_name, gr1), (gr2_name, gr2) = [(name, gr) for name, gr in df.groupby("group")]

        for th in df["thresholds"].unique():
            # these 2 lines are used only to verify that the threshold `th` exists in each group df
            gr1_frr = gr1[gr1["thresholds"] == th]["FRR"]
            gr2_frr = gr2[gr2["thresholds"] == th]["FRR"]

            if not gr1_frr.empty and not gr2_frr.empty:
                gr_dist: Dict[str, Any] = func(th, gr1_name, gr1, gr2_name, gr2, **kwargs)

                for col, dist in gr_dist.items():
                    fairness_scores.loc[fairness_scores["thresholds"] == th, col] = dist

        return fairness_scores

    return wrapper


@_fair_plot
def fair_demographic_parity(th, gr1_name, gr1, gr2_name, gr2, **kwargs):
    """
    The percentage of receiving a positive score ((TP_g + FP_g) / len(users_g)) must be equal among groups
    Source: https://dl.acm.org/doi/pdf/10.1145/3194770.3194776
    """
    labels_df = kwargs.pop("labels_df")

    gr1_pp = len(labels_df[gr1_name][labels_df[gr1_name]["simlarity"] >= th])
    gr2_pp = len(labels_df[gr2_name][labels_df[gr2_name]["simlarity"] >= th])

    gr_dist = abs((gr1_pp / len(labels_df[gr1_name])) - (gr2_pp / len(labels_df[gr2_name])))

    return {"demographic_parity": gr_dist}


@_fair_plot
def fair_equal_opportunity(th, gr1_name, gr1, gr2_name, gr2, **kwargs):
    gr1_tpr = 1 - gr1[gr1["thresholds"] == th]["FRR"].iloc[0]
    gr2_tpr = 1 - gr2[gr2["thresholds"] == th]["FRR"].iloc[0]

    gr_dist = abs(gr1_tpr - gr2_tpr)

    return {"equal_opportunity": gr_dist}


@_fair_plot
def fair_equalized_odds(th, gr1_name, gr1, gr2_name, gr2, **kwargs):
    gr1_tpr = 1 - gr1[gr1["thresholds"] == th]["FRR"].iloc[0]
    gr2_tpr = 1 - gr2[gr2["thresholds"] == th]["FRR"].iloc[0]

    gr1_fpr = 1 - gr1[gr1["thresholds"] == th]["FAR"].iloc[0]
    gr2_fpr = 1 - gr2[gr2["thresholds"] == th]["FAR"].iloc[0]

    gr_dist = [abs(gr1_tpr - gr2_tpr), abs(gr1_fpr - gr2_fpr)]

    return dict(zip(["equalized_odds_TPR", "equalized_odds_FPR"], gr_dist))


@_fair_plot
def fair_FDR(th, gr1_name, gr1, gr2_name, gr2, **kwargs):
    # FDR: https://arxiv.org/pdf/2011.02395.pdf
    ds_far = abs(gr1[gr1["thresholds"] == th]["FAR"].iloc[0] - gr2[gr2["thresholds"] == th]["FAR"].iloc[0])
    ds_frr = abs(gr1[gr1["thresholds"] == th]["FRR"].iloc[0] - gr2[gr2["thresholds"] == th]["FRR"].iloc[0])

    alpha = kwargs.pop('alpha')

    fdr = 1 - (alpha * ds_far + (1 - alpha) * ds_frr)

    # 1 - fdr because in the paper the fairness is achieved if fdr >= 1 - epsilon
    return {"FDR": 1 - fdr}


def plot_metrics_on_eer_far_1percent(roc_metrics,
                                     data_labels_dfs,
                                     binary_cols,
                                     binary_labels,
                                     metadata_plot_path,
                                     fairness_metrics="all"):
    if fairness_metrics == "all":
        fairness_metrics = [(name, fair_func) for name, fair_func in inspect.getmembers(sys.modules[__name__])
                            if inspect.isfunction(fair_func) and name.startswith('fair_')]
    elif fairness_metrics is not None:
        fairness_metrics = [(name, fair_func) for name, fair_func in inspect.getmembers(sys.modules[__name__])
                            if inspect.isfunction(fair_func) and name.startswith('fair_') and name in fairness_metrics]
    else:
        fairness_metrics = []

    for binary_col, (bin_label1, bin_label2) in zip(binary_cols, binary_labels):
        th_eer = roc_metrics["total"].thresholds[roc_metrics["total"].min_index]
        far_1percent = roc_metrics["total"].FAR[(np.abs(roc_metrics["total"].FAR - 0.01)).argmin()]
        th_far_1p = roc_metrics["total"].thresholds[roc_metrics["total"].FAR == far_1percent]

        ths_groups_EER = [roc_metrics[bin_label1].thresholds[roc_metrics[bin_label1].thresholds == th_eer][0],
                          roc_metrics[bin_label2].thresholds[roc_metrics[bin_label2].thresholds == th_eer][0]]
        FAR_groups_EER = [roc_metrics[bin_label1].FAR[roc_metrics[bin_label1].thresholds == ths_groups_EER[0]][0],
                          roc_metrics[bin_label2].FAR[roc_metrics[bin_label2].thresholds == ths_groups_EER[1]][0]]
        FRR_groups_EER = [roc_metrics[bin_label1].FRR[roc_metrics[bin_label1].thresholds == ths_groups_EER[0]][0],
                          roc_metrics[bin_label2].FRR[roc_metrics[bin_label2].thresholds == ths_groups_EER[1]][0]]
        attributes = [bin_label1] + [bin_label2]

        df_groups_EER = pd.DataFrame(
            zip(FAR_groups_EER, FRR_groups_EER, ths_groups_EER, attributes),
            columns=["FAR", "FRR", "thresholds", "group"]
        )

        ths_groups_FAR_1p = [roc_metrics[bin_label1].thresholds[roc_metrics[bin_label1].thresholds == th_far_1p][0],
                             roc_metrics[bin_label2].thresholds[roc_metrics[bin_label2].thresholds == th_far_1p][0]]
        FAR_groups_FAR_1p = [roc_metrics[bin_label1].FAR[roc_metrics[bin_label1].thresholds == ths_groups_EER[0]][0],
                             roc_metrics[bin_label2].FAR[roc_metrics[bin_label2].thresholds == ths_groups_EER[1]][0]]
        FRR_groups_FAR_1p = [roc_metrics[bin_label1].FRR[roc_metrics[bin_label1].thresholds == ths_groups_EER[0]][0],
                             roc_metrics[bin_label2].FRR[roc_metrics[bin_label2].thresholds == ths_groups_EER[1]][0]]

        df_groups_FAR_1p = pd.DataFrame(
            zip(FAR_groups_FAR_1p, FRR_groups_FAR_1p, ths_groups_FAR_1p, attributes),
            columns=["FAR", "FRR", "thresholds", "group"]
        )

        fair_eer_df = pd.DataFrame([df_groups_EER["thresholds"][0]], columns=["thresholds"])
        fair_far_1p_df = pd.DataFrame([df_groups_FAR_1p["thresholds"][0]], columns=["thresholds"])

        for i, (fair_metric_name, fair_metric) in enumerate(fairness_metrics):
            fair_scores: pd.DataFrame = fair_metric(df_groups_EER, labels_df=data_labels_dfs, alpha=0.5)
            fair_scores = fair_scores.dropna()
            fair_eer_df = fair_eer_df.join(fair_scores.set_index("thresholds"), on="thresholds")

            fair_scores: pd.DataFrame = fair_metric(df_groups_FAR_1p, labels_df=data_labels_dfs, alpha=0.5)
            fair_scores = fair_scores.dropna()
            fair_far_1p_df = fair_far_1p_df.join(fair_scores.set_index("thresholds"), on="thresholds")

        test_type = metadata_plot_path[4][-1]
        train_type = metadata_plot_path[3][-1]

        save_metadata_plot = metadata_plot_path[:]

        save_metadata_plot[4] = re.sub(r'test.', f'test-{map_test[test_type]}', save_metadata_plot[4])
        save_metadata_plot[3] = re.sub(r'train.', f'train-{map_train[train_type]}', save_metadata_plot[3])

        fair_eer_df.to_csv(os.path.join(DEST_PLOTS_RQ1_PATH, f"eer {' '.join(save_metadata_plot)} {bin_label1}_{bin_label2}.csv"))
        fair_far_1p_df.to_csv(os.path.join(DEST_PLOTS_RQ1_PATH, f"far_1p {' '.join(save_metadata_plot)} {bin_label1}_{bin_label2}.csv"))


def plot_metrics_by_th(roc_metrics,
                       data_labels_dfs,
                       binary_cols,
                       binary_labels,
                       metadata_plot_path,
                       fairness_metrics="all",
                       eps=1e-2,
                       far_frr=True):
    if fairness_metrics == "all":
        fairness_metrics = [(name, fair_func) for name, fair_func in inspect.getmembers(sys.modules[__name__])
                            if inspect.isfunction(fair_func) and name.startswith('fair_')]
    elif fairness_metrics is not None:
        fairness_metrics = [(name, fair_func) for name, fair_func in inspect.getmembers(sys.modules[__name__])
                            if inspect.isfunction(fair_func) and name.startswith('fair_') and name in fairness_metrics]
    else:
        fairness_metrics = []

    for binary_col, (bin_label1, bin_label2) in zip(binary_cols, binary_labels):
        # The threshold with value 2.0 is not considered to "zoom" better the range 0.0-1.0

        FAR_groups = np.concatenate((roc_metrics[bin_label1].FAR[1:], roc_metrics[bin_label2].FAR[1:]))
        FRR_groups = np.concatenate((roc_metrics[bin_label1].FRR[1:], roc_metrics[bin_label2].FRR[1:]))
        ths_groups = np.concatenate((roc_metrics[bin_label1].thresholds[1:], roc_metrics[bin_label2].thresholds[1:]))
        attributes = [bin_label1] * len(roc_metrics[bin_label1].thresholds[1:]) + \
                     [bin_label2] * len(roc_metrics[bin_label2].thresholds[1:])

        FAR = roc_metrics["total"].FAR[1:]
        FRR = roc_metrics["total"].FRR[1:]
        ths = roc_metrics["total"].thresholds[1:]

        df_plot = pd.DataFrame(
            zip(FAR, FRR, ths),
            columns=["FAR", "FRR", "thresholds"]
        )
        df_groups = pd.DataFrame(
            zip(FAR_groups, FRR_groups, ths_groups, attributes),
            columns=["FAR", "FRR", "thresholds", "group"]
        )
        fig, ax = plt.subplots(1, len(fairness_metrics), figsize=(18, 5))

        for i, (fair_metric_name, fair_metric) in enumerate(fairness_metrics):
            if far_frr:
                """
                sns.lineplot(x="thresholds", y="FAR", data=df_plot, ax=ax[i], marker='v', hue="group",
                             palette=["blue", "pink"], legend=False, zorder=1)
                sns.lineplot(x="thresholds", y="FRR", data=df_plot, ax=ax[i], marker=".", hue="group",
                             palette=["blue", "pink"], legend=False, zorder=1)
                plot_labels.extend([f'FAR {bin_label1}', f'FAR {bin_label2}', f'FRR {bin_label1}', f'FRR {bin_label2}'])
                """
                sns.lineplot(x="thresholds", y="FAR", label="FAR", data=df_plot, ax=ax[i], color="blue")
                sns.lineplot(x="thresholds", y="FRR", label="FRR", data=df_plot, ax=ax[i], color="red")

            fair_scores: pd.DataFrame = fair_metric(df_groups, labels_df=data_labels_dfs, alpha=0.5)
            fair_scores = fair_scores.dropna()

            # fair_labels = []
            fair_metrics_colors_iter = iter(['gold', 'green', 'aqua'])
            for col in fair_scores.columns:
                if col != "thresholds":
                    sns.lineplot(x="thresholds", y=col, label=col, data=fair_scores, ax=ax[i], color=next(fair_metrics_colors_iter))
                    # fair_labels.append(col.replace('_', ' '))

            # plot_labels.extend(fair_labels)
            ax[i].legend(ncol=2, prop={'size': 6})

            ax[i].set_ylim(top=1.0, bottom=-0.05)
            ax[i].set_xlim(left=fair_scores["thresholds"].min(), right=fair_scores["thresholds"].max())

            # plot epsilon
            xlim = ax[i].get_xlim()
            # ax[i].plot(xlim, [eps, eps], color='red', zorder=3)

            ax[i].minorticks_on()
            ax[i].tick_params(axis='x', which='minor', bottom=False)
            ax[i].yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
            ax[i].grid(ls=':')

            ax[i].set_xlabel("Thresholds")
            ax[i].set_ylabel("")

            title= fair_metric_name.replace('fair_', '').replace('_', ' ').title() \
                if '_' in fair_metric_name.replace('fair_', '') else fair_metric_name.replace('fair_', '')
            ax[i].set_title(f"{title}")

            test_type = metadata_plot_path[4][-1]
            train_type = metadata_plot_path[3][-1]

            save_metadata_plot = metadata_plot_path[:]

            save_metadata_plot[4] = re.sub(r'test.', f'test-{map_test[test_type]}', save_metadata_plot[4])
            save_metadata_plot[3] = re.sub(r'train.', f'train-{map_train[train_type]}', save_metadata_plot[3])

            """
            fig.suptitle(f"{save_metadata_plot[0]} "
                         f"{save_metadata_plot[4]} "
                         f"{save_metadata_plot[3]} "
                         f"{bin_label1[0].upper()}/{bin_label2[0].upper()}")
            """

            fig.savefig(
                os.path.join(DEST_PLOTS_RQ2_PATH, f"{' '.join(save_metadata_plot)} {bin_label1}_{bin_label2}.png"),
                bbox_inches="tight",
                pad_inches=0
            )

            plt.close(fig)


def calculate_parameters(y, y_score):
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

    far, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    frr = 1 - tpr
    abs_diffs = np.abs(far - frr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((far[min_index], frr[min_index]))
    # return min_index, far, frr, far[min_index], frr[min_index], eer, thresholds[min_index], thresholds
    return ROCMetrics(eer, far, frr, thresholds, min_index)


def load_results(data,
                 res_filename,
                 binary_cols: Sequence[str] = None,
                 binary_labels: Sequence[Sequence[str]] = None):
    """
    Function used to extract from from data
    :param data: contains the results reported in the current .csv file stored from the test
    :param res_filename: is the name of the current .csv used to retrieve all the useful info to describe the result
    :param binary_cols: list of columns of the csv containing sensitive attributes
    :param binary_labels: list of list of binary sensitives attributes to filter csv rows, each sublist must contain
                          attributes of the respective index column, first sublist filter rows of first 'binary_cols'
                          column
    :return: three records: one concerning EER, one FAR, one FRR with all the related information
    """
    binary_cols = ["gender_1", "age_1"] if binary_cols is None else binary_cols
    binary_labels = [["male", "female"], ["young", "old"]] if binary_labels is None else binary_labels

    flat_binary_labels = np.array(binary_labels).flatten()

    sensitive_metrics = dict.fromkeys(flat_binary_labels)
    records = dict.fromkeys(METRICS_NAMES)
    data_labels_dfs = dict.fromkeys(flat_binary_labels)

    # Here are taken the columns containing the expected results (label) and the predicted ones (similarity)
    label = data.loc[:, 'label']
    similarity = data.loc[:, 'simlarity']
    # The data are passed to the function in order to calculate all the results
    total_metrics = calculate_parameters(label, similarity)
    sensitive_metrics["total"] = total_metrics

    # Extraction of similarity scores for each sensitive attribute in `binary_labels`
    for col, labels in zip(binary_cols, binary_labels):
        for lab in labels:
            data_lab = data[data[col] == lab]
            similarity = data_lab.loc[:, "simlarity"]
            cos_sim_labels = data_lab.loc[:, "label"]
            sensitive_metrics[lab] = calculate_parameters(cos_sim_labels, similarity)
            data_labels_dfs[lab] = data_lab

    # CSV Metadata
    info = res_filename.split('#')[0].split('_')
    arch_name = info[0]
    train_model_name = info[1].split('@')[0]
    epoch = info[1].split('@')[1]
    accuracy = info[2]
    accuracy = (accuracy[:2] + '.' + accuracy[-1]).replace('.', ',')
    test_file_name = info[4]

    balancing_levels = list(map(lambda x: x[:2], res_filename.split('#')[1].split('_')))
    loss_bal = f"[{' - '.join(['.'.join(bal_level) for bal_level in balancing_levels])}]"

    base_record = [arch_name, loss_bal, epoch, train_model_name, test_file_name, accuracy]

    plot_metrics_by_th(sensitive_metrics, data_labels_dfs, binary_cols, binary_labels, base_record)
    plot_metrics_on_eer_far_1percent(sensitive_metrics, data_labels_dfs, binary_cols, binary_labels, base_record)

    # Creation of records rows for the dataframe, by means of which the final csvs will be created, one for each
    # `records` entry
    for _metric in METRICS_NAMES:
        if _metric == 'EER':
            metr_records = [getattr(total_metrics, _metric)]
        else:
            metr_records = [getattr(total_metrics, _metric)[total_metrics.min_index]]

        for (bin_label1, bin_label2) in binary_labels:
            metr1 = sensitive_metrics[bin_label1]
            metr2 = sensitive_metrics[bin_label2]

            if _metric == "EER":
                lab1_metr_value = getattr(metr1, _metric)
                lab2_metr_value = getattr(metr2, _metric)
            else:
                lab1_metr_value = getattr(metr1, _metric)[metr1.min_index]
                lab2_metr_value = getattr(metr2, _metric)[metr2.min_index]

            metr_records.append(lab1_metr_value)
            metr_records.append(lab2_metr_value)
            metr_records.append(abs(lab1_metr_value - lab2_metr_value))

        metr_records = list(map(lambda x: f"{x*100:.2f}", metr_records))  # metrics are given in percent format (%)
        records[_metric] = base_record + metr_records

    return records


def create_Experiment_CSV_details(eer,
                                  far,
                                  frr,
                                  dest_path,
                                  binary_out_indexes: Sequence[Sequence[str]] = None,
                                  DS_suffices: Sequence[str] = None):
    """
    Function used to create the .csv metric report considering all the distinct measures taken (so divided per sensitive
    categories)
    :param eer: contains the list of records concerning the calculated EER
    :param far: contains the list of records concerning the calculated FAR
    :param frr: contains the list of records concerning the calculated FRR
    :param dest_path: path where create and store the results
    :param binary_out_indexes: list of list of binary sensitives identifiers for the columns containing data
                               for a certain sensitive group. This list must reflect the parameter `binary_labels`
                               of `load_results` function
    :param DS_suffices: list of suffices to add to the Disparity Score columns.
                        ex: "DS EER M/F" -> "DS EER " + DS_suffices[0], where DS_suffices = ["M/F", ...]
    :return: NONE
    """

    if not os.path.exists(dest_path):
        shutil.rmtree(dest_path)
        os.mkdir(dest_path)

    binary_out_indexes = [["male", "female"], ["young", "old"]] if binary_out_indexes is None else binary_out_indexes
    DS_suffices = [" M/F", " Y/O"] if DS_suffices is None else DS_suffices

    for _metric, data in zip(METRICS_NAMES, [eer, far, frr]):
        metric_columns = [f"{_metric}"]
        for (bin_lab1, bin_lab2), ds_suffix in zip(binary_out_indexes, DS_suffices):
            metric_columns.append(f"{_metric} {bin_lab1}")
            metric_columns.append(f"{_metric} {bin_lab2}")
            metric_columns.append(f"DS {_metric} {ds_suffix}")

        data_columns = RESULTS_BASE_COLUMNS + metric_columns
        pd.DataFrame(data, columns=data_columns).to_csv(os.path.join(dest_path, f"{_metric}.csv"), index=False)
        print(f'> {_metric} CSV GENERATED in \t' + dest_path)


def create_Experiment_CSV_details_totonly(eer, far, frr, dest_path):
    """
    Function used to create the .csv metric report considering just the total metric
    categories)
    :param eer: contains the list of records concerning the calculated EER
    :param far: contains the list of records concerning the calculated FAR
    :param frr: contains the list of records concerning the calculated FRR
    :param dest_path: path where create and store the results
    :return: NONE
    """
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    else:
        shutil.rmtree(dest_path)
        os.mkdir(dest_path)

    for metric, metric_val in zip(['EER', 'FAR', 'FRR'], [eer, far, frr]):
        df = pd.DataFrame(metric_val[:, :(len(RESULTS_BASE_COLUMNS) + 1)], columns=RESULTS_BASE_COLUMNS + [metric])
        df.to_csv(os.path.join(dest_path, metric + '.csv'), index=False)
        print(f'> {metric} CSV GENERATED in \t' + dest_path)


def merge_plots_same_test():
    plots = [f.path for f in os.scandir(DEST_PLOTS_RQ2_PATH) if f.name.endswith('.png')]

    df_plots = pd.DataFrame(zip(plots, [os.path.basename(pl).replace("[0.0 - 1.0] ", '') for pl in plots]))
    df_plots[["Architecture", "Epoch", "Train File", "Test File", "Accuracy", "Attributes+ext"]] = df_plots[1].str.split(' ', expand=True)
    df_plots["Train File mod"] = [re.sub(r"UB|NB|SB", '', f) for f in df_plots["Train File"]]
    df_plots["Train Type"] = [re.search(r"UB|NB|SB", f)[0] for f in df_plots["Train File"]]

    train_sorter = dict(zip(["NB", "UB", "SB"], range(3)))

    for n, gr in df_plots.groupby(["Architecture", "Test File", "Train File mod", "Attributes+ext"]):
        gr = gr.assign(**{"Train File Order": lambda df: df["Train Type"].map(train_sorter).to_numpy()})
        gr = gr.sort_values("Train File Order")
        gr.drop("Train File Order", axis=1, inplace=True)

        # get only the final number from train string except the first so at the end obtain e.g. ...TrainNB-UB-TB
        train_files_gr_str = [f.split('-')[-1] if i > 0 else f for i, f in enumerate(gr_comb["Train File"].to_list())]

        imgs = [plt.imread(f)[:, :, :3] for f in gr[0]]

        # for a vertical stacking it is simple: use vstack
        imgs_comb = np.vstack([img for img in imgs])

        # image is saved removing epoch and accuracy
        plt.imsave(
            os.path.join(DEST_PLOTS_RQ1_PATH,
                         re.sub(
                             r'(?<=test-Age) ....|(?<=test-Gender) ....|(?<=test-Random) ....',
                             '',
                             re.sub(
                                r'(?<=\[0.0 - 1.0]) ..',
                                '',
                                gr_comb[0].to_list()[0].replace(train_files_gr_str[0], '-'.join(train_files_gr_str))
                             )
                         )),
            imgs_comb[::-1]
        )

    for plot_path in df_plots[0]:
        os.remove(plot_path)


def merge_same_test_barplot_from_csvs():
    csvs = [f.path for f in os.scandir(DEST_PLOTS_RQ1_PATH) if f.name.endswith('.csv')]

    df_csvs = pd.DataFrame(zip(csvs, [os.path.basename(f).replace("[0.0 - 1.0] ", '') for f in csvs]))
    df_csvs[["Type", "Architecture", "Epoch", "Train File", "Test File", "Accuracy", "Attributes+ext"]] = df_csvs[1].str.split(' ', expand=True)
    df_csvs["Train File mod"] = [re.sub(r"-UB|-NB|-SB", '', f) for f in df_csvs["Train File"]]
    df_csvs["Train Balance"] = [re.search(r"UB|NB|SB", f)[0] for f in df_csvs["Train File"]]

    train_sorter = dict(zip(["NB", "UB", "SB"], range(3)))

    n_fairness_metrics = 4

    for n, gr in df_csvs.groupby(["Type", "Architecture", "Test File", "Train File mod", "Attributes+ext"]):
        gr = gr.assign(**{"Train File Order": lambda _df: _df["Train Balance"].map(train_sorter).to_numpy()})
        gr = gr.sort_values("Train File Order")
        gr.drop("Train File Order", axis=1, inplace=True)

        fig, ax = plt.subplots(1, n_fairness_metrics, figsize=(18, 5))

        train_dfs = []
        for csv_file, train_type in gr[[0, "Train Balance"]].values:
            df = pd.read_csv(csv_file, index_col=0)
            df["Train Balance"] = [train_type]
            train_dfs.append(df)

        df = pd.concat(train_dfs)

        for n_metr, metric in enumerate(["FDR", "demographic_parity", "equal_opportunity", "equalized_odds"]):
            if metric == "equalized_odds":
                eq_odds_data = []
                for tt, tt_gr in df.groupby("Train Balance"):
                    eq_odds_data.append([tt, tt_gr.loc[0, metric + '_TPR'], "TPR"])
                    eq_odds_data.append([tt, tt_gr.loc[0, metric + '_FPR'], "FPR"])

                eq_odds_df = pd.DataFrame(eq_odds_data, columns=["Train Balance", "value", "Metric"])

                sns.barplot(x="Train Balance", y="value", hue="Metric", data=eq_odds_df, ax=ax[n_metr])
                ax[n_metr].legend(ncol=2, prop={'size': 8})
            else:
                sns.barplot(x="Train Balance", y=metric, data=df, ax=ax[n_metr])

            ax[n_metr].set_title(metric.replace('_', ' ').title() if '_' in metric else metric)
            ax[n_metr].set_ylabel("")

        first_row = gr.iloc[0]
        arch = first_row["Architecture"]
        train_f = first_row["Train File"].replace('-', ' ').rsplit(' ', 1)[0]
        test_f = first_row["Test File"].replace('-', ' ')
        th_type = first_row["Type"].upper()
        attrs = ' '.join(first_row["Attributes+ext"].replace('.csv', '').split('_')).title()

        # fig.suptitle(f"{th_type.replace('_', ' ').replace('P', '%')} {arch} {train_f} {test_f} {attrs}")
        fig.savefig(
            os.path.join(DEST_PLOTS_RQ1_PATH, f"{th_type} {arch} {train_f} {test_f} {attrs}.png"),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(fig)

    for csv_file in df_csvs[0]:
        os.remove(csv_file)


def main():
    parser = argparse.ArgumentParser(description='Operations utils for test results elaboration')

    parser.add_argument('--result_path', dest='result_path', default=RESULT_PATH, type=str,
                        action='store', help='Base path for results')
    parser.add_argument('--dest_folder', dest='dest_folder', default=DEST_PATH, type=str,
                        action='store', help='Base path for destination folder results')

    args = parser.parse_args()

    eer_to_load, far_to_load, frr_to_load = [], [], []
    binary_attributes = [["male", "female"], ["young", "old"]]

    binary_csv_indexes = [["M", "F"], ["Y", "O"]]
    DS_suffices = ["M/F", "Y/O"]

    shutil.rmtree(DEST_PLOTS_RQ1_PATH)
    shutil.rmtree(DEST_PLOTS_RQ2_PATH)
    os.mkdir(DEST_PLOTS_RQ1_PATH)
    os.mkdir(DEST_PLOTS_RQ2_PATH)

    print('>Start Scanning Results folder')
    # The result folder is scanned
    for res in os.listdir(args.result_path):
        # Only the folders are taken in consideration
        if not os.path.isdir((os.path.join(args.result_path, res))):
            print('>Elaborating ---> \t' + res.replace('_meta_metadata_test_', ''))

            # the current result .csv is read
            res_csv = pd.read_csv(os.path.join(args.result_path, res))

            # the main measures are calculated from the results
            records = load_results(res_csv, res, binary_cols=["gender_1", "age_1"], binary_labels=binary_attributes)

            # the returned records above are added in the corresponding lists
            eer_to_load.append(records["EER"])
            far_to_load.append(records["FAR"])
            frr_to_load.append(records["FRR"])

    eer_to_load, far_to_load, frr_to_load = np.array(eer_to_load), np.array(far_to_load), np.array(frr_to_load)
    create_Experiment_CSV_details(eer_to_load,
                                  far_to_load,
                                  frr_to_load,
                                  args.dest_folder,
                                  binary_out_indexes=binary_csv_indexes,
                                  DS_suffices=DS_suffices)

    # merge_plots_same_test()
    merge_same_test_barplot_from_csvs()

    # create_Experiment_CSV_details_totEERonly(eer_to_load, far_to_load, frr_to_load, args.dest_folder)

    print('\n\n> {} RESULTS ELABORATED!'.format(len(eer_to_load)))


if __name__ == '__main__':
    main()
