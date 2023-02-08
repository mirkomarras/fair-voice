import os
import argparse
import pickle

import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Classifier analysis')

    parser.add_argument('--model', dest='model',
                        default='/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10/causal_classifier_RF.model',
                        type=str, action='store', help='Causal classifier')

    parser.add_argument('--train_set_x', dest='train_set_x',
                        default='/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/classification/train_set_x.csv',
                        type=str, action='store', help='Train set X')

    parser.add_argument('--train_set_y', dest='train_set_y',
                        default='/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/classification/train_set_y.csv',
                        type=str, action='store', help='Train set Y')

    parser.add_argument('--out_path', dest='out_path',
                        default='/home/meddameloni/dl-fair-voice/exp/counterfactual_fairness/counterfactual_analysis',
                        type=str, action='store', help='Out Path')

    parser.add_argument('--disparity', dest='disparity', action='store_true',
                        help='If true kde will be created on disparity between real and counterfactual probabilities')

    args = parser.parse_args()

    with open(args.model, 'rb') as classifier_file:
        model = pickle.load(classifier_file)

    net = args.model.split('__')[1].split('_')[0]
    cc = os.path.basename(args.model).split('.')[0].split('_')[-1]

    train_set_x, train_set_y = pd.read_csv(args.train_set_x), pd.read_csv(args.train_set_y)

    categorical_features = {
        'age': ['age_gender_language_x0_younger', 'age_gender_language_x0_older'],
        'gender': ['age_gender_language_x1_female', 'age_gender_language_x1_male'],
        'lang': ['age_gender_language_x2_English', 'age_gender_language_x2_Spanish']
    }

    cat_order = ["age", "gender", "lang"]

    if hasattr(model, "predict_proba"):
        probabilities = np.full((len(train_set_x), len(categorical_features) + 1), np.nan, dtype=np.float_)
    else:
        probabilities = None
    predictions = np.full((len(train_set_x), len(categorical_features) + 1), np.nan, dtype=np.float_)
    for i, row in tqdm.tqdm(train_set_x.iterrows(), desc="Extracting probabilities"):
        if probabilities is not None:
            real_prob = model.predict_proba([row])[0, 0]
        real_pred = int(float(model.predict([row])[0]))

        fake_prob = []
        fake_pred = []
        for k, v in zip(cat_order, list(categorical_features.values())):
            _row = row.copy()
            for sens_attr in v:
                _row[sens_attr] = abs(_row[sens_attr] - 1)  # swap 0 to 1 and vice versa because of one-hot encoding
            if probabilities is not None:
                fake_prob.append(model.predict_proba([_row])[0, 0])
            fake_pred.append(int(float(model.predict([_row])[0])))

        if probabilities is not None:
            probabilities[i] = [real_prob, *fake_prob]
        predictions[i] = [real_pred, *fake_pred]

    if probabilities is not None:
        if args.disparity:
            data_proba = [[], []]
        else:
            data_proba = np.stack([
                probabilities.reshape((probabilities.shape[0] * (len(cat_order) + 1),)),
                np.concatenate([[cat] * probabilities.shape[0] for cat in ["orig", *cat_order]])
            ])
    data_pred = dict.fromkeys(cat_order)
    for i, cat in enumerate(cat_order):
        if probabilities is not None and args.disparity:
            data_proba[0].extend(np.abs(probabilities[:, 0] - probabilities[:, i + 1]).tolist())
            data_proba[1].extend([cat] * probabilities.shape[0])
        data_pred[cat] = pd.crosstab(
            pd.Series(predictions[:, 0], name='Actual'),
            pd.Series(predictions[:, i + 1], name='Predicted')
        )

    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    plt.rc('legend', fontsize=18, title_fontsize=18)
    # Probabilities
    if probabilities is not None:
        data_proba = pd.DataFrame(data_proba, index=["value", "Sensitive Flip"]).T
        data_proba["value"] = data_proba["value"].astype(float)
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.kdeplot(x="value", data=data_proba, hue="Sensitive Flip", ax=ax)
        plt.xlabel("")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid('y', ls=':')
        ax.set_ylim((0, 0.69))
        ax.set_xlim((-0.39, 1.35))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_path, f'{net}_{cc}_kde_plot_hue_all.png'), bbox_inches='tight', pad_inches=0, dpi=250)
        plt.close()

    # Matrix Confusion
    for k, v in data_pred.items():
        sns.heatmap(data=v, annot=True, linewidths=.5, cmap="viridis", fmt="d")
        plt.suptitle(f"{net} {k}")
        plt.savefig(os.path.join(args.out_path, f"{net}_{cc}_confusion_{k}.png"))
        plt.close()
