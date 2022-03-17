import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.exceptions import NotFittedError

from inputer import DataInputer


class CausalClassifier:
    # set RF number of trees
    NUM_ESTIMATORS = 350

    # set iteration range for RF depth
    MAX_DEPTH_MIN = 50
    MAX_DEPTH_MAX = 250
    MAX_DEPTH_STEP = 10

    # set iteration range for LR max_iter param
    MIN_ITER = 10
    MAX_ITER = 200
    STEP_ITER = 10

    # set iteration range for LR C param
    C_MIN = 1
    C_STEP = 1
    C_GRID = C_MIN * np.power(10, np.arange(0, C_STEP, dtype=float))

    # set iteration range for LR L1 ratio param
    L1_RATIO_MIN = 0
    L1_RATIO_STEP = 0.1
    L1_RATIO_MAX = 1
    L1_RATIO_GRID = np.arange(L1_RATIO_MIN, L1_RATIO_MAX + L1_RATIO_STEP, L1_RATIO_STEP)

    def __init__(self, dataset: pd.DataFrame, target_col_name: str, use_grid: bool = False,
                 test_size: float = 0.2, classifier: str = "RF", model_path: str = None):
        # self._imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self._save_path = model_path
        self._target_col = dataset[target_col_name].astype(str)
        self._target_col_name = target_col_name
        self._dataset = dataset.drop(columns=target_col_name)
        self._imputer = DataInputer(self.dataset)
        self._test_size = test_size
        self.__cc = classifier
        if classifier == "RF":
            self._classifier = RandomForestClassifier()
        elif classifier == "LR":
            self._classifier = LogisticRegression()
        else:
            raise ValueError("Causal Classifier alias not supported!")
        self._best_gs_res = None

        if test_size:
            self._train_set_X, self._test_set_X = train_test_split(self.dataset,
                                                                   test_size=self._test_size,
                                                                   random_state=27)
            self._train_set_X = self._imputer.transform(self._train_set_X)
            self._test_set_X = self._imputer.transform(self._test_set_X)
            self._train_set_Y, self._test_set_Y = train_test_split(self.dataset[self._target_col_name],
                                                                   test_size=self._test_size,
                                                                   random_state=27)
        else:
            self._train_set_X = self._imputer.transform(self.dataset)
            self._train_set_Y = self._target_col
            self._test_set_X, self._test_set_Y = None, None
        if use_grid:
            self._params_grid = self.generate_grid()
            self._gs_list: List[GridSearchCV] = [GridSearchCV(estimator=self._classifier,
                                                              param_grid=param_grid,
                                                              n_jobs=-1,
                                                              scoring=["balanced_accuracy", "roc_auc", "f1_weighted"],
                                                              refit="f1_weighted",
                                                              return_train_score=True)
                                                 for param_grid in self._params_grid]
            self._classifier = None
        else:
            self._params_grid, self._gs_list = None, None

    @property
    def dataset(self):
        return self._dataset

    @property
    def classifier(self):
        return self._classifier

    @property
    def train_set(self):
        return self._train_set_X, self._train_set_Y

    @property
    def test_set(self):
        return self._test_set_X, self._test_set_Y

    def fit(self):
        if not self._gs_list:
            self.classifier.fit(self._train_set_X, self._train_set_Y)
            if self.__cc == "RF":
                return self.classifier.feature_importances_
            elif self.__cc == "LR":
                return self.classifier.coef_, self.classifier.intercept_
            else:
                raise ValueError("Wrong Estimator!")
        else:
            gs_res = []
            for gs in self._gs_list:
                gs.fit(self._train_set_X, self._train_set_Y)
                gs_res.append({"classifier": gs.best_estimator_,
                               "params": gs.best_params_,
                               "score": gs.best_score_,
                               "gs": gs})
            gs_res = sorted(gs_res, key=lambda x: x["score"])
            self._best_gs_res = gs_res[0]
            y_pred = gs_res[0]["classifier"].predict(self._train_set_X)
            metrics = {"f1_score": f1_score(self._train_set_Y, y_pred, average="weighted"),
                       "roc_auc_score": roc_auc_score(self._train_set_Y, y_pred, average="weighted"),
                       "balanced_accuracy_score": balanced_accuracy_score(self._train_set_Y, y_pred)}
            if self._save_path:
                with open(self._save_path, 'wb') as model_file:
                    pickle.dump(gs_res[0]["classifier"], model_file)
            if self.__cc == "RF":
                return gs_res[0]["classifier"].feature_importances_, metrics
            elif self.__cc == "LR":
                return gs_res[0]["classifier"].coef_, gs_res[0]["classifier"].intercept_, metrics
            else:
                raise ValueError("Wrong Estimator!")

    @classmethod
    def generate_grid(cls, classifier: str = "RF"):
        if classifier == "RF":
            params_grid = [{"n_estimators": [cls.NUM_ESTIMATORS],
                            "criterion": ["gini", "entropy"],
                            "max_depth": range(cls.MAX_DEPTH_MIN, cls.MAX_DEPTH_MAX + cls.MAX_DEPTH_STEP,
                                               cls.MAX_DEPTH_STEP),
                            "max_features": ["auto", "sqrt", "log2"],
                            "bootstrap": [True],
                            "oob_score": [True, False]},
                           {"n_estimators": [cls.NUM_ESTIMATORS],
                            "criterion": ["gini", "entropy"],
                            "max_depth": range(cls.MAX_DEPTH_MIN, cls.MAX_DEPTH_MAX + cls.MAX_DEPTH_STEP,
                                               cls.MAX_DEPTH_STEP),
                            "max_features": ["auto", "sqrt", "log2"],
                            "bootstrap": [False],
                            "oob_score": [False]}]

        elif classifier == "LR":
            params_grid = [{"penalty": ["l2"],
                            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                            "C": cls.C_GRID,
                            "class_weight": ["balanced"],
                            "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                            "multi_class": ["auto", "ovr", "multinomial"]},
                           {"penalty": ["none"],
                            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                            "class_weight": ["balanced"],
                            "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                            "multi_class": ["auto", "ovr", "multinomial"]},
                           {"penalty": ["elasticnet"],
                            "solver": ["saga"],
                            "C": cls.C_GRID,
                            "l1_ratio": cls.L1_RATIO_GRID,
                            "class_weight": ["balanced"],
                            "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                            "multi_class": ["auto", "ovr", "multinomial"]},
                           {"penalty": ["l1"],
                            "solver": ["liblinear"],
                            "C": cls.C_GRID,
                            "class_weight": ["balanced"],
                            "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                            "multi_class": ["auto", "ovr"]},
                           {"penalty": ["l1"],
                            "solver": ["saga"],
                            "C": cls.C_GRID,
                            "class_weight": ["balanced"],
                            "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                            "multi_class": ["auto", "ovr", "multinomial"]}]
        else:
            params_grid = None

        return params_grid

    def is_fitted(self):
        try:
            self._classifier.predict(some_test_data)
            return True
        except NotFittedError:
            return False

    def importance_barplot(self, sns_kw=None):
        sns_kw = sns_kw or {}
        if self.is_fitted():
            if args.causal_classifier == "RF":
                feat_names = self.train_set[0].columns.to_numpy().astype(str)
                feat_importance = self._best_gs_res["classifier"].feature_importances_

                df = pd.DataFrame.from_dict(dict(zip(feat_names, feat_importance)), orient='index')
                df = df.sort_values(0, ascending=False).T

                sns.barplot(data=df, **sns_kw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Logistic Regression Classifier training pipeline')

    parser.add_argument('--af_path', dest='audio_feature_path', default='test_audio_features.pkl',
                        type=str, action='store', help='Audio features pickle file path')

    parser.add_argument('--al_path', dest='audio_label_path',
                        default='far_data__resnet34vox_English-Spanish-train1@15_920_08032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.pkl',
                        type=str, action='store', help='Audio labels pickle file path')

    cc_action = parser.add_argument('--cc', dest='causal_classifier', default='RF', choices=["RF", "LR"], type=str,
                                    action='store', help='Causal classifier alias')

    parser.add_argument('--model_save_path', dest='model_save_path', default='causal_classifier.model',
                        type=str, action='store', help='Save path to store the causal classifier model')

    parser.add_argument('--metr_feats_sfolder', dest='metrics_features_save_folderpath',
                        default='metrics_features_causal_classifier', type=str, action='store',
                        help='Save folderpath to store the metrics and the feature importance')

    args = parser.parse_args()

    print("Composing Dataset...")
    with open(args.audio_feature_path, "rb") as taf_pkl:
        test_audio_features = pickle.load(taf_pkl)
        with open(args.audio_label_path, "rb") as fsl_pkl:
            far_labels = pickle.load(fsl_pkl)
        audio_features_dataset = pd.DataFrame()
        for k, v in test_audio_features.items():
            elements = k.split("/")
            lan, user = elements[0], elements[1]
            record_to_add = {}
            for ik, iv in v.items():
                if ik not in ["gender", "mood", "age"]:
                    record_to_add[ik] = iv
            record_to_add["gender"] = far_labels[lan][user][1].split()[0]
            record_to_add["age"] = far_labels[lan][user][1].split()[1]
            record_to_add["label"] = 0 if far_labels[lan][user][0] > 0 else 1
            audio_features_dataset = audio_features_dataset.append(record_to_add, ignore_index=True)
    audio_features_dataset.loc[np.isinf(audio_features_dataset['signaltonoise_dB']), 'signaltonoise_dB'] = -1
    print("Dataset ready!")

    print("Setting up Causal Classifier...")
    causal_classifier = CausalClassifier(dataset=audio_features_dataset,
                                         target_col_name="label",
                                         use_grid=True,
                                         test_size=0,
                                         classifier=args.causal_classifier,
                                         model_path=args.model_save_path)
    print("Causal Classifier set up")

    metadata_filename = f"{args.causal_classifier}_{os.path.splitext(os.path.basename(args.audio_label_path))[0]}"

    if not os.path.exists(args.metrics_features_save_folderpath):
        os.makedirs(args.metrics_features_save_folderpath)

    features_names = causal_classifier.train_set[0].columns.to_numpy().astype(str)
    np.save(
        file=os.path.join(args.metrics_features_save_folderpath, f"features_names_{metadata_filename}.npy"),
        arr=features_names
    )

    print("Start fitting...")
    if args.causal_classifier == "RF":
        feature_importance, metrics = causal_classifier.fit()
        print("Training done!")
        print("Saving Causal Classifier weights...")
        np.save(
            file=os.path.join(args.metrics_features_save_folderpath, f"feature_importance_{metadata_filename}.npy"),
            arr=feature_importance
        )

        df = pd.DataFrame.from_dict(dict(zip(features_names, feature_importance)), orient='index')
        df.sort_values(0, ascending=False).T.to_csv(os.path.join(
            args.metrics_features_save_folderpath,
            f'{metadata_filename}.csv'
        ), index=False)

        causal_classifier.importance_barplot()
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(os.path.join(args.metrics_features_save_folderpath, f"barplot#{metadata_filename}.json"))
    elif args.causal_classifier == "LR":
        coef, intercept, metrics = causal_classifier.fit()
        print("Training done!")
        print("Saving Causal Classifier weights...")

        coef_path = f"coef_{metadata_filename}.npy"

        intercept_path = f"intercept_{metadata_filename}.npy"
        np.save(file=os.path.join(args.metrics_features_save_folderpath, coef_path), arr=coef)
        np.save(file=os.path.join(args.metrics_features_save_folderpath, intercept_path), arr=intercept)
    else:
        raise ValueError(f"`{args.causal_classifier}` is not a supported classifier. Select one of {cc_action.choices}")

    with open(os.path.join(args.metrics_features_save_folderpath, f"metrics_{metadata_filename}.json"), "w")\
            as metrics_file:
        json.dump(metrics, metrics_file)

    print("Causal Classifier weights stored!")
