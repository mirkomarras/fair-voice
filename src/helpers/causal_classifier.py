import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from inputer import DataInputer


class CausalClassifier:
    # set RF number of trees
    NUM_ESTIMATORS = 350

    # set iteration range for RF depth
    MAX_DEPTH_MIN = 50
    MAX_DEPTH_MAX = 200
    MAX_DEPTH_STEP = 10

    # set iteration range for LR max_iter param
    MIN_ITER = 0
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
                 test_size: float = 0.2, classifier: str = "RF"):
        # self._imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
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
                                                              n_jobs=-1) for param_grid in self._params_grid]
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
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

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
            if self.__cc == "RF":
                return gs_res[0]["classifier"].feature_importances_
            elif self.__cc == "LR":
                return gs_res[0]["classifier"].coef_, gs_res[0]["classifier"].intercept_
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Logistic Regression Classifier training pipeline')

    parser.add_argument('--af_path', dest='audio_feature_path', default='test_audio_features.pkl',
                        type=str, action='store', help='Audio features pickle file path')

    parser.add_argument('--al_path', dest='audio_label_path',
                        default='far_data__resnet34vox_English-Spanish-train1@15_920_08032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.pkl',
                        type=str, action='store', help='Audio labels pickle file path')

    parser.add_argument('--cc', dest='causal_classifier', default='RF', type=str, action='store',
                        help='Causal classifier alias')

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
    print("Dataset ready!")

    print("Setting up Causal Classifier...")
    causal_classifier = CausalClassifier(dataset=audio_features_dataset,
                                         target_col_name="label",
                                         use_grid=True,
                                         test_size=0,
                                         classifier=args.causal_classifier)
    print("Causal Classifier set up")

    print("Start fitting...")
    if args.causal_classifier == "RF":
        feature_importance = causal_classifier.fit()
        print("Training done!")
        print("Saving Causal Classifier weights...")
        with open(f"coef_{args.causal_classifier}.npy", "wb") as fi_file:
            np.save(file=fi_file, arr=feature_importance)
    elif args.causal_classifier == "LR":
        coef, intercept = causal_classifier.fit()
        print("Training done!")
        print("Saving Causal Classifier weights...")
        with open(f"coef_{args.causal_classifier}.npy", "wb") as coef_file:
            np.save(file=coef_file, arr=coef)
        with open(f"intercept_{args.causal_classifier}.npy", "wb") as intercept_file:
            np.save(file=intercept_file, arr=intercept)

    print("Causal Classifier weights stored!")
