from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from pandas import DataFrame
import numpy as np


class CausalLRClassifier:
    MIN_ITER = 0
    MAX_ITER = 100
    STEP_ITER = 1

    C_MIN = 1
    C_STEP = 1
    C_GRID = C_MIN * np.power(10, np.arange(0, C_STEP, dtype=float))

    L1_RATIO_MIN = 0
    L1_RATIO_STEP = 0.1
    L1_RATIO_MAX = 1
    L1_RATIO_GRID = np.arange(L1_RATIO_MIN, L1_RATIO_MAX + L1_RATIO_STEP, L1_RATIO_STEP)

    def __init__(self, dataset: Dataframe, target_col_name: str, use_grid: bool = False, test_size: float = 0.2):
        self._imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self._test_size = test_size
        self._target_col_name = target_col_name
        self._dataset = self._imputer.fit_transform(dataset)
        if test_size:
            self._train_set_X, self._test_set_X = train_test_split(self.dataset.drop(columns=self._target_col_name),
                                                                   test_size=self._test_size,
                                                                   random_state=27)
            self._train_set_Y, self._test_set_Y = train_test_split(self.dataset[self._target_col_name],
                                                                   test_size=self._test_size,
                                                                   random_state=27)
        else:
            self._train_set_X = self.dataset.drop(columns=self._target_col_name)
            self._train_set_Y = self.dataset[self._target_col_name]
            self._test_set_X, self._test_set_Y = None, None
        if use_grid:
            self._params_grid = self.generate_grid()
            self._gs_list = [GridSearchCV(estimator=LogisticRegression(),
                                          param_grid=param_grid,
                                          n_jobs=-1) for param_grid in self._params_grid]
            self._classifier = None
        else:
            self._params_grid, self._gs_list = None, None
            self._classifier = LogisticRegression()

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
            self.estimator.fit(self._train_set_X, self._train_set_Y)
        else:
            pass

    @classmethod
    def generate_grid(cls):
        params_grid = [{"penalty": ["l2"],
                        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                        "C": cls.C_GRID,
                        "class_weight": ["balanced"],
                        "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                        "multi_class": ["auto", "ovr", "multinomial"],
                        "n_jobs": -1},
                       {"penalty": ["none"],
                        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
                        "class_weight": ["balanced"],
                        "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                        "multi_class": ["auto", "ovr", "multinomial"],
                        "n_jobs": -1},
                       {"penalty": ["elasticnet"],
                        "solver": ["saga"],
                        "C": cls.C_GRID,
                        "l1_ratio": cls.L1_RATIO_GRID,
                        "class_weight": ["balanced"],
                        "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                        "multi_class": ["auto", "ovr", "multinomial"],
                        "n_jobs": -1},
                       {"penalty": ["l1"],
                        "solver": ["liblinear"],
                        "C": cls.C_GRID,
                        "class_weight": ["balanced"],
                        "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                        "multi_class": ["auto", "ovr"],
                        "n_jobs": -1},
                       {"penalty": ["l1"],
                        "solver": ["saga"],
                        "C": cls.C_GRID,
                        "class_weight": ["balanced"],
                        "max_iter": range(cls.MIN_ITER, cls.MAX_ITER, cls.STEP_ITER),
                        "multi_class": ["auto", "ovr", "multinomial"],
                        "n_jobs": -1}]

        return params_grid
