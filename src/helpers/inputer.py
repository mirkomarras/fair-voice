import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.feature_extraction.text import HashingVectorizer


class DataInputer:
    CATEGORICAL_RATIO = 0.5

    def __init__(self, df: pd.DataFrame):
        self._numerical = []
        self._categorical = []
        self._textual = []

        self._transformed_dataset = df

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                self._numerical.append([column])
            elif pd.api.types.is_string_dtype(df[column]):
                if df[column].nunique()/len(df[column]) < self.CATEGORICAL_RATIO:
                    self._categorical.append(column)
                else:
                    self._textual.append(column)
        self._max_word_sentence = 0
        for text_column in self._textual:
            current_words = df[text_column].str.split().apply(len).max()
            if current_words > self._max_word_sentence:
                self._max_word_sentence = current_words
        if self._numerical:
            numerical_feat = gen_features(self._numerical, [{"class": SimpleImputer,
                                                             "missing_values": np.nan,
                                                             "strategy": "mean"}, StandardScaler])
        else:
            numerical_feat = []

        if self._categorical:
            categorical_feat = gen_features([self._categorical], [{"class": OneHotEncoder}])
        else:
            categorical_feat = []

        if self._textual:
            textual_feat = gen_features(self._textual, [{"class": HashingVectorizer,
                                                         "n_features": self._max_word_sentence}])
        else:
            textual_feat = []

        self._data_mapper = DataFrameMapper(numerical_feat + categorical_feat + textual_feat, df_out=True)
        self._data_mapper.fit(df)

    @property
    def transformed_dataset(self):
        return self._transformed_dataset

    def transform(self, data):
        return self._data_mapper.transform(data)
