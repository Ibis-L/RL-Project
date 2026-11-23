"""Preprocessing, always fitted on the training rows a model is given (never val/test).

* "standard" : median impute + scale numeric; most-frequent impute + one-hot categorical
               (levels beyond `max_categories` grouped as infrequent). Dense float32 output.
* "raw_cat"  : for CatBoost. Numeric left as float (NaN allowed), categorical as strings with
               "missing" for NaN. CatBoost encodes categoricals itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .sources import categorical_columns


class StandardPreprocessor:
    def __init__(self, max_categories: int = 20):
        self.max_categories = max_categories

    def fit(self, X: pd.DataFrame):
        cats = categorical_columns(X)
        nums = [c for c in X.columns if c not in cats]
        parts = []
        if nums:
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                           ("sc", StandardScaler())]), nums))
        if cats:
            parts.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=self.max_categories,
                                     sparse_output=False, dtype=np.float32))]), cats))
        self.ct = ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)
        self.ct.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = self.ct.transform(X).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class RawCatPreprocessor:
    def fit(self, X: pd.DataFrame):
        self.cats = categorical_columns(X)
        self.cols = list(X.columns)
        self.cat_idx = [self.cols.index(c) for c in self.cats]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.cols].copy()
        for c in self.cats:
            X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
        return X


def make_preprocessor(kind: str, max_categories: int = 20):
    if kind == "standard":
        return StandardPreprocessor(max_categories)
    if kind == "raw_cat":
        return RawCatPreprocessor()
    raise ValueError(kind)
