"""The candidate classifiers.

All use fixed default hyperparameters (this is algorithm selection, not hyperparameter search).
Each entry says which preprocessing it receives and which package it needs. Optional packages
are imported only when the classifier is used, and `check_available` fails loudly before a run
starts, so the set of classifiers is identical for every dataset in the database.
"""
from __future__ import annotations

import importlib
import inspect

from ..utils import cuda_available


def _make(cls, **kwargs):
    """Instantiate `cls`, passing only the keyword arguments its constructor accepts.

    Keeps the code working across library versions whose constructors differ slightly.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return cls(**kwargs)
        return cls(**{k: v for k, v in kwargs.items() if k in params})
    except (TypeError, ValueError):
        return cls()


def _logreg(seed, cfg):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=seed)


def _lda(seed, cfg):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def _gnb(seed, cfg):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def _knn(seed, cfg):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5, n_jobs=cfg["n_jobs"])


def _svm(seed, cfg):
    from sklearn.svm import SVC
    return SVC(kernel="rbf", probability=False, random_state=seed, cache_size=500)


def _dtree(seed, cfg):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=seed)


def _rf(seed, cfg):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, n_jobs=cfg["n_jobs"], random_state=seed)


def _extratrees(seed, cfg):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=100, n_jobs=cfg["n_jobs"], random_state=seed)


def _hgb(seed, cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=seed)


def _mlp_sklearn(seed, cfg):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(100,), early_stopping=True, max_iter=300, random_state=seed)


def _lightgbm(seed, cfg):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_jobs=cfg["n_jobs"], random_state=seed, verbose=-1)


def _catboost(seed, cfg):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(verbose=0, thread_count=cfg["n_jobs"], random_seed=seed, allow_writing_files=False)


def _realmlp(seed, cfg):
    from pytabkit import RealMLP_TD_Classifier
    return _make(RealMLP_TD_Classifier, device=cfg.get("realmlp_device", "cpu"), random_state=seed,
                 n_threads=cfg["n_jobs"], verbosity=0)


def _tabpfn(seed, cfg):
    from tabpfn import TabPFNClassifier
    dev = cfg.get("tabpfn_device", "auto")
    if dev == "auto":
        dev = "cuda" if cuda_available() else "cpu"
    return _make(TabPFNClassifier, device=dev, random_state=seed, ignore_pretraining_limits=True)


REGISTRY: dict[str, dict] = {
    #  name          factory       input        package      family
    "logreg":      {"make": _logreg, "input": "standard", "pkg": "sklearn", "family": "linear"},
    "lda":         {"make": _lda, "input": "standard", "pkg": "sklearn", "family": "linear (generative)"},
    "gnb":         {"make": _gnb, "input": "standard", "pkg": "sklearn", "family": "probabilistic"},
    "knn":         {"make": _knn, "input": "standard", "pkg": "sklearn", "family": "instance-based"},
    "svm":         {"make": _svm, "input": "standard", "pkg": "sklearn", "family": "kernel"},
    "dtree":       {"make": _dtree, "input": "standard", "pkg": "sklearn", "family": "single tree"},
    "rf":          {"make": _rf, "input": "standard", "pkg": "sklearn", "family": "bagging ensemble"},
    "extratrees":  {"make": _extratrees, "input": "standard", "pkg": "sklearn", "family": "randomized ensemble"},
    "lightgbm":    {"make": _lightgbm, "input": "standard", "pkg": "lightgbm", "family": "gradient boosting"},
    "catboost":    {"make": _catboost, "input": "raw_cat", "pkg": "catboost", "family": "gradient boosting"},
    "realmlp":     {"make": _realmlp, "input": "standard", "pkg": "pytabkit", "family": "deep learning (MLP)"},
    "tabpfn":      {"make": _tabpfn, "input": "standard", "pkg": "tabpfn", "family": "deep learning (transformer)"},
    # sklearn-only fallbacks (used in the offline smoke test, or if a package will not install)
    "hgb":         {"make": _hgb, "input": "standard", "pkg": "sklearn", "family": "gradient boosting"},
    "mlp_sklearn": {"make": _mlp_sklearn, "input": "standard", "pkg": "sklearn", "family": "neural network"},
}


def check_available(names: list[str]) -> None:
    missing = []
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"unknown classifier '{n}'. Known: {sorted(REGISTRY)}")
        try:
            importlib.import_module(REGISTRY[n]["pkg"])
        except ImportError:
            missing.append(f"{n} (pip install {REGISTRY[n]['pkg']})")
    if missing:
        raise ImportError("Missing packages for: " + ", ".join(missing) +
                          ". Install them, or remove these classifiers from the config.")


def make_classifier(name: str, seed: int, perf_cfg: dict):
    return REGISTRY[name]["make"](seed, perf_cfg)
