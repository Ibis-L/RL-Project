"""Dataset features ("meta-features") for the RL state, computed on the TRAIN rows of each
(dataset, seed) only. There is no meta-learner: these numbers are simply part of the agent's state.

Two groups:
* hand-written (always available, fast): size, types, missingness, class balance, simple
  statistics, information-theoretic measures, PCA structure;
* PyMFE (if installed and enabled): groups general / statistical / info-theory, summarised by
  mean and sd (Alcobaca et al., JMLR 2020).
Landmarkers (quick model fits) are deliberately excluded: they cost compute that would have to
be charged to the selection budget.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from ..data.preprocess import StandardPreprocessor
from ..data.sources import categorical_columns, load_dataset
from ..data.splits import budget_subsample, load_row_split
from ..utils import get_logger

log = get_logger()


def _entropy(counts) -> float:
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def handcrafted(X: pd.DataFrame, y: np.ndarray, X_enc: np.ndarray, n_train_total: int) -> dict:
    cats = categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    n, d = n_train_total, X.shape[1]
    counts = np.bincount(y)
    counts = counts[counts > 0]
    k = len(counts)
    mf = {
        "log_n": np.log10(n), "log_d": np.log10(d), "log_d_encoded": np.log10(max(1, X_enc.shape[1])),
        "log_n_over_d": np.log10(n / d), "n_classes": k, "cat_ratio": len(cats) / d,
        "missing_ratio": float(X.isna().to_numpy().mean()),
        "class_entropy_norm": _entropy(counts) / np.log2(k) if k > 1 else 0.0,
        "majority_ratio": counts.max() / counts.sum(), "minority_ratio": counts.min() / counts.sum(),
        "imbalance_ratio": counts.max() / counts.min(),
    }
    if nums:
        A = X[nums].to_numpy(float)
        A = np.where(np.isnan(A), np.nanmedian(A, axis=0), A)
        A = A[:, np.nanstd(A, axis=0) > 0] if A.size else A
        if A.shape[1] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk, ku = stats.skew(A, axis=0), stats.kurtosis(A, axis=0)
            mf.update(skew_mean=float(np.nanmean(np.abs(sk))), skew_sd=float(np.nanstd(sk)),
                      kurt_mean=float(np.nanmean(ku)), kurt_sd=float(np.nanstd(ku)))
            if A.shape[1] > 1:
                B = A[:, :50]
                C = np.corrcoef(B, rowvar=False)
                iu = np.triu_indices_from(C, 1)
                mf["abs_corr_mean"] = float(np.nanmean(np.abs(C[iu])))
    if cats:
        ents = [_entropy(X[c].value_counts().to_numpy()) for c in cats]
        mf["cat_entropy_mean"] = float(np.mean(ents))
    if X_enc.shape[1] > 0:
        Z = X_enc[:, : min(X_enc.shape[1], 200)]
        Z = Z[:, Z.std(axis=0) > 0]
        if Z.shape[1] > 0:
            p = PCA().fit(Z)
            cum = np.cumsum(p.explained_variance_ratio_)
            mf["pca_frac_95"] = float((np.searchsorted(cum, 0.95) + 1) / Z.shape[1])
            mf["pca_first_ratio"] = float(p.explained_variance_ratio_[0])
            cols = np.random.default_rng(0).choice(Z.shape[1], size=min(50, Z.shape[1]), replace=False)
            mi = mutual_info_classif(Z[:, cols], y, random_state=0)
            mf["mutual_info_mean"] = float(np.mean(mi))
            mf["mutual_info_max"] = float(np.max(mi))
            mf["mi_over_class_entropy"] = float(np.mean(mi) / max(_entropy(counts) * np.log(2), 1e-9))
    return mf


def pymfe_features(X_enc: np.ndarray, y: np.ndarray) -> dict:
    try:
        from pymfe.mfe import MFE
    except ImportError:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MFE(groups=["general", "statistical", "info-theory"], summary=["mean", "sd"], random_state=0)
            m.fit(X_enc, y, transform_cat=None, suppress_warnings=True)
            names, vals = m.extract(suppress_warnings=True)
        return {f"mfe_{n}": (float(v) if v is not None and np.isfinite(v) else np.nan) for n, v in zip(names, vals)}
    except Exception as e:  # noqa: BLE001
        log.warning(f"pymfe failed: {e}")
        return {}


def compute_one(paths, did: int, seed: int, cfg: dict) -> dict:
    data = load_dataset(paths, did)
    sp = load_row_split(paths, did, seed)
    X = data["X"].iloc[sp["train"]].reset_index(drop=True)
    y = LabelEncoder().fit_transform(data["y"].iloc[sp["train"]])
    t0 = time.perf_counter()
    m = cfg["meta"]["mf_max_rows"]
    if len(y) > m:   # stratified subsample for speed; deterministic
        pos = budget_subsample(y, m / len(y), seed)
        Xs, ys = X.iloc[pos].reset_index(drop=True), y[pos]
    else:
        Xs, ys = X, y
    ys = LabelEncoder().fit_transform(ys)
    X_enc = StandardPreprocessor(cfg["filters"]["max_categories"]).fit(Xs).transform(Xs)
    mf = handcrafted(Xs, ys, X_enc, n_train_total=len(y))
    if cfg["meta"]["use_pymfe"]:
        mf.update(pymfe_features(X_enc, ys))
    return {"did": did, "seed": seed, "mf_time_s": time.perf_counter() - t0, **mf}


def compute_all(cfg: dict, paths, dids: list[int]) -> pd.DataFrame:
    out = paths.meta / "metafeatures.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["did", "seed"])
    have = set(zip(old["did"].astype(int), old["seed"].astype(int)))
    rows = []
    for did in dids:
        for seed in cfg["split"]["seeds"]:
            if (did, seed) in have:
                continue
            rows.append(compute_one(paths, did, seed, cfg))
            log.info(f"meta-features {did} seed {seed}: {len(rows[-1]) - 3} features")
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    df.to_csv(out, index=False)
    return df
