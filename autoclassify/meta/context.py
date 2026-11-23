"""Per-dataset context for the RL agent (no meta-learner).

The agent's state includes a fixed set of simple dataset descriptors, standardised with statistics
from meta-train datasets only. There is no separate meta-learning model: the RL policy itself
learns which classifiers to try on which kind of data.

Two plain statistics of meta-train datasets are also saved for the non-learning baselines:
  * average-rank order of the classifiers (Average-rank baseline, and the order used by
    best-on-small-budget and successive halving);
  * average normalised cost of each classifier at 100% (the baselines' cost estimate when they
    check whether an evaluation fits the budget).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger, load_json, save_json

log = get_logger()

# Fixed, documented feature set (computed in metafeatures.handcrafted); missing values -> 0 after scaling.
STATE_FEATURES = ["log_n", "log_d", "log_d_encoded", "log_n_over_d", "n_classes", "cat_ratio", "missing_ratio",
                  "class_entropy_norm", "majority_ratio", "imbalance_ratio", "skew_mean", "kurt_mean",
                  "abs_corr_mean", "pca_frac_95", "pca_first_ratio", "mutual_info_mean", "mi_over_class_entropy"]
LOG_FEATURES = {"n_classes", "imbalance_ratio", "kurt_mean"}   # heavy-tailed -> log1p before scaling


def _prepare(mf: pd.DataFrame) -> pd.DataFrame:
    X = mf.reindex(columns=STATE_FEATURES).astype(float)
    for c in LOG_FEATURES:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X


def build_context(cfg: dict, paths, tables: dict, mf: pd.DataFrame, split: dict, classifiers: list[str]) -> pd.DataFrame:
    """Write results/<stage>/meta/context.csv (standardised features) + baseline statistics."""
    mf = mf.set_index(["did", "seed"])
    split_of = {d: s for s, ds in split.items() for d in ds}
    keys = [k for k in tables if k in mf.index]
    tr = [k for k in keys if split_of.get(k[0]) == "train"]
    if not tr:
        raise ValueError("no meta-train datasets with complete performance data")

    X = _prepare(mf.loc[keys])
    Xtr = X.loc[tr]
    med, mu = Xtr.median(), Xtr.fillna(Xtr.median()).mean()
    sd = Xtr.fillna(Xtr.median()).std().replace(0, 1).fillna(1)
    Z = ((X.fillna(med) - mu) / sd).clip(-5, 5).fillna(0.0)
    ctx = Z.reset_index().rename(columns={c: f"f_{c}" for c in STATE_FEATURES})
    ctx.insert(0, "split", [split_of[d] for d in ctx["did"]])
    ctx.to_csv(paths.meta / "context.csv", index=False)
    save_json({"features": STATE_FEATURES, "median": med.tolist(), "mean": mu.tolist(), "std": sd.tolist()},
              paths.meta / "feature_scaling.json")

    perf = np.stack([tables[k].val[:, -1] for k in tr])
    ranks = (-perf).argsort(axis=1).argsort(axis=1).mean(axis=0)
    cost = np.stack([tables[k].cost[:, -1] for k in tr]).mean(axis=0)
    save_json({"classifiers": classifiers, "avg_rank": ranks.tolist(),
               "avg_rank_order": [int(i) for i in np.argsort(ranks)], "avg_cost": cost.tolist()},
              paths.meta / "baseline_stats.json")
    log.info(f"context: {len(ctx)} (dataset, seed) rows, {len(STATE_FEATURES)} features; "
             f"average-rank order: {[classifiers[i] for i in np.argsort(ranks)]}")
    return ctx


def load_context(paths) -> dict:
    ctx = pd.read_csv(paths.meta / "context.csv")
    fcols = [c for c in ctx.columns if c.startswith("f_")]
    return {(int(r.did), int(r.seed)): {"split": r.split, "features": np.asarray([getattr(r, c) for c in fcols], float)}
            for r in ctx.itertuples(index=False)}


def load_baseline_stats(paths) -> dict:
    return load_json(paths.meta / "baseline_stats.json")
