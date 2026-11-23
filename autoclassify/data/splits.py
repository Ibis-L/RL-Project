"""Two-level splitting.

1. Dataset level: meta-train / meta-val / meta-test, stratified by size bucket and by
   binary-vs-multiclass. Saved once to dataset_split.json and never changed afterwards.
2. Row level, inside each dataset and seed: stratified train / val / test rows.
   Budget subsamples are *nested* subsets of the train rows (10% within 25% within ...).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import load_json, save_json


def _target_counts(n: int, fractions: dict) -> dict:
    names = list(fractions)
    raw = {k: fractions[k] * n for k in names}
    cnt = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(names, key=lambda k: raw[k] - cnt[k], reverse=True)[: n - sum(cnt.values())]:
        cnt[k] += 1
    # every split with a positive fraction gets at least one dataset when possible
    for k in names:
        if fractions[k] > 0 and cnt[k] == 0 and n >= len(names):
            donor = max(cnt, key=cnt.get)
            cnt[donor] -= 1
            cnt[k] += 1
    return cnt


def make_meta_split(meta: pd.DataFrame, fractions: dict, seed: int) -> dict:
    ok = meta[meta["status"] == "ok"].copy()
    ok["size_bucket"] = pd.cut(ok["n_rows"], [0, 2000, 10000, np.inf], labels=False)
    ok["binary"] = (ok["n_classes"] == 2).astype(int)
    rng = np.random.default_rng(seed)
    ok["r"] = rng.random(len(ok))
    ok = ok.sort_values(["size_bucket", "binary", "r"])
    target = _target_counts(len(ok), fractions)
    current = {k: 0 for k in target}
    out = {k: [] for k in target}
    for did in ok["did"]:
        # assign to the split furthest below its target share -> strata are spread evenly
        k = max((k for k in target if current[k] < target[k]),
                key=lambda k: (target[k] - current[k]) / max(target[k], 1))
        out[k].append(int(did))
        current[k] += 1
    return {k: sorted(v) for k, v in out.items()}


def save_meta_split(split: dict, path, overwrite: bool = False) -> dict:
    if path.exists() and not overwrite:
        existing = load_json(path)
        return existing  # frozen: never silently reshuffle datasets between splits
    save_json(split, path)
    return split


def make_row_split(y: pd.Series, seed: int, fractions: dict) -> dict:
    idx = np.arange(len(y))
    tr = fractions["train"]
    va = fractions["val"] / (1 - tr)
    i_tr, i_rest = train_test_split(idx, train_size=tr, stratify=y, random_state=seed)
    i_va, i_te = train_test_split(i_rest, train_size=va, stratify=y.iloc[i_rest], random_state=seed + 10_000)
    return {"train": np.sort(i_tr), "val": np.sort(i_va), "test": np.sort(i_te)}


def save_row_split(split: dict, paths, did: int, seed: int) -> None:
    np.savez(paths.splits / f"{int(did)}_s{seed}.npz", **split)


def load_row_split(paths, did: int, seed: int) -> dict:
    z = np.load(paths.splits / f"{int(did)}_s{seed}.npz")
    return {k: z[k] for k in ("train", "val", "test")}


def budget_subsample(y_train: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Positions (into the train rows) of a stratified, nested subsample.

    A fixed per-class permutation is drawn from `seed`; the subsample for fraction f takes the
    first ceil(f * n_c) rows of each class (at least one). Because the permutation does not
    depend on f, the 10% subsample is contained in the 25% one, and so on.
    """
    y_train = np.asarray(y_train)
    rng = np.random.default_rng(seed + 777)
    keep = []
    for c in np.unique(y_train):
        pos = np.flatnonzero(y_train == c)
        pos = pos[rng.permutation(len(pos))]
        keep.append(pos[: max(1, math.ceil(frac * len(pos)))])
    return np.sort(np.concatenate(keep))
