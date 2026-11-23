"""Load the performance database and turn it into lookup tables for the environment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Table:
    """All results of one (dataset, seed). Arrays are [n_classifiers, n_budget_levels]."""
    key: tuple
    val: np.ndarray        # validation balanced accuracy (what a method observes)
    test: np.ndarray       # test balanced accuracy (reporting only)
    cost_s: np.ndarray     # seconds
    cost: np.ndarray       # normalised: seconds / (sum over classifiers of 100%-budget seconds)
    status: np.ndarray
    chance: float
    exhaustive_s: float


def load_perfdb(paths, dids=None) -> pd.DataFrame:
    files = sorted(paths.perfdb.glob("*.csv"))
    if dids is not None:
        keep = {f"{int(d)}.csv" for d in dids}
        files = [f for f in files if f.name in keep]
    if not files:
        raise FileNotFoundError(f"no performance-database files in {paths.perfdb}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def completeness(df: pd.DataFrame, classifiers, budgets, seeds) -> pd.DataFrame:
    need = len(classifiers) * len(budgets) * len(seeds)
    g = df.groupby("did").size().rename("rows").to_frame()
    g["expected"] = need
    g["complete"] = g["rows"] >= need
    return g


def build_tables(df: pd.DataFrame, classifiers: list[str], budgets: list[float]) -> dict:
    K, L = len(classifiers), len(budgets)
    cidx = {c: i for i, c in enumerate(classifiers)}
    tables = {}
    for (did, seed), g in df.groupby(["did", "seed"]):
        val = np.full((K, L), np.nan)
        test, cost_s = val.copy(), val.copy()
        status = np.full((K, L), "missing", dtype=object)
        for r in g.itertuples(index=False):
            if r.clf not in cidx or int(r.level) >= L:
                continue
            i, l = cidx[r.clf], int(r.level)
            val[i, l], test[i, l], cost_s[i, l], status[i, l] = r.val_bacc, r.test_bacc, r.cost_s, r.status
        if np.isnan(val).any() or np.isnan(cost_s).any():
            missing = [(classifiers[i], budgets[l]) for i, l in zip(*np.where(np.isnan(val) | np.isnan(cost_s)))]
            raise ValueError(f"dataset {did} seed {seed} incomplete: {missing[:5]}... run `perfdb` again")
        cost_s = np.maximum(cost_s, 1e-4)
        ex = float(cost_s[:, -1].sum())
        tables[(int(did), int(seed))] = Table((int(did), int(seed)), val, test, cost_s, cost_s / ex, status,
                                              float(g["chance"].iloc[0]), ex)
    return tables
