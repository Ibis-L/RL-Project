"""Configuration loading.

Every experiment is driven by one YAML file in configs/. Values missing from the file
fall back to DEFAULTS below, so a config only needs to state what differs.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "project": "autoclassify",
    "paths": {"data_dir": "data/default", "results_dir": "results/default", "cache_dir": "data/openml_cache"},
    "datasets": {
        "source": "openml",          # "openml" or "synthetic" (synthetic = offline pipeline test only)
        "ids": [],                   # explicit OpenML dataset IDs
        "suite": None,               # e.g. "OpenML-CC18": adds every dataset of the suite
        "n_extra": 0,                # extra datasets sampled from the OpenML catalogue (after filters)
        "extra_seed": 0,
        "exclude_ids": [],
        "synthetic": {"n": 12, "seed": 0},
    },
    "filters": {
        "min_rows": 500,
        "max_rows": 50000,           # larger datasets are stratified-subsampled to this size
        "max_features_encoded": 500,
        "max_classes": 50,
        "min_class_count": 10,       # rarer classes are dropped (logged)
        "max_categories": 20,        # one-hot levels per categorical column (rest grouped)
    },
    "split": {
        "meta_fractions": {"train": 0.7, "val": 0.1, "test": 0.2},
        "row_fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
        "seeds": [0, 1, 2],
        "split_seed": 42,
    },
    "budgets": [0.1, 0.25, 0.5, 1.0],
    "classifiers": ["logreg", "lda", "gnb", "knn", "svm", "dtree", "rf", "extratrees",
                    "lightgbm", "catboost", "realmlp", "tabpfn"],
    "perfdb": {
        "n_jobs": 4,
        "timeout_s": 300,
        "tabpfn_device": "auto",     # "auto" | "cuda" | "cpu"
        "realmlp_device": "cpu",
        "growth_exponent": {"svm": 2.0, "knn": 1.0, "default": 1.2},
    },
    "meta": {"mf_max_rows": 2000, "use_pymfe": False},
    "env": {"episode_budgets": [0.1, 0.25, 0.5, 1.0], "lam": 0.05},
    "agents": {
        "algos": ["qlearning", "dqn", "ddqn"],
        "seeds": [0, 1, 2, 3, 4],
        "dqn": {
            "hidden": [128, 128], "lr": 5e-4, "gamma": 0.99, "batch_size": 64,
            "buffer_size": 50000, "warmup": 1000, "target_update": 1000,
            "train_steps": 100000, "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.5,
            "eval_every": 5000, "grad_clip": 10.0,
        },
        "qlearning": {
            "episodes": 50000, "alpha": 0.1, "gamma": 1.0, "eps_start": 1.0, "eps_end": 0.05,
            "eps_decay_frac": 0.5, "eval_every": 5000, "use_data_bucket": True,
        },
    },
    "eval": {"budgets": [0.1, 0.25, 0.5, 1.0], "random_repeats": 10, "rl_algo_for_ablations": "ddqn"},
    "ablations": {
        "full": {},
        "no_dataset_features": {"flags": {"features": False}},
        "no_budget": {"flags": {"budget": False}},
        "no_cost_obs": {"flags": {"cost": False}},
        "one_shot": {"flags": {"obs": False}},
        "full_budget_only": {"allowed_levels": [-1]},
        "lam_0": {"lam": 0.0},
        "lam_0.1": {"lam": 0.1},
        "lam_0.2": {"lam": 0.2},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "ablations":
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Paths:
    root: Path
    data: Path
    results: Path
    cache: Path

    @property
    def raw(self) -> Path: return self.data / "raw"
    @property
    def splits(self) -> Path: return self.data / "splits"
    @property
    def metadata_csv(self) -> Path: return self.data / "metadata.csv"
    @property
    def candidates_csv(self) -> Path: return self.data / "candidates.csv"
    @property
    def dataset_split(self) -> Path: return self.data / "dataset_split.json"
    @property
    def perfdb(self) -> Path: return self.results / "perfdb"
    @property
    def meta(self) -> Path: return self.results / "meta"
    @property
    def agents(self) -> Path: return self.results / "agents"
    @property
    def eval(self) -> Path: return self.results / "eval"
    @property
    def figures(self) -> Path: return self.results / "figures"

    def make(self) -> None:
        for p in [self.raw, self.splits, self.perfdb, self.meta, self.agents, self.eval, self.figures, self.cache]:
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def get_paths(cfg: dict, root: str | os.PathLike | None = None) -> Paths:
    root = Path(root or os.getcwd()).resolve()
    p = cfg["paths"]
    paths = Paths(root=root, data=root / p["data_dir"], results=root / p["results_dir"], cache=root / p["cache_dir"])
    paths.make()
    return paths
