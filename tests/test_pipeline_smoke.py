"""End-to-end run of every stage on 6 synthetic datasets with 3 fast classifiers (~10 s)."""
import numpy as np
import pandas as pd
import yaml

from autoclassify.cli import main
from autoclassify.utils import load_json


def test_full_pipeline(tmp_path):
    cfg = {
        "project": "t", "paths": {"data_dir": "data", "results_dir": "results", "cache_dir": "cache"},
        "datasets": {"source": "synthetic", "synthetic": {"n": 6, "seed": 1}},
        "split": {"meta_fractions": {"train": 0.5, "val": 0.25, "test": 0.25}, "seeds": [0]},
        "classifiers": ["logreg", "gnb", "dtree"],
        "perfdb": {"n_jobs": 1, "timeout_s": 60},
        "meta": {"use_pymfe": False},
        "agents": {"algos": ["qlearning", "ddqn"], "seeds": [0],
                   "dqn": {"train_steps": 600, "warmup": 100, "eval_every": 300, "target_update": 100, "buffer_size": 2000},
                   "qlearning": {"episodes": 400, "eval_every": 200}},
        "eval": {"random_repeats": 2},
        "ablations": {"full": {}, "one_shot": {"flags": {"obs": False}}},
    }
    c = tmp_path / "cfg.yaml"
    c.write_text(yaml.safe_dump(cfg))
    base = ["--config", str(c), "--root", str(tmp_path)]
    for cmd in ("fetch", "split", "perfdb", "features", "train", "evaluate", "ablate", "figures"):
        assert main([cmd, *base]) == 0, cmd

    res = pd.read_csv(tmp_path / "results/eval/val_results.csv")
    assert {"ddqn", "qlearning", "succ_halving", "exhaustive"} <= set(res["method"])
    assert (res["regret"].between(0, 1)).all()
    aware = res[res["budget_aware"]]
    assert (aware["budget_used"] <= aware["budget"] + 1e-9).all()

    # perfdb resumes instead of refitting
    n_rows = len(pd.read_csv(next((tmp_path / "results/perfdb").glob("*.csv"))))
    assert main(["perfdb", *base]) == 0
    assert len(pd.read_csv(next((tmp_path / "results/perfdb").glob("*.csv")))) == n_rows

    split = load_json(tmp_path / "data/dataset_split.json")
    ctx = pd.read_csv(tmp_path / "results/meta/context.csv")
    assert set(ctx.loc[ctx["split"] == "test", "did"]) == set(split["test"])
    assert set(ctx.loc[ctx["split"] == "train", "did"]).isdisjoint(split["test"])
    feats = ctx.filter(like="f_").to_numpy()
    assert np.isfinite(feats).all() and feats.shape[1] == 17
    # scaling statistics come from meta-train only: train features are ~centred, by construction
    assert abs(ctx.loc[ctx["split"] == "train"].filter(like="f_").mean().mean()) < 0.5

    # meta-test is guarded
    assert main(["evaluate", *base, "--split", "test"]) == 0
    assert main(["evaluate", *base, "--split", "test"]) == 1
