import numpy as np
import pandas as pd

from autoclassify.data.preprocess import StandardPreprocessor
from autoclassify.data.splits import _target_counts, budget_subsample, make_meta_split, make_row_split


def test_row_split_disjoint_and_complete():
    y = pd.Series(np.repeat(["a", "b", "c"], [300, 150, 50]))
    sp = make_row_split(y, seed=0, fractions={"train": 0.6, "val": 0.2, "test": 0.2})
    allidx = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert len(set(allidx)) == len(allidx) == len(y)          # no row in two splits
    for part in sp.values():                                   # stratified: every class everywhere
        assert set(y.iloc[part]) == {"a", "b", "c"}


def test_budget_subsamples_are_nested_and_keep_all_classes():
    y = np.repeat([0, 1, 2], [200, 60, 12])
    subs = [set(budget_subsample(y, f, seed=3)) for f in (0.1, 0.25, 0.5, 1.0)]
    for small, big in zip(subs, subs[1:]):
        assert small <= big
    assert subs[-1] == set(range(len(y)))
    for s in subs:
        assert set(y[list(s)]) == {0, 1, 2}


def test_meta_split_counts_and_disjoint():
    meta = pd.DataFrame({"did": range(20), "status": "ok", "n_rows": np.linspace(600, 40000, 20),
                         "n_classes": [2, 3] * 10})
    sp = make_meta_split(meta, {"train": 0.7, "val": 0.1, "test": 0.2}, seed=0)
    assert [len(sp[k]) for k in ("train", "val", "test")] == [14, 2, 4]
    assert not (set(sp["train"]) & set(sp["test"])) and not (set(sp["val"]) & set(sp["test"]))
    assert _target_counts(5, {"train": 0.6, "val": 0.2, "test": 0.2}) == {"train": 3, "val": 1, "test": 1}


def test_preprocessor_uses_train_statistics_only():
    tr = pd.DataFrame({"x": [0.0, 2.0, 4.0], "c": ["u", "v", "u"]})
    te = pd.DataFrame({"x": [100.0], "c": ["unseen"]})
    pre = StandardPreprocessor().fit(tr)
    out = pre.transform(te)
    assert out.shape[1] == 3                                  # 1 numeric + 2 one-hot levels
    assert np.isclose(out[0, 0], (100 - 2) / np.std([0, 2, 4]))  # scaled with TRAIN mean/std
