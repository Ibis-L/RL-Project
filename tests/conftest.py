"""Tiny hand-made performance tables so environment/agent tests need no real data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoclassify.perfdb.store import Table  # noqa: E402

CLFS = ["a", "b", "c"]
LEVELS = [0.1, 0.25, 0.5, 1.0]


def make_table(key, seed):
    rng = np.random.default_rng(seed)
    val = np.sort(rng.uniform(0.5, 0.9, (3, 4)), axis=1)      # more data -> better (usually)
    test = np.clip(val + rng.normal(0, 0.02, val.shape), 0, 1)
    cost_s = np.array([[0.1, 0.25, 0.5, 1.0]]) * np.array([[1.0], [3.0], [6.0]])
    return Table(key, val, test, cost_s, cost_s / cost_s[:, -1].sum(), np.full((3, 4), "ok", object), 0.5, float(cost_s[:, -1].sum()))


@pytest.fixture
def tiny():
    keys = [(i, 0) for i in range(6)]
    tables = {k: make_table(k, i) for i, k in enumerate(keys)}
    rng = np.random.default_rng(0)
    context = {k: {"split": "train" if k[0] < 4 else "val", "features": rng.normal(size=4)} for k in keys}
    return tables, context, keys
