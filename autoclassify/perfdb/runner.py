"""Build the performance database: every (dataset, seed, classifier, budget) is fitted once.

Budget semantics (stated in the report):
  budget b = fraction of the TRAIN rows (stratified, nested subsample). Every fit starts from
  scratch: no classifier is trained incrementally. For TabPFN, which does no gradient training,
  the fraction is the number of rows given as in-context examples.

Cost = wall-clock seconds of [fit preprocessing + fit model + transform val + predict val].
Test-set prediction is timed separately and NOT counted, because test rows are only for reporting.

Time-outs: Python cannot safely kill a running C-level fit, so we use an explicit, documented rule.
  * if a finished fit took longer than `timeout_s` -> status "timeout" (scores set to chance,
    cost = timeout_s), as if the system had stopped it at the limit;
  * before a larger budget, the cost is extrapolated from the previous budget as
    t_prev * (f / f_prev) ** exponent; if that exceeds `timeout_s` the fit is skipped as
    "timeout_predicted" (same scoring). Higher budgets after a time-out are skipped too.
Errors -> status "error", scores = chance, cost = time until the error.
Chance score for balanced accuracy = 1 / n_classes.

Results are appended to results/<stage>/perfdb/<did>.csv one row at a time, so an interrupted
session (Kaggle/Colab disconnect) resumes exactly where it stopped.
"""
from __future__ import annotations

import csv
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from ..data.preprocess import make_preprocessor
from ..data.sources import load_dataset
from ..data.splits import budget_subsample, load_row_split
from ..utils import get_logger, hardware_info, package_versions
from .classifiers import REGISTRY, check_available, make_classifier

log = get_logger()

COLUMNS = ["did", "seed", "clf", "budget", "level", "n_train_used", "status", "val_bacc", "test_bacc",
           "val_f1", "test_f1", "train_bacc", "fit_time", "predict_val_time", "predict_test_time",
           "cost_s", "chance", "message", "hardware", "versions"]


def _done_keys(path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["seed"].astype(int), df["clf"], df["level"].astype(int)))


def _append(path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def _predict(model, X):
    return np.asarray(model.predict(X)).ravel().astype(int)


def run_one_fit(name, X_tr, y_tr, X_va, y_va, X_te, y_te, seed, perf_cfg, max_categories):
    """Fit one classifier on one subsample. Returns dict of scores and timings."""
    spec = REGISTRY[name]
    t0 = time.perf_counter()
    pre = make_preprocessor(spec["input"], max_categories).fit(X_tr)
    A_tr = pre.transform(X_tr)
    model = make_classifier(name, seed, perf_cfg)
    if spec["input"] == "raw_cat":
        model.fit(A_tr, y_tr, cat_features=pre.cat_idx)
    else:
        model.fit(A_tr, y_tr)
    t1 = time.perf_counter()
    p_va = _predict(model, pre.transform(X_va))
    t2 = time.perf_counter()
    p_te = _predict(model, pre.transform(X_te))
    t3 = time.perf_counter()
    p_tr = _predict(model, A_tr)
    return {
        "val_bacc": balanced_accuracy_score(y_va, p_va), "test_bacc": balanced_accuracy_score(y_te, p_te),
        "val_f1": f1_score(y_va, p_va, average="macro"), "test_f1": f1_score(y_te, p_te, average="macro"),
        "train_bacc": balanced_accuracy_score(y_tr, p_tr),
        "fit_time": t1 - t0, "predict_val_time": t2 - t1, "predict_test_time": t3 - t2,
    }


def run_dataset(cfg: dict, paths, did: int) -> None:
    perf_cfg, budgets, seeds = cfg["perfdb"], cfg["budgets"], cfg["split"]["seeds"]
    classifiers = cfg["classifiers"]
    timeout = float(perf_cfg["timeout_s"])
    expo = perf_cfg["growth_exponent"]
    out = paths.perfdb / f"{int(did)}.csv"
    done = _done_keys(out)
    todo = len(seeds) * len(classifiers) * len(budgets) - len(done)
    if todo <= 0:
        log.info(f"[skip] {did}: complete")
        return
    data = load_dataset(paths, did)
    X, y_str = data["X"], data["y"]
    y = LabelEncoder().fit_transform(y_str)
    chance = 1.0 / len(np.unique(y))
    hw, ver = hardware_info(), package_versions()
    log.info(f"[run] {did} {data['name']}: {todo} fits to do")

    for seed in seeds:
        sp = load_row_split(paths, did, seed)
        X_tr_full, y_tr_full = X.iloc[sp["train"]], y[sp["train"]]
        X_va, y_va, X_te, y_te = X.iloc[sp["val"]], y[sp["val"]], X.iloc[sp["test"]], y[sp["test"]]
        for name in classifiers:
            prev_t, prev_f, timed_out = None, None, False
            for level, frac in enumerate(budgets):
                pos = budget_subsample(y_tr_full, frac, seed)
                base = {"did": did, "seed": seed, "clf": name, "budget": frac, "level": level,
                        "n_train_used": len(pos), "chance": chance, "hardware": hw, "versions": ver}
                if (seed, name, level) in done:
                    # recover timing for the extrapolation rule from the stored row
                    row = pd.read_csv(out).query("seed == @seed and clf == @name and level == @level").iloc[0]
                    timed_out = timed_out or str(row["status"]).startswith("timeout")
                    prev_t, prev_f = float(row["fit_time"] or 0) + float(row["predict_val_time"] or 0), frac
                    continue
                k = expo.get(name, expo["default"])
                predicted = prev_t * (frac / prev_f) ** k if prev_t is not None else 0.0
                if timed_out or predicted > timeout:
                    _append(out, {**base, "status": "timeout_predicted", "val_bacc": chance, "test_bacc": chance,
                                  "val_f1": np.nan, "test_f1": np.nan, "train_bacc": np.nan, "fit_time": np.nan,
                                  "predict_val_time": np.nan, "predict_test_time": np.nan, "cost_s": timeout,
                                  "message": f"predicted {predicted:.1f}s"})
                    timed_out = True
                    continue
                t_start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = run_one_fit(name, X_tr_full.iloc[pos], y_tr_full[pos], X_va, y_va, X_te, y_te,
                                        seed, perf_cfg, cfg["filters"]["max_categories"])
                    t = r["fit_time"] + r["predict_val_time"]
                    if t > timeout:
                        _append(out, {**base, **r, "status": "timeout", "val_bacc": chance, "test_bacc": chance,
                                      "cost_s": timeout, "message": f"took {t:.1f}s"})
                        timed_out = True
                    else:
                        _append(out, {**base, **r, "status": "ok", "cost_s": t, "message": ""})
                    prev_t, prev_f = t, frac
                except Exception as e:  # noqa: BLE001
                    t = time.perf_counter() - t_start
                    _append(out, {**base, "status": "error", "val_bacc": chance, "test_bacc": chance,
                                  "val_f1": np.nan, "test_f1": np.nan, "train_bacc": np.nan,
                                  "fit_time": t, "predict_val_time": 0.0, "predict_test_time": 0.0,
                                  "cost_s": min(t, timeout),
                                  "message": (type(e).__name__ + ": " + str(e))[:300].replace("\n", " ")})
                    log.warning(f"  error {name} b={frac} seed={seed}: {e}")
                    log.debug(traceback.format_exc())
    log.info(f"[done] {did}")


def warmup(classifiers, perf_cfg, max_categories=20) -> None:
    """Fit every classifier once on a tiny synthetic problem, untimed.

    First calls pay one-off costs (TabPFN weight download, CUDA initialisation, JIT/imports) that
    must not be charged to the first real dataset.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4, n_classes=3, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    for name in classifiers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_one_fit(name, X.iloc[:80], y[:80], X.iloc[80:100], y[80:100], X.iloc[100:], y[100:], 0, perf_cfg, max_categories)
        except Exception as e:  # noqa: BLE001
            log.warning(f"warm-up of {name} failed: {e}  (its real fits will be recorded as errors if this persists)")


def run_perfdb(cfg: dict, paths, dids: list[int], shard: tuple[int, int] | None = None) -> None:
    check_available(cfg["classifiers"])
    log.info(f"hardware: {hardware_info()}")
    warmup(cfg["classifiers"], cfg["perfdb"], cfg["filters"]["max_categories"])
    if shard is not None:
        i, n = shard
        dids = [d for j, d in enumerate(sorted(dids)) if j % n == i]
        log.info(f"shard {i}/{n}: {len(dids)} datasets")
    for did in dids:
        run_dataset(cfg, paths, did)
