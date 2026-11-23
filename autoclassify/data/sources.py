"""Dataset acquisition and filtering.

Sources
-------
* ``openml``    : real datasets, downloaded with scikit-learn's ``fetch_openml`` (cached on disk).
                  The suite listing (OpenML-CC18) and the optional catalogue search use the public
                  OpenML REST API (https://www.openml.org/api/v1/json/...).
* ``synthetic`` : generated with ``make_classification``. ONLY for testing the pipeline offline.
                  Never report synthetic results.

Every accepted dataset is cleaned the same way and saved as ``data/<stage>/raw/<did>.pkl``
(a dict with X: DataFrame, y: Series of str labels, name). A ``metadata.csv`` row records what
was kept, dropped or rejected, and why.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger()

OPENML_API = "https://www.openml.org/api/v1/json"
SUITE_IDS = {"OpenML-CC18": 99, "OpenML-CC21": 225}  # study IDs on OpenML


# ---------------------------------------------------------------------------------------------
# OpenML
# ---------------------------------------------------------------------------------------------
def _get_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https host)
        return json.loads(r.read().decode())


def list_suite_dataset_ids(suite: str) -> list[int]:
    """Dataset IDs of an OpenML benchmark suite, e.g. 'OpenML-CC18'."""
    study_id = SUITE_IDS.get(suite, suite)
    try:
        data = _get_json(f"{OPENML_API}/study/{study_id}")
        ids = data["study"]["data"]["data_id"]
    except Exception as e:  # noqa: BLE001 - fall back to the official client if the REST format differs
        log.warning(f"REST suite listing failed ({e}); trying the openml package")
        import openml
        ids = openml.study.get_suite(suite).data
    return sorted(int(i) for i in ids)


def search_openml_catalogue(filters: dict, limit: int = 3000) -> pd.DataFrame:
    """Active OpenML datasets whose size passes the filters (one row per dataset *name*)."""
    f = filters
    url = (f"{OPENML_API}/data/list/status/active"
           f"/number_instances/{f['min_rows']}..{10 * f['max_rows']}"
           f"/number_features/2..{f['max_features_encoded']}"
           f"/number_classes/2..{f['max_classes']}/limit/{limit}")
    rows = []
    for d in _get_json(url)["data"]["dataset"]:
        q = {x["name"]: x.get("value") for x in d.get("quality", [])}
        rows.append({"did": int(d["did"]), "name": d["name"], "version": int(d.get("version", 1)),
                     "n_rows": float(q.get("NumberOfInstances", np.nan)),
                     "n_features": float(q.get("NumberOfFeatures", np.nan)),
                     "n_classes": float(q.get("NumberOfClasses", np.nan))})
    df = pd.DataFrame(rows)
    # OpenML keeps many versions/copies of one dataset: keep the lowest version per name.
    df = df.sort_values(["name", "version", "did"]).drop_duplicates("name", keep="first")
    return df.reset_index(drop=True)


def load_openml(did: int, cache_dir) -> tuple[pd.DataFrame, pd.Series, str]:
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=int(did), as_frame=True, parser="auto", data_home=str(cache_dir))
    y = bunch.target
    if y is None or isinstance(y, pd.DataFrame):
        raise ValueError("no single default target")
    name = bunch.details.get("name", str(did)) if hasattr(bunch, "details") else str(did)
    return bunch.data, y, name


# ---------------------------------------------------------------------------------------------
# Synthetic (offline pipeline test only)
# ---------------------------------------------------------------------------------------------
def make_synthetic(i: int, seed: int) -> tuple[pd.DataFrame, pd.Series, str]:
    from sklearn.datasets import make_classification
    rng = np.random.default_rng(seed * 1000 + i)
    n = int(rng.integers(500, 3000))
    d = int(rng.integers(4, 40))
    c = int(rng.integers(2, 6))
    inf = max(2, int(d * rng.uniform(0.2, 0.9)))
    red = int(min(d - inf, rng.integers(0, 5)))
    w = None
    if rng.random() < 0.4:
        p = rng.dirichlet(np.ones(c) * 0.8)
        w = list(np.clip(p, 0.05, None) / np.clip(p, 0.05, None).sum())
    X, y = make_classification(n_samples=n, n_features=d, n_informative=inf, n_redundant=red,
                               n_classes=c, n_clusters_per_class=int(rng.integers(1, 3)),
                               weights=w, class_sep=float(rng.uniform(0.4, 1.8)),
                               flip_y=float(rng.uniform(0, 0.08)), random_state=int(rng.integers(1e9)))
    X = pd.DataFrame(X, columns=[f"f{j}" for j in range(d)])
    for j in range(int(rng.integers(0, 3))):              # a few categorical columns
        col = f"f{j}"
        X[col] = pd.cut(X[col], bins=int(rng.integers(3, 8)), labels=False).astype(str)
        X[col] = "c" + X[col]
    if rng.random() < 0.5:                                 # some missing values
        mask = rng.random(X.shape) < 0.03
        X = X.mask(mask)
    return X, pd.Series(y.astype(str), name="target"), f"synthetic_{i}"


# ---------------------------------------------------------------------------------------------
# Cleaning + filters (identical for every source)
# ---------------------------------------------------------------------------------------------
def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns
            if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object
            or pd.api.types.is_bool_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])]


def clean_dataset(X: pd.DataFrame, y: pd.Series, filters: dict, seed: int = 0):
    """Return (X, y, info). Raises ValueError with a reason if the dataset is rejected."""
    info: dict = {}
    X = X.copy()
    y = y.astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)
    keep = y.notna() & (y != "nan")
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    cats = categorical_columns(X)
    for c in X.columns:
        if c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), None)
            X[c] = X[c].map(lambda v: None if v is None else str(v)).astype(object)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    drop = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]           # constant / empty
    drop += [c for c in cats if c not in drop and X[c].nunique() > 0.9 * len(X)]  # ID-like text
    X = X.drop(columns=drop)
    info["dropped_columns"] = len(drop)
    if X.shape[1] == 0:
        raise ValueError("no usable features")

    counts = y.value_counts()
    rare = counts[counts < filters["min_class_count"]].index
    n_before = len(y)
    if len(rare):
        m = ~y.isin(rare)
        X, y = X[m].reset_index(drop=True), y[m].reset_index(drop=True)
    info["dropped_rare_class_rows"] = int(n_before - len(y))
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError("fewer than 2 classes after dropping rare classes")
    if n_classes > filters["max_classes"]:
        raise ValueError(f"{n_classes} classes > max_classes")
    if len(y) < filters["min_rows"]:
        raise ValueError(f"{len(y)} rows < min_rows")

    info["n_rows_original"] = int(len(y))
    if len(y) > filters["max_rows"]:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(y)), train_size=filters["max_rows"], stratify=y, random_state=seed)
        idx = np.sort(idx)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    cats = categorical_columns(X)
    n_enc = (X.shape[1] - len(cats)) + sum(min(X[c].nunique(), filters["max_categories"]) for c in cats)
    if n_enc > filters["max_features_encoded"]:
        raise ValueError(f"{n_enc} encoded features > max_features_encoded")
    info.update(n_rows=int(len(y)), n_features=int(X.shape[1]), n_categorical=len(cats),
                n_numeric=int(X.shape[1] - len(cats)), n_features_encoded=int(n_enc),
                n_classes=int(n_classes), min_class_count=int(y.value_counts().min()),
                missing_ratio=float(X.isna().to_numpy().mean()))
    return X, y, info


def fetch_all(cfg: dict, paths) -> pd.DataFrame:
    """Download, clean, filter and cache every configured dataset. Writes metadata.csv."""
    dcfg, filters = cfg["datasets"], cfg["filters"]
    jobs: list[tuple[str, int]] = []
    if dcfg["source"] == "synthetic":
        jobs = [("synthetic", i) for i in range(dcfg["synthetic"]["n"])]
    else:
        ids = list(dcfg.get("ids") or [])
        if dcfg.get("suite"):
            suite_ids = list_suite_dataset_ids(dcfg["suite"])
            log.info(f"suite {dcfg['suite']}: {len(suite_ids)} datasets")
            ids += suite_ids
        if dcfg.get("n_extra", 0) > 0:
            cand = search_openml_catalogue(filters)
            cand = cand[~cand["did"].isin(ids)]
            cand.to_csv(paths.candidates_csv, index=False)
            pick = cand.sample(n=min(len(cand), 3 * dcfg["n_extra"]), random_state=dcfg["extra_seed"])
            ids += pick["did"].tolist()  # oversample: some will fail the filters
            log.info(f"catalogue search: {len(cand)} candidates, trying {len(pick)}; see {paths.candidates_csv}")
        ids = [i for i in dict.fromkeys(ids) if i not in set(dcfg.get("exclude_ids") or [])]
        if dcfg.get("max_datasets"):
            # Stage 2: a reproducible random subset of the suite, always keeping explicit IDs.
            # Oversample x1.5 because some datasets fail the filters.
            rest = [i for i in ids if i not in set(dcfg.get("ids") or [])]
            rng = np.random.default_rng(dcfg.get("sample_seed", 0))
            n_more = max(0, int(1.5 * dcfg["max_datasets"]) - len(dcfg.get("ids") or []))
            ids = list(dcfg.get("ids") or []) + list(rng.permutation(rest)[:n_more])
        jobs = [("openml", int(i)) for i in ids]

    old = pd.read_csv(paths.metadata_csv) if paths.metadata_csv.exists() else pd.DataFrame()
    rows, n_extra_ok = [], 0
    explicit = set(dcfg.get("ids") or [])
    for source, key in jobs:
        did = key if source == "openml" else 900000 + key
        out = paths.raw / f"{did}.pkl"
        if out.exists() and len(old) and did in set(old["did"]):
            rows.append(old[old["did"] == did].iloc[0].to_dict())
            continue
        try:
            if source == "openml":
                X, y, name = load_openml(did, paths.cache)
            else:
                X, y, name = make_synthetic(key, dcfg["synthetic"]["seed"])
            X, y, info = clean_dataset(X, y, filters)
            pd.to_pickle({"X": X, "y": y, "name": name, "did": did}, out)
            rows.append({"did": did, "name": name, "status": "ok", "reason": "", **info})
            log.info(f"[ok] {did} {name}: {info['n_rows']} rows, {info['n_features']} feats, {info['n_classes']} classes")
        except Exception as e:  # noqa: BLE001 - every rejection is logged, never silent
            rows.append({"did": did, "name": "", "status": "rejected", "reason": str(e)[:200]})
            log.info(f"[rejected] {did}: {e}")
    meta = pd.DataFrame(rows)
    if dcfg.get("n_extra", 0) > 0 and len(meta):
        # keep explicit + suite datasets, then the first n_extra accepted extras
        suite_ids = set(list_suite_dataset_ids(dcfg["suite"])) if dcfg.get("suite") else set()
        is_core = meta["did"].isin(explicit | suite_ids)
        extra_ok = meta[~is_core & (meta["status"] == "ok")].index[: dcfg["n_extra"]]
        drop = meta[~is_core & (meta["status"] == "ok")].index.difference(extra_ok)
        meta.loc[drop, ["status", "reason"]] = ["rejected", "not needed (n_extra reached)"]
        n_extra_ok = len(extra_ok)
    if dcfg.get("max_datasets") and len(meta):
        ok = meta.index[meta["status"] == "ok"]
        surplus = ok[dcfg["max_datasets"]:]
        meta.loc[surplus, ["status", "reason"]] = ["rejected", "not needed (max_datasets reached)"]
    meta.to_csv(paths.metadata_csv, index=False)
    log.info(f"{(meta['status'] == 'ok').sum()} datasets accepted ({n_extra_ok} extras), metadata -> {paths.metadata_csv}")
    return meta


def load_dataset(paths, did: int) -> dict:
    return pd.read_pickle(paths.raw / f"{int(did)}.pkl")


def accepted_ids(paths) -> list[int]:
    meta = pd.read_csv(paths.metadata_csv)
    return sorted(int(d) for d in meta.loc[meta["status"] == "ok", "did"])
