"""Summary tables and statistical tests.

* summary: per method x budget - mean/median normalised regret with a bootstrap 95% CI over
  datasets, mean test balanced accuracy, mean budget used, real seconds, top-1 hit rate.
* average rank of methods (rank 1 = lowest regret) per budget.
* area under the regret-vs-budget curve (anytime metric, like ALC in the MetaLC challenge).
* Friedman test + Nemenyi critical difference (Demsar, JMLR 2006); Wilcoxon signed-rank of the
  RL method vs each baseline with Holm correction.
Datasets are the unit of analysis: seeds/repeats are averaged within a dataset first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def per_dataset(res: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds and repeats -> one row per (method, dataset, budget)."""
    r = res.copy()
    r["budget"] = r["budget"].fillna(-1.0)  # -1 = unlimited (budget-agnostic methods)
    r["hit"] = (r["regret"] <= 1e-12).astype(float)
    g = r.groupby(["method", "did", "budget"], as_index=False)[
        ["regret", "regret_val", "test_score", "budget_used", "raw_seconds", "n_evals", "hit"]].mean()
    return g


def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(res: pd.DataFrame) -> pd.DataFrame:
    g = per_dataset(res)
    rows = []
    for (m, b), d in g.groupby(["method", "budget"]):
        lo, hi = _boot_ci(d["regret"])
        rows.append({"method": m, "budget": "unlimited" if b < 0 else b, "n_datasets": len(d),
                     "regret_mean": d["regret"].mean(), "regret_ci_low": lo, "regret_ci_high": hi,
                     "regret_median": d["regret"].median(), "test_bacc_mean": d["test_score"].mean(),
                     "budget_used_mean": d["budget_used"].mean(), "seconds_mean": d["raw_seconds"].mean(),
                     "n_evals_mean": d["n_evals"].mean(), "top1_hit_rate": d["hit"].mean()})
    return pd.DataFrame(rows).sort_values(["budget", "regret_mean"], key=lambda s: s.astype(str))


def average_ranks(res: pd.DataFrame, budget: float, include_unlimited: bool = True) -> pd.DataFrame:
    """Matrix datasets x methods of regret at `budget` (+ unlimited methods), and average ranks."""
    g = per_dataset(res)
    sel = g[(g["budget"] == budget) | ((g["budget"] < 0) & include_unlimited)]
    mat = sel.pivot_table(index="did", columns="method", values="regret").dropna(axis=0)
    ranks = mat.rank(axis=1, method="average")
    return mat, ranks.mean().sort_values()


def aurc(res: pd.DataFrame) -> pd.DataFrame:
    """Area under the regret-vs-budget curve per dataset (budget-aware methods), normalised by the budget range."""
    g = per_dataset(res)
    g = g[g["budget"] >= 0]
    out = []
    for (m, did), d in g.groupby(["method", "did"]):
        d = d.sort_values("budget")
        if len(d) < 2:
            continue
        x, y = d["budget"].to_numpy(), d["regret"].to_numpy()
        trap = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
        out.append({"method": m, "did": did, "aurc": float(trap(y, x) / (x[-1] - x[0]))})
    a = pd.DataFrame(out)
    return a.groupby("method", as_index=False)["aurc"].agg(["mean", "std", "count"]).sort_values("mean")


def friedman_nemenyi(mat: pd.DataFrame, alpha: float = 0.05) -> dict:
    k, n = mat.shape[1], mat.shape[0]
    out = {"n_datasets": n, "n_methods": k}
    if n < 2 or k < 3:
        return {**out, "note": "too few datasets/methods for Friedman"}
    stat, p = stats.friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {**out, "friedman_stat": float(stat), "friedman_p": float(p), "nemenyi_cd": float(cd), "alpha": alpha}


def wilcoxon_vs(mat: pd.DataFrame, method: str) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank of `method` vs every other column, Holm-corrected."""
    rows = []
    for c in mat.columns:
        if c == method:
            continue
        diff = mat[method] - mat[c]
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(mat[method], mat[c], zero_method="zsplit").pvalue)
        rows.append({"vs": c, "mean_diff": float(diff.mean()), "wins": int((diff < 0).sum()),
                     "ties": int((diff == 0).sum()), "losses": int((diff > 0).sum()), "p": p})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df)
    adj, running = [], 0.0
    for i, p in enumerate(df["p"]):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    df["p_holm"] = adj
    return df


def full_report(res: pd.DataFrame, rl_method: str, budgets) -> dict:
    rep = {"summary": summary(res), "aurc": aurc(res), "tests": {}}
    for b in budgets:
        mat, ranks = average_ranks(res, b)
        if mat.shape[0] == 0:
            continue
        entry = {"avg_ranks": ranks, "friedman": friedman_nemenyi(mat)}
        if rl_method in mat.columns and mat.shape[0] >= 2:
            entry["wilcoxon"] = wilcoxon_vs(mat, rl_method)
        rep["tests"][b] = entry
    return rep
