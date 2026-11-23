"""Figures for the report. All are produced from saved CSVs, never from in-memory state."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .stats import per_dataset  # noqa: E402

RL = {"qlearning", "dqn", "ddqn"}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def regret_vs_budget(res: pd.DataFrame, path) -> None:
    g = per_dataset(res)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    aware = g[g["budget"] >= 0]
    for m, d in aware.groupby("method"):
        s = d.groupby("budget")["regret"].agg(["mean", "sem"]).reset_index()
        lw = 2.4 if m in RL else 1.2
        ax.errorbar(s["budget"], s["mean"], yerr=1.96 * s["sem"].fillna(0), label=m, lw=lw, marker="o", ms=3, capsize=2)
    for m, d in g[g["budget"] < 0].groupby("method"):
        ax.scatter(d["budget_used"].mean(), d["regret"].mean(), marker="s", s=40, label=f"{m} (one-shot)", zorder=5)
    ax.set_xscale("log")
    _style(ax, "Normalised regret vs budget (mean, 95% CI over datasets)", "budget (fraction of exhaustive cost)", "normalised regret (lower is better)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cost_vs_regret(res: pd.DataFrame, path) -> None:
    g = per_dataset(res)
    s = g.groupby("method")[["budget_used", "regret"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in s.itertuples():
        ax.scatter(r.budget_used, r.regret, s=50 if r.method in RL else 25)
        ax.annotate(r.method, (r.budget_used, r.regret), fontsize=7, xytext=(4, 3), textcoords="offset points")
    _style(ax, "Average budget used vs regret", "mean budget used (fraction of exhaustive)", "mean normalised regret")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cd_diagram(avg_ranks: pd.Series, cd: float, path, title: str) -> None:
    """Critical-difference diagram (Demsar 2006): methods whose ranks differ by < CD are joined."""
    names, ranks = list(avg_ranks.index), avg_ranks.to_numpy()
    k = len(names)
    order = np.argsort(ranks)
    sr = ranks[order]
    # maximal groups of methods whose rank spread is below CD (not significantly different)
    cliques, last_j = [], -1
    for i in range(k):
        j = i
        while j + 1 < k and sr[j + 1] - sr[i] < cd:
            j += 1
        if j > i and j > last_j:
            cliques.append((i, j))
            last_j = j
    half = (k + 1) // 2
    top_lab = -0.35 - 0.18 * len(cliques)
    fig, ax = plt.subplots(figsize=(8, 0.32 * half + 0.25 * len(cliques) + 1.4))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(top_lab - 0.4 * half - 0.2, 0.9)
    ax.hlines(0, 1, k, color="black", lw=1)
    for i in range(1, k + 1):
        ax.vlines(i, -0.06, 0.06, color="black")
        ax.text(i, 0.15, str(i), ha="center", fontsize=8)
    for c, (i, j) in enumerate(cliques):
        y = -0.25 - 0.18 * c
        ax.hlines(y, sr[i] - 0.05, sr[j] + 0.05, lw=3, color="tab:red", alpha=0.7)
    for j, idx in enumerate(order):
        left = j < half
        y = top_lab - 0.4 * (j if left else j - half)
        xr = ranks[idx]
        xt = 0.6 if left else k + 0.4
        ax.plot([xr, xr, xt], [0, y, y], color="0.4", lw=0.7)
        ax.text(xt, y, f" {names[idx]} ({xr:.2f}) ", ha="right" if left else "left", va="center", fontsize=8)
    if np.isfinite(cd):
        ax.hlines(0.6, 1, min(1 + cd, k), lw=2, color="black")
        ax.text(1 + min(cd, k - 1) / 2, 0.68, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def action_heatmap(res: pd.DataFrame, method: str, classifiers, budgets, path) -> None:
    d = res[res["method"] == method]
    M = np.zeros((len(classifiers), len(budgets)))
    for t in d["trace"]:
        for step in json.loads(t):
            if step["clf"] in classifiers and step["budget"] in budgets:
                M[classifiers.index(step["clf"]), budgets.index(step["budget"])] += 1
    M = M / max(1, len(d))
    fig, ax = plt.subplots(figsize=(5, 0.35 * len(classifiers) + 1.5))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(classifiers)), classifiers, fontsize=8)
    ax.set_xticks(range(len(budgets)), [f"{int(b * 100)}%" for b in budgets])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > M.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="evaluations per episode")
    ax.set_title(f"What {method} evaluates", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def search_traces(res: pd.DataFrame, methods, did, budget, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in methods:
        d = res[(res["method"] == m) & (res["did"] == did) & ((res["budget"] == budget) | res["budget"].isna())]
        if d.empty:
            continue
        tr = [s for s in json.loads(d.iloc[0]["trace"]) if not s["cutoff"]]
        if not tr:
            continue
        x = [0] + [s["used"] for s in tr]
        y = [np.nan] + [s["best_val"] for s in tr]
        ax.step(x, y, where="post", label=m, lw=2 if m in RL else 1)
        for s in tr:
            ax.annotate(f"{s['clf']}@{int(s['budget'] * 100)}", (s["used"], s["best_val"]), fontsize=6, alpha=0.7)
    _style(ax, f"Search trace, dataset {did}, budget {budget}", "budget used", "best validation balanced accuracy so far")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def training_curves(paths, variant: str, path) -> None:
    files = sorted((paths.agents / variant).glob("*_history.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for f in files:
        h = pd.read_csv(f)
        x = h["step"] if "step" in h else h["episode"]
        ax.plot(x / x.max(), h["val_regret"], label=f.stem.replace("_history", ""), lw=1)
    _style(ax, "Meta-val regret during training", "fraction of training", "mean meta-val regret")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
