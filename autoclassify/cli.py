"""Command-line entry point. Every stage reads and writes files, so any stage can be re-run.

    python -m autoclassify.cli <command> --config configs/stage1.yaml

Commands, in order:
    fetch      download + clean + filter datasets           -> data/<stage>/raw, metadata.csv
    split      dataset-level + row-level splits              -> dataset_split.json, splits/*.npz
    perfdb     fit every classifier x budget x seed          -> results/<stage>/perfdb/*.csv
    status     completeness / time-out report of perfdb
    features   dataset features for the RL state            -> results/<stage>/meta/
    train      Q-learning, DQN, Double DQN (all seeds)       -> results/<stage>/agents/full/
    evaluate   all methods on meta-val (or --split test)     -> results/<stage>/eval/
    ablate     ablation variants (trained + evaluated on meta-val)
    figures    all plots and tables from saved results       -> results/<stage>/figures/
    all        fetch -> split -> perfdb -> features -> train -> evaluate(val) -> figures
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import get_paths, load_config
from .utils import get_logger, load_json, save_json

log = get_logger()


def cmd_fetch(cfg, paths, args):
    from .data.sources import fetch_all
    fetch_all(cfg, paths)


def cmd_split(cfg, paths, args):
    from .data.sources import accepted_ids, load_dataset
    from .data.splits import make_meta_split, make_row_split, save_meta_split, save_row_split
    meta = pd.read_csv(paths.metadata_csv)
    split = save_meta_split(make_meta_split(meta, cfg["split"]["meta_fractions"], cfg["split"]["split_seed"]),
                            paths.dataset_split, overwrite=args.overwrite)
    log.info("dataset split: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()) + f" -> {paths.dataset_split}")
    for did in accepted_ids(paths):
        y = load_dataset(paths, did)["y"]
        for seed in cfg["split"]["seeds"]:
            f = paths.splits / f"{did}_s{seed}.npz"
            if not f.exists() or args.overwrite:
                save_row_split(make_row_split(y, seed, cfg["split"]["row_fractions"]), paths, did, seed)
    log.info(f"row splits written for {len(accepted_ids(paths))} datasets x {len(cfg['split']['seeds'])} seeds")


def _all_dids(paths, args):
    split = load_json(paths.dataset_split)
    dids = sorted(d for s in split.values() for d in s)
    if args.dids:
        dids = [d for d in dids if d in set(args.dids)]
    return dids


def cmd_perfdb(cfg, paths, args):
    from .perfdb.runner import run_perfdb
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    run_perfdb(cfg, paths, _all_dids(paths, args), shard)


def cmd_status(cfg, paths, args):
    from .perfdb.store import completeness, load_perfdb
    df = load_perfdb(paths)
    comp = completeness(df, cfg["classifiers"], cfg["budgets"], cfg["split"]["seeds"])
    print(comp.to_string())
    print(f"\n{comp['complete'].sum()}/{len(_all_dids(paths, args))} datasets complete")
    print("\nstatus counts per classifier:")
    print(df.pivot_table(index="clf", columns="status", values="did", aggfunc="count", fill_value=0).to_string())
    print("\nmedian seconds at 100% budget:")
    print(df[df["level"] == len(cfg["budgets"]) - 1].groupby("clf")["cost_s"].median().round(3).to_string())


def cmd_features(cfg, paths, args):
    """Dataset features for the RL state (no meta-learner) + baseline statistics."""
    from .meta.context import build_context
    from .meta.metafeatures import compute_all
    from .perfdb.store import build_tables, load_perfdb
    split = load_json(paths.dataset_split)
    dids = sorted(d for s in split.values() for d in s)
    mf = compute_all(cfg, paths, dids)
    tables = build_tables(load_perfdb(paths, dids), cfg["classifiers"], cfg["budgets"])
    build_context(cfg, paths, tables, mf, split, cfg["classifiers"])


def cmd_train(cfg, paths, args):
    from .pipeline import load_workspace, train_agent
    ws = load_workspace(cfg, paths)
    log.info(f"train instances: {len(ws.keys('train'))}, val instances: {len(ws.keys('val'))}")
    algos = args.algos or cfg["agents"]["algos"]
    seeds = args.seeds if args.seeds is not None else cfg["agents"]["seeds"]
    for algo in algos:
        for seed in seeds:
            log.info(f"training {algo} seed {seed}")
            train_agent(ws, algo, seed, "full", cfg["ablations"].get("full", {}))


def cmd_evaluate(cfg, paths, args):
    from .eval.stats import full_report
    from .pipeline import evaluate_all, load_workspace, record_test_opening
    ws = load_workspace(cfg, paths)
    if args.split == "test":
        record_test_opening(paths, args.force)
    res = evaluate_all(ws, args.split)
    rep = full_report(res, args.rl_method, cfg["eval"]["budgets"])
    _write_report(rep, paths, args.split)


def _write_report(rep, paths, split):
    rep["summary"].to_csv(paths.eval / f"{split}_summary.csv", index=False)
    rep["aurc"].to_csv(paths.eval / f"{split}_aurc.csv")
    lines = [f"# Results on meta-{split}\n", "## Summary (per method x budget)\n",
             rep["summary"].round(4).to_markdown(index=False) if _has_tabulate() else rep["summary"].round(4).to_string(index=False),
             "\n\n## Area under regret-vs-budget curve (lower is better)\n", rep["aurc"].round(4).to_string()]
    for b, t in rep["tests"].items():
        lines += [f"\n\n## Budget {b}\n", "Average ranks:\n", t["avg_ranks"].round(3).to_string(), f"\n\nFriedman/Nemenyi: {t['friedman']}"]
        if "wilcoxon" in t:
            lines += ["\n\nWilcoxon (Holm-corrected):\n", t["wilcoxon"].round(4).to_string(index=False)]
    (paths.eval / f"{split}_report.md").write_text("\n".join(lines))
    print(rep["summary"].round(4).to_string(index=False))
    log.info(f"report -> {paths.eval / f'{split}_report.md'}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def cmd_ablate(cfg, paths, args):
    from .eval.stats import per_dataset
    from .pipeline import load_workspace, run_ablations
    ws = load_workspace(cfg, paths)
    res = run_ablations(ws, args.names)
    g = per_dataset(res.assign(method=res["variant"]))
    tab = g.groupby(["method", "budget"])["regret"].mean().unstack()
    tab["mean"] = tab.mean(axis=1)
    tab.round(4).to_csv(paths.eval / "ablations_val_table.csv")
    print(tab.round(4).to_string())


def cmd_figures(cfg, paths, args):
    from .eval import plots
    from .eval.stats import average_ranks, friedman_nemenyi
    f = paths.eval / f"{args.split}_results.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing - run `evaluate` first")
    res = pd.read_csv(f)
    out = paths.figures
    plots.regret_vs_budget(res, out / f"{args.split}_regret_vs_budget.png")
    plots.cost_vs_regret(res, out / f"{args.split}_cost_vs_regret.png")
    for b in cfg["eval"]["budgets"]:
        mat, ranks = average_ranks(res, b)
        if mat.shape[0] >= 2 and mat.shape[1] >= 3:
            fr = friedman_nemenyi(mat)
            plots.cd_diagram(ranks, fr.get("nemenyi_cd", np.nan), out / f"{args.split}_cd_budget{b}.png",
                             f"Average rank, budget {b} (n={mat.shape[0]} datasets)")
    for m in ("ddqn", "dqn", "qlearning"):
        if m in set(res["method"]):
            plots.action_heatmap(res, m, cfg["classifiers"], cfg["budgets"], out / f"{args.split}_actions_{m}.png")
    rl = [m for m in ("ddqn", "dqn") if m in set(res["method"])][:1]
    for did in sorted(res["did"].unique())[:5]:
        plots.search_traces(res, rl + ["succ_halving", "avg_rank", "exhaustive"], did, 0.25,
                            out / f"{args.split}_trace_{did}.png")
    plots.training_curves(paths, "full", out / "training_curves.png")
    log.info(f"figures -> {out}")


def cmd_all(cfg, paths, args):
    for c in (cmd_fetch, cmd_split, cmd_perfdb, cmd_status, cmd_features, cmd_train, cmd_evaluate, cmd_figures):
        log.info(f"=== {c.__name__[4:]} ===")
        c(cfg, paths, args)


COMMANDS = {"fetch": cmd_fetch, "split": cmd_split, "perfdb": cmd_perfdb, "status": cmd_status, "features": cmd_features, "meta": cmd_features,
            "train": cmd_train, "evaluate": cmd_evaluate, "ablate": cmd_ablate, "figures": cmd_figures, "all": cmd_all}


def main(argv=None):
    p = argparse.ArgumentParser(prog="autoclassify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--config", required=True)
    p.add_argument("--root", default=None, help="project root (default: current directory)")
    p.add_argument("--dids", type=int, nargs="*", help="restrict to these dataset IDs")
    p.add_argument("--shard", default=None, help="i/n: run only datasets with index %% n == i (perfdb)")
    p.add_argument("--overwrite", action="store_true", help="split: regenerate (only before any results exist!)")
    p.add_argument("--algos", nargs="*", help="train: subset of qlearning dqn ddqn")
    p.add_argument("--seeds", type=int, nargs="*", help="train: subset of agent seeds")
    p.add_argument("--names", nargs="*", help="ablate: subset of ablation names")
    p.add_argument("--split", default="val", choices=["val", "test"], help="evaluate/figures: meta-val or meta-test")
    p.add_argument("--rl-method", default="ddqn", help="RL method compared in Wilcoxon tests")
    p.add_argument("--force", action="store_true", help="evaluate --split test: allow re-opening meta-test")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg, args.root)
    try:
        COMMANDS[args.command](cfg, paths, args)
    except (RuntimeError, FileNotFoundError, ImportError, ValueError) as e:
        log.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
