"""Glue: load everything needed for RL (tables + context), train agents, evaluate methods."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agents.dqn import DQNAgent, train_dqn
from .agents.mlp import MLP
from .agents.qlearning import QLearningAgent, train_qlearning
from .baselines.policies import AgentPolicy, QPolicy, make_baselines
from .env.autoclassify_env import AutoClassifyEnv
from .meta.context import load_baseline_stats, load_context
from .perfdb.store import build_tables, load_perfdb
from .utils import get_logger, load_json, save_json, set_seed

log = get_logger()


@dataclass
class Workspace:
    cfg: dict
    paths: object
    tables: dict
    context: dict
    split: dict
    stats: dict

    @property
    def classifiers(self):
        return self.cfg["classifiers"]

    def keys(self, split_name: str) -> list:
        return sorted(k for k, v in self.context.items() if v["split"] == split_name and k in self.tables)

    def make_env(self, split_name: str, quality: str, variant: dict | None = None, seed: int = 0) -> AutoClassifyEnv:
        variant = variant or {}
        return AutoClassifyEnv(self.tables, self.context, self.classifiers, self.cfg["budgets"],
                               self.keys(split_name), episode_budgets=self.cfg["env"]["episode_budgets"],
                               lam=variant.get("lam", self.cfg["env"]["lam"]), quality=quality,
                               flags=variant.get("flags"), allowed_levels=variant.get("allowed_levels"), seed=seed,
                               cost_profile=self.stats["avg_cost"])


def load_workspace(cfg: dict, paths) -> Workspace:
    split = load_json(paths.dataset_split)
    df = load_perfdb(paths, dids=[d for s in split.values() for d in s])
    tables = build_tables(df, cfg["classifiers"], cfg["budgets"])
    return Workspace(cfg, paths, tables, load_context(paths), split, load_baseline_stats(paths))


# ------------------------------------------------------------------------------------------ evaluation
def run_episode(env, policy, instance, budget) -> dict:
    obs, info = env.reset(options={"instance": instance, "budget": budget})
    policy.reset(env)
    while True:
        a = policy.act(env, obs, info)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            return info


def evaluate_policies(ws: Workspace, policies: list, split_name: str, budgets=None, repeats: dict | None = None,
                      variant: dict | None = None) -> pd.DataFrame:
    budgets = budgets or ws.cfg["eval"]["budgets"]
    env = ws.make_env(split_name, quality="test", variant=variant)
    rows = []
    for pol in policies:
        reps = (repeats or {}).get(pol.name, 1)
        blist = budgets if pol.budget_aware else [None]
        for key in ws.keys(split_name):
            for B in blist:
                for rep in range(reps):
                    info = run_episode(env, pol, key, B)
                    rows.append({"method": pol.name, "did": key[0], "seed": key[1],
                                 "budget": np.nan if B is None else B, "rep": rep,
                                 "budget_aware": pol.budget_aware, "selected": info["selected"],
                                 "regret": info["regret_test"], "regret_val": info["regret_val"],
                                 "test_score": info["test_score"], "best_test": info["best_test"],
                                 "budget_used": info["budget_used"], "raw_seconds": info["raw_seconds"],
                                 "n_evals": info["n_evals"], "trace": json.dumps(info["trace"])})
    return pd.DataFrame(rows)


def mean_val_regret(ws: Workspace, policy, variant=None) -> float:
    df = evaluate_policies(ws, [policy], "val", budgets=ws.cfg["env"]["episode_budgets"], variant=variant)
    return float(df["regret"].mean())


# ------------------------------------------------------------------------------------------ training
def train_agent(ws: Workspace, algo: str, seed: int, variant_name: str = "full", variant: dict | None = None):
    """Train one agent on meta-train (val-based reward); select the checkpoint on meta-val."""
    set_seed(seed)
    variant = variant or {}
    env = ws.make_env("train", quality="val", variant=variant, seed=seed)
    out = ws.paths.agents / variant_name
    out.mkdir(parents=True, exist_ok=True)
    acfg = ws.cfg["agents"]
    if algo == "qlearning":
        agent = QLearningAgent(env.action_space.n, acfg["qlearning"], seed)
        bucket = acfg["qlearning"].get("use_data_bucket", True)
        ev = lambda ag: mean_val_regret(ws, QPolicy(ag, "qlearning", bucket), variant)  # noqa: E731
        hist, best_snap = train_qlearning(env, agent, ev, log)
        if best_snap is not None:
            agent.restore(best_snap)
        agent.save(out / f"qlearning_seed{seed}.pkl")
    else:
        agent = DQNAgent(env.obs_dim, env.action_space.n, acfg["dqn"], double=(algo == "ddqn"), seed=seed)
        ev = lambda ag: mean_val_regret(ws, AgentPolicy(ag, algo), variant)  # noqa: E731
        hist, best = train_dqn(env, agent, ev, log)
        agent.q.set_params(best)
        agent.q.save(out / f"{algo}_seed{seed}.npz")
    pd.DataFrame(hist).to_csv(out / f"{algo}_seed{seed}_history.csv", index=False)
    return agent


def load_agent_policy(ws: Workspace, algo: str, seed: int, variant_name: str = "full"):
    d = ws.paths.agents / variant_name
    acfg = ws.cfg["agents"]
    n_actions = len(ws.classifiers) * len(ws.cfg["budgets"]) + 1
    if algo == "qlearning":
        ag = QLearningAgent.load(d / f"qlearning_seed{seed}.pkl", n_actions, acfg["qlearning"])
        return QPolicy(ag, "qlearning", acfg["qlearning"].get("use_data_bucket", True))
    ag = DQNAgent.__new__(DQNAgent)
    ag.q, ag.n_actions, ag.rng, ag.double = MLP.load(d / f"{algo}_seed{seed}.npz"), n_actions, np.random.default_rng(0), algo == "ddqn"
    return AgentPolicy(ag, algo)


def evaluate_all(ws: Workspace, split_name: str, variant_name: str = "full") -> pd.DataFrame:
    """Baselines + every trained agent (all seeds) on one split; saves results/eval/<split>_results.csv."""
    baselines = make_baselines(ws.classifiers, ws.stats["avg_rank_order"])
    reps = {"random": ws.cfg["eval"]["random_repeats"]}
    frames = [evaluate_policies(ws, baselines, split_name, repeats=reps)]
    variant = ws.cfg["ablations"].get(variant_name, {})
    for algo in ws.cfg["agents"]["algos"]:
        for seed in ws.cfg["agents"]["seeds"]:
            try:
                pol = load_agent_policy(ws, algo, seed, variant_name)
            except FileNotFoundError:
                log.warning(f"no trained {algo} seed {seed} ({variant_name}); skipped")
                continue
            df = evaluate_policies(ws, [pol], split_name, variant=variant)
            df["rep"] = seed
            frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(ws.paths.eval / f"{split_name}_results.csv", index=False)
    return res


def run_ablations(ws: Workspace, names=None, split_name: str = "val") -> pd.DataFrame:
    algo = ws.cfg["eval"]["rl_algo_for_ablations"]
    frames = []
    for name, variant in ws.cfg["ablations"].items():
        if names and name not in names:
            continue
        for seed in ws.cfg["agents"]["seeds"]:
            log.info(f"ablation {name}: training {algo} seed {seed}")
            train_agent(ws, algo, seed, f"ablation_{name}", variant)
            pol = load_agent_policy(ws, algo, seed, f"ablation_{name}")
            pol.name = f"{algo}[{name}]"
            df = evaluate_policies(ws, [pol], split_name, variant=variant)
            df["rep"], df["variant"] = seed, name
            frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(ws.paths.eval / f"ablations_{split_name}.csv", index=False)
    return res


def record_test_opening(paths, force: bool) -> None:
    """Meta-test may be evaluated once. A marker file makes a second opening deliberate."""
    marker = paths.eval / "TEST_OPENED.json"
    if marker.exists() and not force:
        raise RuntimeError(f"meta-test was already evaluated ({marker}). Re-running on test after looking at "
                           "results is a form of test-set tuning. Use --force only if you have a documented reason.")
    save_json({"opened": pd.Timestamp.now().isoformat()}, marker)
