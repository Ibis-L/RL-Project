"""Baseline and RL policies. Every method runs through the same environment, budget,
selection rule and lookup database.

Policies expose:  reset(env)  and  act(env, obs, info) -> action.
`budget_aware = False` marks one-shot methods (fixed RF, exhaustive): they are evaluated once
with an unlimited budget and reported with the cost they actually used.
Budget-aware baselines use only an ESTIMATE (`env.predicted_cost` = meta-train average cost of the
classifier x budget fraction), never the true cost, to decide whether an evaluation fits; a wrong
estimate is cut off by the environment, exactly as for the RL agent.
Classifier order for the ordered baselines = average rank on meta-train (a plain statistic, no model).
"""
from __future__ import annotations

import math

import numpy as np


class Policy:
    name = "policy"
    budget_aware = True

    def reset(self, env):  # noqa: D401
        pass

    def act(self, env, obs, info) -> int:
        raise NotImplementedError


def _fits(env, k, l) -> bool:
    return env.predicted_cost(k, l) <= env.remaining + 1e-12


class OrderedFull(Policy):
    """Evaluate classifiers at 100% in a given order while the predicted cost fits."""

    def __init__(self, name, order_fn):
        self.name, self.order_fn = name, order_fn

    def reset(self, env):
        self.queue = list(self.order_fn(env))

    def act(self, env, obs, info):
        L = env.L - 1
        while self.queue:
            k = self.queue.pop(0)
            if info["action_mask"][env.action(k, L)] and _fits(env, k, L):
                return env.action(k, L)
        return env.STOP


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return OrderedFull("random", lambda env: rng.permutation(env.K))


def avg_rank_policy(order):
    return OrderedFull("avg_rank", lambda env: order)


class OneShot(Policy):
    """Evaluate a fixed list of classifiers at 100%, then stop (budget-agnostic)."""
    budget_aware = False

    def __init__(self, name, pick_fn):
        self.name, self.pick_fn = name, pick_fn

    def reset(self, env):
        self.queue = list(self.pick_fn(env))

    def act(self, env, obs, info):
        while self.queue:
            k = self.queue.pop(0)
            a = env.action(k, env.L - 1)
            if info["action_mask"][a]:
                return a
        return env.STOP


def fixed_policy(clf_name):
    return OneShot(f"fixed_{clf_name}", lambda env: [env.classifiers.index(clf_name)])


def exhaustive_policy():
    return OneShot("exhaustive", lambda env: range(env.K))


class BestOnSmall(Policy):
    """All classifiers at the smallest budget, then the winner at 100% (Nguyen et al. baseline)."""
    name = "best_on_small"

    def __init__(self, order=None):
        self.order = order

    def reset(self, env):
        self.phase1 = list(self.order if self.order is not None else range(env.K))
        self.final_done = False

    def act(self, env, obs, info):
        while self.phase1:
            k = self.phase1.pop(0)
            if _fits(env, k, 0):
                return env.action(k, 0)
            self.phase1 = []
        if not self.final_done and not np.all(np.isnan(env.obs_score)):
            self.final_done = True
            k = int(np.nanargmax(env.obs_score))
            a = env.action(k, env.L - 1)
            if info["action_mask"][a] and _fits(env, k, env.L - 1):
                return a
        return env.STOP


class SuccessiveHalving(Policy):
    """Rung r evaluates the survivors at budget level r and keeps the top half (eta = 2).

    With 12 classifiers and 4 levels: 12 @ 10%, 6 @ 25%, 3 @ 50%, 2 @ 100%.
    If a candidate does not fit the remaining budget it is dropped (order = average rank).
    """
    name = "succ_halving"

    def __init__(self, order=None):
        self.order = order

    def reset(self, env):
        self.cands = list(self.order if self.order is not None else range(env.K))
        self.rung, self.queue = 0, list(self.cands)

    def act(self, env, obs, info):
        while True:
            if self.queue:
                k = self.queue.pop(0)
                a = env.action(k, self.rung)
                if not info["action_mask"][a]:
                    continue
                if not _fits(env, k, self.rung):
                    self.cands.remove(k)     # cannot afford it at this rung: drop it
                    continue
                return a
            # survivors = candidates actually evaluated at this rung
            self.cands = [k for k in self.cands if env.level[k] >= self.rung]
            if self.rung >= env.L - 1 or not self.cands:
                return env.STOP
            scores = env.obs_score[self.cands]
            keep = max(1, math.ceil(len(self.cands) / 2))
            self.cands = [self.cands[i] for i in np.argsort(-np.nan_to_num(scores, nan=-1))[:keep]]
            self.rung += 1
            self.queue = list(self.cands)


class AgentPolicy(Policy):
    """Greedy (masked argmax) policy of a trained DQN / Double DQN agent."""

    def __init__(self, agent, name):
        self.agent, self.name = agent, name

    def act(self, env, obs, info):
        return self.agent.act(obs, info["action_mask"], 0.0)


class QPolicy(Policy):
    def __init__(self, agent, name="qlearning", use_data_bucket=True):
        self.agent, self.name, self.bucket = agent, name, use_data_bucket

    def act(self, env, obs, info):
        return self.agent.act(env.discrete_state(self.bucket), info["action_mask"], 0.0)


def make_baselines(classifiers, avg_rank_order, seed=0) -> list[Policy]:
    pols = [random_policy(seed)]
    if "rf" in classifiers:
        pols.append(fixed_policy("rf"))
    pols += [exhaustive_policy(), avg_rank_policy(avg_rank_order), BestOnSmall(avg_rank_order),
             SuccessiveHalving(avg_rank_order)]
    return pols
