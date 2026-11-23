"""The budget-aware algorithm-selection MDP as a Gymnasium environment.

Episode   : one (dataset, seed) and a budget B (fraction of the cost of exhaustive search, i.e.
            of fitting every classifier at 100%). B = None means unlimited.
Actions   : K*L + 1 discrete. Action k*L + l = "evaluate classifier k at budget level l".
            Only levels ABOVE what classifier k already reached are valid (action mask), which
            maps your Continue (same classifier, higher level) / Switch (other classifier).
            The last action is STOP.
Dynamics  : an evaluation looks up the precomputed validation score and normalised cost.
            If it would exceed the remaining budget, the budget is spent, no score is returned
            and the episode ends (like a real time-out).
Selection : at the end, the classifier with the best OBSERVED validation score is selected
            (same rule for every method). Nothing observed -> regret = 1.
Reward    : -lam * cost per evaluation;  1 - regret at the end.
            regret = (q_max - q_selected) / (q_max - q_min), q = 100%-budget quality.
            quality="val"  -> q from validation scores (RL TRAINING: no test data shapes the policy)
            quality="test" -> q from test scores (evaluation/reporting).
            Both regrets are always reported in `info`.

Observation (all blocks always present; ablations zero a block via `flags`):
  progress[K] (level reached / L) | observed[K] (score - best so far) |
  cost_seen[K] (observed cost of k, scaled to a 100% estimate) |
  budget[3] (used, remaining, total; capped at 2) | best_so_far[1] (minus chance) |
  dataset features[F] (standardised simple descriptors of the dataset)
There is no meta-learner: the policy learns from these features directly.
"""
from __future__ import annotations

import numpy as np

from .compat import gym, spaces

DEFAULT_FLAGS = {"obs": True, "cost": True, "budget": True, "features": True}
CAP = 2.0  # cap for budget features (unlimited budget -> CAP)


class AutoClassifyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tables: dict, context: dict, classifiers: list[str], budget_levels: list[float],
                 instances: list, episode_budgets=(0.1, 0.25, 0.5, 1.0), lam: float = 0.05,
                 quality: str = "val", flags: dict | None = None, allowed_levels=None, seed: int | None = None,
                 cost_profile=None):
        super().__init__()
        assert quality in ("val", "test")
        self.tables, self.context = tables, context
        self.classifiers, self.levels = list(classifiers), list(budget_levels)
        self.K, self.L = len(classifiers), len(budget_levels)
        self.instances = list(instances)
        if not self.instances:
            raise ValueError("no instances for the environment")
        self.episode_budgets = list(episode_budgets)
        self.lam, self.quality = float(lam), quality
        self.flags = {**DEFAULT_FLAGS, **(flags or {})}
        allowed = range(self.L) if allowed_levels is None else [l % self.L for l in allowed_levels]
        self.allowed_levels = np.zeros(self.L, bool)
        self.allowed_levels[list(allowed)] = True
        self.n_feat = len(next(iter(context.values()))["features"])
        # average normalised cost per classifier on meta-train: used ONLY by baselines to check fit
        self.cost_profile = np.ones(self.K) / self.K if cost_profile is None else np.asarray(cost_profile, float)
        self.STOP = self.K * self.L
        self.action_space = spaces.Discrete(self.K * self.L + 1)
        self.obs_dim = 3 * self.K + 3 + 1 + self.n_feat
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------- helpers
    def action(self, k: int, l: int) -> int:
        return int(k) * self.L + int(l)

    def decode(self, a: int) -> tuple[int, int]:
        return divmod(int(a), self.L)

    def action_mask(self) -> np.ndarray:
        m = np.zeros(self.K * self.L + 1, bool)
        lv = np.arange(self.L)
        for k in range(self.K):
            m[k * self.L:(k + 1) * self.L] = (lv > self.level[k]) & self.allowed_levels
        m[self.STOP] = True
        return m

    @property
    def remaining(self) -> float:
        return np.inf if self.budget is None else self.budget - self.used

    def predicted_cost(self, k: int, l: int) -> float:
        """Baselines' cost estimate: meta-train average cost x budget fraction. Never the true cost."""
        return float(self.cost_profile[k] * self.levels[l])

    def best_so_far(self) -> float:
        s = self.obs_score[~np.isnan(self.obs_score)]
        return float(s.max()) if len(s) else np.nan

    def discrete_state(self, use_data_bucket: bool = True) -> tuple:
        """Compact state for tabular Q-learning: progress summary + a coarse dataset bucket
        (dataset size tercile x low/high dimensional, from the standardised features)."""
        B = 1.0 if self.budget is None else self.budget
        frac = min(self.used / B, 0.999)
        tried = np.flatnonzero(self.level >= 0)
        if len(tried):
            best = int(np.nanargmax(self.obs_score))
            best_lv = int(self.level[best]) + 1
        else:
            best, best_lv = self.K, 0
        b_idx = self.episode_budgets.index(self.budget) if self.budget in self.episode_budgets else len(self.episode_budgets)
        bucket = 0
        if use_data_bucket:
            f = self.ctx["features"]          # f[0] = log_n, f[2] = log_d_encoded (standardised)
            bucket = (0 if f[0] < -0.43 else 2 if f[0] > 0.43 else 1) * 2 + int(f[2] > 0)
        return (int(frac * 4), best, best_lv, min(len(tried), 3), b_idx, bucket)

    # ----------------------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        self.key = tuple(options["instance"]) if "instance" in options else self.instances[self._rng.integers(len(self.instances))]
        self.budget = options["budget"] if "budget" in options else float(self._rng.choice(self.episode_budgets))
        self.table, self.ctx = self.tables[self.key], self.context[self.key]
        self.level = np.full(self.K, -1)
        self.obs_score = np.full(self.K, np.nan)
        self.used, self.raw_seconds, self.n_evals, self.trace, self.done = 0.0, 0.0, 0, [], False
        return self._obs(), self._info()

    def step(self, a: int):
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        mask = self.action_mask()
        if not mask[a]:
            raise ValueError(f"invalid action {a} (mask violated)")
        reward, terminated, cut = 0.0, False, False
        if a == self.STOP:
            terminated = True
        else:
            k, l = self.decode(a)
            c = float(self.table.cost[k, l])
            if self.budget is not None and c > self.remaining + 1e-12:
                spent = self.budget - self.used          # budget exhausted mid-fit: no score
                self.raw_seconds += spent * self.table.exhaustive_s
                self.used = self.budget
                reward -= self.lam * spent
                self.trace.append({"clf": self.classifiers[k], "budget": self.levels[l], "cost": spent,
                                   "val": None, "cutoff": True})
                terminated, cut = True, True
            else:
                self.used += c
                self.raw_seconds += float(self.table.cost_s[k, l])
                self.level[k] = l
                self.obs_score[k] = self.table.val[k, l]
                self.n_evals += 1
                reward -= self.lam * c
                self.trace.append({"clf": self.classifiers[k], "budget": self.levels[l], "cost": c,
                                   "val": float(self.table.val[k, l]), "test": float(self.table.test[k, l]),
                                   "used": self.used, "best_val": self.best_so_far(), "cutoff": False})
                if not self.action_mask()[: self.STOP].any():
                    terminated = True
        info = self._info()
        if terminated:
            self.done = True
            final = self._finish()
            reward += 1.0 - final[f"regret_{self.quality}"]
            info.update(final, cutoff=cut)
        return self._obs(), float(reward), terminated, False, info

    # ----------------------------------------------------------------------------- internals
    def _regret(self, q: np.ndarray, s) -> float:
        if s is None:
            return 1.0
        span = q.max() - q.min()
        return 0.0 if span <= 0 else float((q.max() - q[s]) / span)

    def _finish(self) -> dict:
        s = None if np.all(np.isnan(self.obs_score)) else int(np.nanargmax(self.obs_score))
        qv, qt = self.table.val[:, -1], self.table.test[:, -1]
        return {"selected": None if s is None else self.classifiers[s],
                "regret_val": self._regret(qv, s), "regret_test": self._regret(qt, s),
                "test_score": float(qt[s]) if s is not None else float(self.table.chance),
                "best_test": float(qt.max()), "budget_used": float(self.used),
                "raw_seconds": float(self.raw_seconds), "n_evals": int(self.n_evals), "trace": list(self.trace)}

    def _info(self) -> dict:
        return {"action_mask": self.action_mask(), "instance": self.key, "budget": self.budget, "used": self.used}

    def _obs(self) -> np.ndarray:
        f, K = self.flags, self.K
        parts = [(self.level + 1) / self.L]
        best = self.best_so_far()
        if f["obs"]:
            rel = np.where(np.isnan(self.obs_score), 0.0, self.obs_score - (0 if np.isnan(best) else best))
            parts.append(rel)
        else:
            parts.append(np.zeros(K))
        if f["cost"]:
            seen = np.zeros(K)
            tried = self.level >= 0
            lv = self.level[tried]
            seen[tried] = self.table.cost[tried, lv] / np.asarray(self.levels)[lv]
            parts.append(np.clip(seen, 0, CAP))
        else:
            parts.append(np.zeros(K))
        if f["budget"]:
            B = CAP if self.budget is None else min(self.budget, CAP)
            parts.append(np.array([min(self.used, CAP), min(self.remaining, CAP), B]))
        else:
            parts.append(np.zeros(3))
        parts.append(np.array([0.0 if (np.isnan(best) or not f["obs"]) else best - self.table.chance]))
        parts.append(self.ctx["features"] if f["features"] else np.zeros(self.n_feat))
        return np.concatenate(parts).astype(np.float32)
