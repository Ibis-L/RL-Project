"""Tabular Q-learning on a compact, discretised state (no meta-features).

State = (budget-used bucket, current best classifier, its level, number tried, budget index,
coarse dataset bucket). See AutoClassifyEnv.discrete_state.
Update: Q(s,a) += alpha * (r + gamma * max_{a' valid} Q(s',a') - Q(s,a)).

Back-off: a second, coarser table ignores the dataset-bucket component. When the fine state was never
visited in training (common on unseen datasets), the agent acts from the coarse table.
"""
from __future__ import annotations

import pickle

import numpy as np

NEG = -1e9


class QLearningAgent:
    name = "qlearning"

    def __init__(self, n_actions: int, cfg: dict, seed: int = 0):
        self.n_actions, self.cfg = n_actions, cfg
        self.Q: dict = {}        # fine table
        self.Qc: dict = {}       # coarse table (state without the last component)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _coarse(s):
        return s[:-1]

    def _get(self, table, s):
        if s not in table:
            table[s] = np.zeros(self.n_actions)
        return table[s]

    def act(self, s, mask, eps: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q = self.Q.get(s)
        if q is None:
            q = self.Qc.get(self._coarse(s))
        if q is None:
            return int(valid[-1])  # never seen at all: STOP (last valid action)
        return int(np.argmax(np.where(mask, q, NEG)))

    def update(self, s, a, r, s2, mask2, done):
        c = self.cfg
        for table, ss, ss2 in ((self.Q, s, s2), (self.Qc, self._coarse(s), self._coarse(s2))):
            target = r if done else r + c["gamma"] * np.where(mask2, self._get(table, ss2), NEG).max()
            q = self._get(table, ss)
            q[a] += c["alpha"] * (target - q[a])

    def snapshot(self):
        return ({k: v.copy() for k, v in self.Q.items()}, {k: v.copy() for k, v in self.Qc.items()})

    def restore(self, snap):
        self.Q, self.Qc = snap

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "Qc": self.Qc}, f)

    @classmethod
    def load(cls, path, n_actions, cfg):
        a = cls(n_actions, cfg)
        with open(path, "rb") as f:
            d = pickle.load(f)
        a.Q, a.Qc = d["Q"], d["Qc"]
        return a


def train_qlearning(env, agent: QLearningAgent, eval_fn=None, log=None):
    c = agent.cfg
    use_top1 = c.get("use_data_bucket", True)
    n_ep, decay = c["episodes"], max(1, int(c["eps_decay_frac"] * c["episodes"]))
    hist, rets = [], []
    best, best_snap = np.inf, None
    for ep in range(1, n_ep + 1):
        eps = c["eps_end"] + (c["eps_start"] - c["eps_end"]) * max(0.0, 1 - ep / decay)
        _, info = env.reset()
        s, done, G = env.discrete_state(use_top1), False, 0.0
        while not done:
            a = agent.act(s, info["action_mask"], eps)
            _, r, term, trunc, info = env.step(a)
            done = term or trunc
            s2 = env.discrete_state(use_top1)
            agent.update(s, a, r, s2, info["action_mask"], done)
            s, G = s2, G + r
        rets.append(G)
        if eval_fn is not None and (ep % c["eval_every"] == 0 or ep == n_ep):
            score = eval_fn(agent)
            hist.append({"episode": ep, "eps": eps, "train_return": float(np.mean(rets[-500:])),
                         "val_regret": score, "n_states": len(agent.Q)})
            if score < best:
                best, best_snap = score, agent.snapshot()
            if log:
                log.info(f"  [qlearning] episode {ep}: return {hist[-1]['train_return']:.3f} val regret {score:.4f} states {len(agent.Q)}")
    return hist, best_snap
