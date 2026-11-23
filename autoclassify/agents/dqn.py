"""DQN and Double DQN with action masking.

DQN target        : y = r + gamma * max_{a' valid} Q_target(s', a')
Double DQN target : y = r + gamma * Q_target(s', argmax_{a' valid} Q_online(s', a'))
                    (van Hasselt et al., 2016) - reduces DQN's over-estimation bias.
Loss: Huber on the chosen action. Experience replay + periodically synced target network
(Mnih et al., 2015). Invalid actions are masked both when acting and in the target.
"""
from __future__ import annotations

import numpy as np

from .mlp import MLP

NEG = -1e9


class Replay:
    def __init__(self, cap: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, obs_dim), np.float32)
        self.s2 = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, n_actions), bool)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done, mask2):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j], self.m2[j] = s, a, r, s2, done, mask2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int):
        idx = self.rng.integers(0, self.n, bs)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx]


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, double: bool, seed: int = 0):
        self.cfg, self.double, self.n_actions = cfg, double, n_actions
        sizes = [obs_dim, *cfg["hidden"], n_actions]
        self.q = MLP(sizes, seed)
        self.q_target = MLP(sizes, seed)
        self.q_target.set_params(self.q.get_params())
        self.buf = Replay(cfg["buffer_size"], obs_dim, n_actions, seed)
        self.rng = np.random.default_rng(seed + 1)

    @property
    def name(self) -> str:
        return "ddqn" if self.double else "dqn"

    def act(self, obs, mask, eps: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q = self.q.forward(obs[None])[0]
        return int(np.argmax(np.where(mask, q, NEG)))

    def update(self) -> float:
        c = self.cfg
        s, a, r, s2, d, m2 = self.buf.sample(c["batch_size"])
        q_all, acts = self.q.forward(s, cache=True)
        qt_next = self.q_target.forward(s2)
        if self.double:
            a_star = np.argmax(np.where(m2, self.q.forward(s2), NEG), axis=1)
            nxt = qt_next[np.arange(len(a)), a_star]
        else:
            nxt = np.where(m2, qt_next, NEG).max(axis=1)
        nxt = np.where(d > 0, 0.0, nxt)
        y = r + c["gamma"] * nxt
        q_sa = q_all[np.arange(len(a)), a]
        delta = q_sa - y
        grad = np.zeros_like(q_all)
        grad[np.arange(len(a)), a] = np.clip(delta, -1.0, 1.0) / len(a)  # Huber gradient
        gW, gb = self.q.backward(acts, grad)
        self.q.adam_step(gW, gb, c["lr"], c["grad_clip"])
        return float(np.mean(np.where(np.abs(delta) < 1, 0.5 * delta ** 2, np.abs(delta) - 0.5)))

    def sync_target(self):
        self.q_target.set_params(self.q.get_params())


def train_dqn(env, agent: DQNAgent, eval_fn=None, log=None) -> tuple[list, tuple]:
    """Train for cfg['train_steps'] environment steps. Returns (history, best_params).

    eval_fn(agent) -> mean regret on meta-val (lower is better); the best snapshot is kept.
    """
    c = agent.cfg
    steps, decay = c["train_steps"], max(1, int(c["eps_decay_frac"] * c["train_steps"]))
    obs, info = env.reset()
    hist, ep_ret, ep_rets, losses = [], 0.0, [], []
    best, best_params = np.inf, agent.q.get_params()
    for t in range(1, steps + 1):
        eps = c["eps_end"] + (c["eps_start"] - c["eps_end"]) * max(0.0, 1 - t / decay)
        a = agent.act(obs, info["action_mask"], eps)
        obs2, r, term, trunc, info2 = env.step(a)
        done = term or trunc
        mask2 = info2["action_mask"] if not done else np.eye(agent.n_actions, dtype=bool)[-1]
        agent.buf.add(obs, a, r, obs2, float(term), mask2)
        ep_ret += r
        if done:
            ep_rets.append(ep_ret)
            ep_ret = 0.0
            obs, info = env.reset()
        else:
            obs, info = obs2, info2
        if t >= c["warmup"] and agent.buf.n >= c["batch_size"]:
            losses.append(agent.update())
        if t % c["target_update"] == 0:
            agent.sync_target()
        if eval_fn is not None and (t % c["eval_every"] == 0 or t == steps):
            score = eval_fn(agent)
            rec = {"step": t, "eps": eps, "train_return": float(np.mean(ep_rets[-200:])) if ep_rets else np.nan,
                   "loss": float(np.mean(losses[-500:])) if losses else np.nan, "val_regret": score}
            hist.append(rec)
            if score < best:
                best, best_params = score, agent.q.get_params()
            if log:
                log.info(f"  [{agent.name}] step {t}: return {rec['train_return']:.3f} loss {rec['loss']:.4f} val regret {score:.4f}")
    return hist, best_params
