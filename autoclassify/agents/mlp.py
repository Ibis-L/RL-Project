"""A small multilayer perceptron with manual backpropagation and Adam, in NumPy.

Used as the Q-network of DQN / Double DQN. Writing it by hand keeps the agents dependency-free,
fully transparent for the report, and fast enough for this problem (tens of inputs/outputs).
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / a) if i < len(sizes) - 2 else np.sqrt(1.0 / a) * 0.1
            self.W.append(rng.normal(0, scale, (a, b)).astype(np.float64))
            self.b.append(np.zeros(b))
        self._adam_init()

    def _adam_init(self):
        self.t = 0
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, x: np.ndarray, cache: bool = False):
        h, acts = np.asarray(x, np.float64), []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            acts.append(h)
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
        return (h, acts) if cache else h

    def backward(self, acts, grad_out: np.ndarray):
        gW, gb, g = [None] * len(self.W), [None] * len(self.W), grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def adam_step(self, gW, gb, lr: float, clip: float | None = 10.0, b1=0.9, b2=0.999, eps=1e-8):
        if clip:
            norm = np.sqrt(sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb))
            if norm > clip:
                gW = [g * clip / norm for g in gW]
                gb = [g * clip / norm for g in gb]
        self.t += 1
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                mh, vh = m[i] / (1 - b1 ** self.t), v[i] / (1 - b2 ** self.t)
                p[i] = p[i] - lr * mh / (np.sqrt(vh) + eps)

    def get_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, params):
        W, b = params
        self.W, self.b = [w.copy() for w in W], [x.copy() for x in b]

    def save(self, path):
        np.savez(path, **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        m = cls.__new__(cls)
        m.W = [z[f"W{i}"] for i in range(n)]
        m.b = [z[f"b{i}"] for i in range(n)]
        m._adam_init()
        return m
