"""Use Gymnasium when installed; otherwise a minimal stand-in with the same interface.

The stand-in only exists so the code and tests run in environments where Gymnasium cannot be
installed. Kaggle/Colab: `pip install gymnasium` and the real classes are used.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces
    HAVE_GYM = True
except (ImportError, Exception):  # pragma: no cover
    HAVE_GYM = False

    class _Discrete:
        def __init__(self, n, seed=None):
            self.n, self.shape, self.dtype = int(n), (), np.int64
            self._rng = np.random.default_rng(seed)

        def sample(self, mask=None):
            if mask is not None:
                return int(self._rng.choice(np.flatnonzero(mask)))
            return int(self._rng.integers(self.n))

        def contains(self, x):
            return 0 <= int(x) < self.n

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype

        def contains(self, x):
            return np.asarray(x).shape == self.shape

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    spaces = _Spaces()

    class _Env:
        metadata: dict = {}
        np_random = None

        def reset(self, *, seed=None, options=None):
            if seed is not None or self.np_random is None:
                self.np_random = np.random.default_rng(seed)

    class _Gym:
        Env = _Env

    gym = _Gym()
