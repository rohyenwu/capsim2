"""Minimal subset of gym.spaces used by the WiFi eval scripts."""

from __future__ import annotations

import numpy as np


class Space:
    def sample(self):
        raise NotImplementedError


class Discrete(Space):
    def __init__(self, n: int, seed: int | None = None, start: int = 0):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)
        self.start = int(start)
        self.shape = ()
        self.dtype = np.int64
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return int(self._rng.integers(self.start, self.start + self.n))

    def contains(self, x) -> bool:
        try:
            value = int(x)
        except (TypeError, ValueError):
            return False
        return self.start <= value < self.start + self.n


class Box(Space):
    def __init__(self, low, high, shape=None, dtype=np.float32, seed: int | None = None):
        self.dtype = np.dtype(dtype)
        self.low = np.array(low, dtype=self.dtype)
        self.high = np.array(high, dtype=self.dtype)
        if shape is None:
            shape = np.broadcast(self.low, self.high).shape
        self.shape = tuple(shape)
        self.low = np.broadcast_to(self.low, self.shape).astype(self.dtype)
        self.high = np.broadcast_to(self.high, self.shape).astype(self.dtype)
        self._rng = np.random.default_rng(seed)

    def sample(self):
        finite = np.isfinite(self.low) & np.isfinite(self.high)
        out = np.zeros(self.shape, dtype=self.dtype)
        if np.any(finite):
            out[finite] = self._rng.uniform(self.low[finite], self.high[finite])
        return out.astype(self.dtype)

    def contains(self, x) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return arr.shape == self.shape and bool(np.all(arr >= self.low) and np.all(arr <= self.high))
