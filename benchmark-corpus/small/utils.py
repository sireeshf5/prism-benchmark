"""Utility functions: checkpointing, sampling, and logging."""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_top_k(logits: list[float], k: int = 10) -> int:
    """Sample from top-k logits (pure Python fallback)."""
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:k]
    total = sum(v for _, v in indexed)
    r = random.random() * total
    cumulative = 0.0
    for idx, val in indexed:
        cumulative += val
        if r <= cumulative:
            return idx
    return indexed[-1][0]


def format_number(n: int | float, decimals: int = 2) -> str:
    if abs(n) >= 1e9:
        return f"{n / 1e9:.{decimals}f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.{decimals}f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.{decimals}f}K"
    return str(n)


class Timer:
    def __init__(self) -> None:
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed * 1000:.1f}ms"
