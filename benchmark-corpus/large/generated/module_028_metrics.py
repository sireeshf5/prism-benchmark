"""Module 28: Evaluation metrics."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 28
MODULE_NAME = "metrics_28"


class Config28:
    """Configuration for module 28."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 196) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module28:
    """Main class for evaluation metrics."""

    def __init__(self, config: Config28) -> None:
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        random.seed(self.config.seed)
        self._initialized = True

    def forward(self, x: List[float]) -> List[float]:
        if not self._initialized:
            raise RuntimeError("Call initialize() first")
        scale = math.sqrt(self.config.hidden_size)
        return [v / scale for v in x]

    def parameters(self) -> Dict[str, Any]:
        return {"hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "module_id": MODULE_ID}


def create_metrics_28(hidden_size: int = 256, **kwargs) -> Module28:
    """Factory function for Module28."""
    cfg = Config28(hidden_size=hidden_size, **kwargs)
    m = Module28(cfg)
    m.initialize()
    return m
