"""Module 8: Evaluation metrics."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 8
MODULE_NAME = "metrics_8"


class Config8:
    """Configuration for module 8."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 56) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module8:
    """Main class for evaluation metrics."""

    def __init__(self, config: Config8) -> None:
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


def create_metrics_8(hidden_size: int = 256, **kwargs) -> Module8:
    """Factory function for Module8."""
    cfg = Config8(hidden_size=hidden_size, **kwargs)
    m = Module8(cfg)
    m.initialize()
    return m
