"""Module 18: Evaluation metrics."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 18
MODULE_NAME = "metrics_18"


class Config18:
    """Configuration for module 18."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 126) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module18:
    """Main class for evaluation metrics."""

    def __init__(self, config: Config18) -> None:
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


def create_metrics_18(hidden_size: int = 256, **kwargs) -> Module18:
    """Factory function for Module18."""
    cfg = Config18(hidden_size=hidden_size, **kwargs)
    m = Module18(cfg)
    m.initialize()
    return m
