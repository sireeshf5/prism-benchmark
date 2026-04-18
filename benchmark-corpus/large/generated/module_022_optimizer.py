"""Module 22: Custom optimizer implementations."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 22
MODULE_NAME = "optimizer_22"


class Config22:
    """Configuration for module 22."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 154) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module22:
    """Main class for custom optimizer implementations."""

    def __init__(self, config: Config22) -> None:
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


def create_optimizer_22(hidden_size: int = 256, **kwargs) -> Module22:
    """Factory function for Module22."""
    cfg = Config22(hidden_size=hidden_size, **kwargs)
    m = Module22(cfg)
    m.initialize()
    return m
