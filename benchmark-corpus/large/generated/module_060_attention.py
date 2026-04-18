"""Module 60: Multi-head attention mechanism."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 60
MODULE_NAME = "attention_60"


class Config60:
    """Configuration for module 60."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 420) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module60:
    """Main class for multi-head attention mechanism."""

    def __init__(self, config: Config60) -> None:
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


def create_attention_60(hidden_size: int = 256, **kwargs) -> Module60:
    """Factory function for Module60."""
    cfg = Config60(hidden_size=hidden_size, **kwargs)
    m = Module60(cfg)
    m.initialize()
    return m
