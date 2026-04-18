"""Module 16: Checkpoint save/load logic."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 16
MODULE_NAME = "checkpoint_16"


class Config16:
    """Configuration for module 16."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 112) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module16:
    """Main class for checkpoint save/load logic."""

    def __init__(self, config: Config16) -> None:
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


def create_checkpoint_16(hidden_size: int = 256, **kwargs) -> Module16:
    """Factory function for Module16."""
    cfg = Config16(hidden_size=hidden_size, **kwargs)
    m = Module16(cfg)
    m.initialize()
    return m
