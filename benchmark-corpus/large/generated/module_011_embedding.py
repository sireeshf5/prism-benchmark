"""Module 11: Token and positional embeddings."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 11
MODULE_NAME = "embedding_11"


class Config11:
    """Configuration for module 11."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 77) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module11:
    """Main class for token and positional embeddings."""

    def __init__(self, config: Config11) -> None:
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


def create_embedding_11(hidden_size: int = 256, **kwargs) -> Module11:
    """Factory function for Module11."""
    cfg = Config11(hidden_size=hidden_size, **kwargs)
    m = Module11(cfg)
    m.initialize()
    return m
