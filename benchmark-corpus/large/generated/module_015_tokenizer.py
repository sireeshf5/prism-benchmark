"""Module 15: Tokenizer base classes."""
from __future__ import annotations
import math
import random
from typing import Optional, List, Dict, Any

MODULE_ID = 15
MODULE_NAME = "tokenizer_15"


class Config15:
    """Configuration for module 15."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, seed: int = 105) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed


class Module15:
    """Main class for tokenizer base classes."""

    def __init__(self, config: Config15) -> None:
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


def create_tokenizer_15(hidden_size: int = 256, **kwargs) -> Module15:
    """Factory function for Module15."""
    cfg = Config15(hidden_size=hidden_size, **kwargs)
    m = Module15(cfg)
    m.initialize()
    return m
