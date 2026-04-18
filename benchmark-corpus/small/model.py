"""Minimal transformer language model."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True


class LayerNorm:
    """Layer normalisation with optional bias."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        self.ndim = ndim
        self.bias = bias

    def forward(self, x):
        raise NotImplementedError("Requires a tensor library")


class CausalSelfAttention:
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x  # batch, time, channel (placeholder shapes)
        scale = math.sqrt(self.head_dim)
        return scale  # stub


class MLP:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def forward(self, x):
        return x  # stub


class Block:
    def __init__(self, config: ModelConfig) -> None:
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x


class GPT:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.blocks = [Block(config) for _ in range(config.n_layer)]

    def forward(self, idx, targets: Optional[list] = None):
        for block in self.blocks:
            idx = block.forward(idx)
        return idx

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        supported = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        if model_type not in supported:
            raise ValueError(f"model_type must be one of {supported}")
        config = ModelConfig()
        return cls(config)
