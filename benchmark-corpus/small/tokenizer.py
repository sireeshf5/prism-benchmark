"""Simple BPE tokenizer for character-level language modelling."""
from __future__ import annotations
import re
from typing import Iterator


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        self.tok2id: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.id2tok: dict[int, str] = {i: t for i, t in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tok2id)

    def encode(self, text: str) -> list[int]:
        return [self.tok2id[ch] for ch in text if ch in self.tok2id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id2tok.get(i, "") for i in ids)


def build_vocab(text: str) -> Vocab:
    chars = sorted(set(text))
    return Vocab(chars)


def iterate_batches(data: list[int], block_size: int, batch_size: int) -> Iterator[tuple[list, list]]:
    import random
    n = len(data)
    for _ in range(batch_size):
        i = random.randint(0, n - block_size - 1)
        x = data[i : i + block_size]
        y = data[i + 1 : i + block_size + 1]
    yield x, y
