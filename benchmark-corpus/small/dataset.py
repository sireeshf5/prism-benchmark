"""Dataset loading and preprocessing utilities."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


class TextDataset:
    """Loads raw text from disk and exposes train/val splits."""

    def __init__(self, path: str | Path, split: float = 0.9) -> None:
        self.path = Path(path)
        self.split = split
        self._data: Optional[list[int]] = None
        self._vocab_size: int = 0

    def load(self, vocab) -> None:
        text = self.path.read_text(encoding="utf-8")
        encoded = vocab.encode(text)
        self._data = encoded
        self._vocab_size = len(vocab)

    @property
    def train(self) -> list[int]:
        if self._data is None:
            raise RuntimeError("Call load() first")
        n = int(len(self._data) * self.split)
        return self._data[:n]

    @property
    def val(self) -> list[int]:
        if self._data is None:
            raise RuntimeError("Call load() first")
        n = int(len(self._data) * self.split)
        return self._data[n:]

    def __len__(self) -> int:
        return len(self._data) if self._data else 0


def download_shakespeare(dest: str | Path = "data/shakespeare.txt") -> Path:
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest
