"""Utility module 6: data augmentation helper."""
from __future__ import annotations
import random

MODULE_ID = 6


def augment(text: str, prob: float = 0.1) -> str:
    words = text.split()
    return " ".join(w for w in words if random.random() > prob)


def batch_augment(texts: list[str], prob: float = 0.1) -> list[str]:
    return [augment(t, prob) for t in texts]


def shuffle_sentences(text: str) -> str:
    sentences = text.split(".")
    random.shuffle(sentences)
    return ". ".join(s.strip() for s in sentences if s.strip())
