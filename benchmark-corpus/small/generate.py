"""Text generation with temperature scaling and top-k/top-p sampling."""
from __future__ import annotations
import math
import random
from typing import Callable, Optional


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature == 0.0:
        # Argmax (greedy)
        m = max(range(len(logits)), key=lambda i: logits[i])
        return [1.0 if i == m else 0.0 for i in range(len(logits))]
    scaled = [x / temperature for x in logits]
    max_val = max(scaled)
    exps = [math.exp(x - max_val) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def top_p_filter(probs: list[float], p: float = 0.9) -> list[float]:
    """Nucleus sampling: zero out tokens outside the top-p mass."""
    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    result = [0.0] * len(probs)
    for idx, prob in sorted_probs:
        if cumulative < p:
            result[idx] = prob
            cumulative += prob
        else:
            break
    total = sum(result)
    return [x / total if total > 0 else 0.0 for x in result]


def generate(
    model_forward: Callable,
    prompt_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> list[int]:
    """Auto-regressive generation loop."""
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model_forward(ids)
        probs = softmax(logits, temperature)
        if top_p is not None:
            probs = top_p_filter(probs, top_p)
        if top_k is not None:
            indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            keep = {i for i, _ in indexed[:top_k]}
            probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]
            total = sum(probs)
            probs = [p / total for p in probs]
        r = random.random()
        cumulative = 0.0
        chosen = len(probs) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break
        ids.append(chosen)
    return ids
