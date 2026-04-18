"""Training loop with gradient accumulation and learning rate scheduling."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    compile: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    checkpoint_dir: str = "out"
    checkpoint_name: str = "ckpt.pt"
    seed: int = 1337
    loss_history: list[float] = field(default_factory=list)


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


class Trainer:
    def __init__(self, model, optimizer, train_cfg: TrainConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = train_cfg
        self.iter_num = 0
        self.best_val_loss = float("inf")

    def train_step(self, batch_x, batch_y) -> float:
        """Single forward + backward pass, returns scalar loss."""
        raise NotImplementedError("Subclass must implement train_step")

    def evaluate(self, get_batch) -> dict[str, float]:
        losses: dict[str, float] = {}
        for split in ("train", "val"):
            total = 0.0
            for _ in range(self.cfg.eval_iters):
                x, y = get_batch(split)
                loss = self.train_step(x, y)
                total += loss
            losses[split] = total / self.cfg.eval_iters
        return losses

    def run(self, get_batch) -> None:
        t0 = time.time()
        while self.iter_num < self.cfg.max_iters:
            lr = get_lr(self.iter_num, self.cfg)
            for group in self.optimizer.param_groups if hasattr(self.optimizer, "param_groups") else []:
                group["lr"] = lr
            if self.iter_num % self.cfg.eval_interval == 0:
                losses = self.evaluate(get_batch)
                print(f"step {self.iter_num}: train loss {losses.get('train', 0):.4f}, val loss {losses.get('val', 0):.4f}")
                if losses.get("val", float("inf")) < self.best_val_loss:
                    self.best_val_loss = losses["val"]
            x, y = get_batch("train")
            loss = self.train_step(x, y)
            self.cfg.loss_history.append(loss)
            if self.iter_num % self.cfg.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"iter {self.iter_num}: loss {loss:.4f}, time {dt*1000:.1f}ms")
            self.iter_num += 1
