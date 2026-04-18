"""Experiment script 10 for hyperparameter sweep."""
from __future__ import annotations

LEARNING_RATES = [1e-4, 3e-4, 1e-3]
BATCH_SIZES = [32, 64, 128]
EXPERIMENT_ID = 10


def run_experiment(lr: float, batch_size: int, seed: int = 42) -> dict:
    return {"lr": lr, "batch_size": batch_size, "seed": seed, "experiment_id": EXPERIMENT_ID}


def sweep() -> list[dict]:
    return [run_experiment(lr, bs) for lr in LEARNING_RATES for bs in BATCH_SIZES]


if __name__ == "__main__":
    results = sweep()
    print(f"Experiment 10: " + str(len(results)) + " runs completed")
