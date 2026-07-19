"""Utilities for merged window datasets used by CQMLP training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


DatasetSplits = Dict[str, np.ndarray]


def combine_windows_and_labels(
    datasets: list[Tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple (X, y) datasets into one array pair."""

    if not datasets:
        raise ValueError("datasets must contain at least one (X, y) pair")

    X_parts, y_parts = zip(*datasets)
    return np.vstack(X_parts), np.concatenate(y_parts)


def balance_benign_windows(
    X: np.ndarray,
    y: np.ndarray,
    benign_label: int = 0,
    benign_multiplier: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample benign rows while keeping all attack rows."""

    if benign_multiplier <= 0:
        raise ValueError("benign_multiplier must be greater than 0")

    rng = np.random.default_rng(seed=seed)
    benign_idx = np.where(y == benign_label)[0]
    attack_idx = np.where(y != benign_label)[0]

    if benign_idx.size == 0 or attack_idx.size == 0:
        return X.copy(), y.copy()

    benign_cap = min(int(attack_idx.size * benign_multiplier), benign_idx.size)
    sampled_benign = rng.choice(benign_idx, size=benign_cap, replace=False)

    final_idx = np.concatenate([sampled_benign, attack_idx])
    rng.shuffle(final_idx)
    return X[final_idx], y[final_idx]


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetSplits:
    """Create reproducible per-class train/val/test splits."""

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed=seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    for class_id in np.unique(y):
        cls_idx = np.where(y == class_id)[0]
        rng.shuffle(cls_idx)

        n = cls_idx.size
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train : n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val :])

    def _shuffle_join(parts: list[np.ndarray]) -> np.ndarray:
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        return idx

    train = _shuffle_join(train_idx)
    val = _shuffle_join(val_idx)
    test = _shuffle_join(test_idx)

    return {
        "X_train": X[train],
        "y_train": y[train],
        "X_val": X[val],
        "y_val": y[val],
        "X_test": X[test],
        "y_test": y[test],
    }


def save_processed_dataset(path: str | Path, **splits: np.ndarray) -> Path:
    """Save split arrays to a compressed .npz file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **splits)
    return output_path


def load_processed_dataset(path: str | Path) -> DatasetSplits:
    """Load split arrays from a .npz file."""

    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}
