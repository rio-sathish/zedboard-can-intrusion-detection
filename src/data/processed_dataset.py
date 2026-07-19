"""Utilities for building balanced train/val/test CAN IDS datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


ArrayPair = Tuple[np.ndarray, np.ndarray]


def combine_windowed_datasets(dataset_pairs: Iterable[ArrayPair]) -> ArrayPair:
    """Combine multiple ``(X, y)`` window datasets into one dataset."""

    features = []
    labels = []

    for x_part, y_part in dataset_pairs:
        if x_part.shape[0] != y_part.shape[0]:
            raise ValueError("Each dataset pair must have matching first dimension.")
        features.append(x_part)
        labels.append(y_part)

    if not features:
        raise ValueError("At least one dataset pair is required.")

    return np.vstack(features), np.concatenate(labels)


def subsample_benign(
    X: np.ndarray,
    y: np.ndarray,
    benign_label: int = 0,
    benign_multiplier: int = 3,
    seed: int = 42,
) -> ArrayPair:
    """Subsample benign windows while keeping all attack windows."""

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching first dimension.")

    rng = np.random.default_rng(seed=seed)
    benign_mask = y == benign_label
    attack_mask = ~benign_mask

    benign_indices = np.where(benign_mask)[0]
    attack_indices = np.where(attack_mask)[0]

    benign_cap = min(benign_indices.size, attack_indices.size * benign_multiplier)

    sampled_benign_indices = (
        rng.choice(benign_indices, size=benign_cap, replace=False)
        if benign_cap > 0
        else np.array([], dtype=np.int64)
    )

    final_indices = np.concatenate([sampled_benign_indices, attack_indices])
    rng.shuffle(final_indices)

    return X[final_indices], y[final_indices]


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.85,
    val_size: float = 0.10,
    test_size: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create reproducible stratified train/val/test splits."""

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=seed,
    )

    temp_test_ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=temp_test_ratio,
        stratify=y_temp,
        random_state=seed,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_processed_dataset(path: str | Path, dataset: Dict[str, np.ndarray]) -> None:
    """Save a processed dataset dictionary to ``.npz``."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **dataset)


def load_processed_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a processed dataset dictionary from ``.npz``."""

    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}


def label_counts(y: np.ndarray) -> Dict[int, int]:
    """Return integer class counts for quick checks."""

    classes, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(classes, counts)}
