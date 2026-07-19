"""Colab-friendly Step 2 script: merge, balance, split, and save dataset.

Expected inputs are NumPy arrays produced by the sliding-window stage:
- X_dos.npy, y_dos.npy
- X_fuzzy.npy, y_fuzzy.npy
- X_rpm.npy, y_rpm.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.processed_dataset import (
    combine_windowed_datasets,
    label_counts,
    save_processed_dataset,
    stratified_train_val_test_split,
    subsample_benign,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dos-x", type=Path, required=True)
    parser.add_argument("--dos-y", type=Path, required=True)
    parser.add_argument("--fuzzy-x", type=Path, required=True)
    parser.add_argument("--fuzzy-y", type=Path, required=True)
    parser.add_argument("--rpm-x", type=Path, required=True)
    parser.add_argument("--rpm-y", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benign-multiplier", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pair(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(x_path), np.load(y_path)


def main() -> None:
    args = parse_args()

    X_dos, y_dos = load_pair(args.dos_x, args.dos_y)
    X_fuzzy, y_fuzzy = load_pair(args.fuzzy_x, args.fuzzy_y)
    X_rpm, y_rpm = load_pair(args.rpm_x, args.rpm_y)

    X_all, y_all = combine_windowed_datasets(
        [(X_dos, y_dos), (X_fuzzy, y_fuzzy), (X_rpm, y_rpm)]
    )
    print("Merged shape:", X_all.shape, y_all.shape)
    print("Merged counts:", label_counts(y_all))

    X_balanced, y_balanced = subsample_benign(
        X_all,
        y_all,
        benign_label=0,
        benign_multiplier=args.benign_multiplier,
        seed=args.seed,
    )
    print("Balanced shape:", X_balanced.shape, y_balanced.shape)
    print("Balanced counts:", label_counts(y_balanced))

    splits = stratified_train_val_test_split(
        X_balanced,
        y_balanced,
        train_size=0.85,
        val_size=0.10,
        test_size=0.05,
        seed=args.seed,
    )

    for split_name in ("train", "val", "test"):
        x_key = f"X_{split_name}"
        y_key = f"y_{split_name}"
        print(f"{split_name}: {splits[x_key].shape} {splits[y_key].shape}")

    save_processed_dataset(args.output, splits)
    print(f"Saved split dataset to {args.output}")


if __name__ == "__main__":
    main()
