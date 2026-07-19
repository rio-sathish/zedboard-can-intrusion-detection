"""Colab-friendly script for merging, balancing, splitting, and saving window data.

Usage example:
python notebooks/02_merge_balance_split_save.py \
  --dos data/dos_windows.npz \
  --fuzzy data/fuzzy_windows.npz \
  --rpm data/rpm_windows.npz \
  --output data/cqmlp_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.processed_dataset import (
    balance_benign_windows,
    combine_windows_and_labels,
    save_processed_dataset,
    stratified_split,
)


def _load_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["X"], data["y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dos", required=True, help="Path to DoS windows .npz (keys: X, y)")
    parser.add_argument("--fuzzy", required=True, help="Path to Fuzzy windows .npz (keys: X, y)")
    parser.add_argument("--rpm", required=True, help="Path to RPM windows .npz (keys: X, y)")
    parser.add_argument("--output", required=True, help="Output .npz path for split dataset")
    parser.add_argument("--benign-multiplier", type=float, default=3.0, help="Benign cap vs total attack")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_dos, y_dos = _load_xy(args.dos)
    X_fuzzy, y_fuzzy = _load_xy(args.fuzzy)
    X_rpm, y_rpm = _load_xy(args.rpm)

    X_all, y_all = combine_windows_and_labels([(X_dos, y_dos), (X_fuzzy, y_fuzzy), (X_rpm, y_rpm)])
    X_final, y_final = balance_benign_windows(
        X_all,
        y_all,
        benign_label=0,
        benign_multiplier=args.benign_multiplier,
        seed=args.seed,
    )
    splits = stratified_split(X_final, y_final, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05, seed=args.seed)
    save_processed_dataset(args.output, **splits)

    print("Merged shape:", X_all.shape, y_all.shape)
    print("Balanced shape:", X_final.shape, y_final.shape)
    for split_name in ("train", "val", "test"):
        print(split_name, splits[f"X_{split_name}"].shape, splits[f"y_{split_name}"].shape)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
