"""Helpers for loading and inspecting CAN intrusion dataset files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_COLUMNS: List[str] = [
    "timestamp",
    "can_id",
    "dlc",
    "data0",
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
    "data6",
    "data7",
    "flag",
]


def load_can_csv(path: str | Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a CAN dataset file using a fixed column layout.

    If the dataset uses a different layout or variable-width rows, this loader
    may need to be adapted for manual parsing.
    """

    path = Path(path)
    cols = columns or DEFAULT_COLUMNS
    return pd.read_csv(path, header=None, names=cols)


def inspect_dataframe(df: pd.DataFrame) -> dict:
    """Return simple summary information for a CAN dataset DataFrame."""

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary


def class_balance(df: pd.DataFrame, label_column: str = "flag") -> pd.Series:
    """Return class counts for the given label column."""

    return df[label_column].value_counts(dropna=False)
