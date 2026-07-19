"""Paper 5 style CQMLP starter with optional Brevitas layers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import brevitas.nn as qnn

    HAS_BREVITAS = True
except ImportError:  # pragma: no cover - beginner-friendly fallback
    qnn = None
    HAS_BREVITAS = False


class CQMLPStarter(nn.Module):
    """Simple 40-byte input CQMLP skeleton for 4-class CAN IDS."""

    def __init__(self, bit_width: int = 2, num_classes: int = 4) -> None:
        super().__init__()

        if HAS_BREVITAS:
            self.quant_in = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
            self.fc1 = qnn.QuantLinear(40, 256, weight_bit_width=bit_width, bias=True)
            self.act1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
            self.fc2 = qnn.QuantLinear(256, 128, weight_bit_width=bit_width, bias=True)
            self.act2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
            self.fc3 = qnn.QuantLinear(128, 64, weight_bit_width=bit_width, bias=True)
            self.act3 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
            self.fc4 = qnn.QuantLinear(64, 32, weight_bit_width=bit_width, bias=True)
            self.act4 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
            self.fc_out = qnn.QuantLinear(32, num_classes, weight_bit_width=bit_width, bias=True)
        else:
            self.quant_in = nn.Identity()
            self.fc1 = nn.Linear(40, 256)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.act2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 64)
            self.act3 = nn.ReLU()
            self.fc4 = nn.Linear(64, 32)
            self.act4 = nn.ReLU()
            self.fc_out = nn.Linear(32, num_classes)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_in(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.act4(self.bn4(self.fc4(x)))
        return self.fc_out(x)


def create_dataloaders(
    dataset: Dict[str, np.ndarray],
    batch_size: int = 1024,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from saved .npz arrays."""

    def _to_loader(X_key: str, y_key: str, shuffle: bool) -> DataLoader:
        X_tensor = torch.tensor(dataset[X_key], dtype=torch.float32)
        y_tensor = torch.tensor(dataset[y_key], dtype=torch.long)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

    return {
        "train": _to_loader("X_train", "y_train", shuffle=True),
        "val": _to_loader("X_val", "y_val", shuffle=False),
        "test": _to_loader("X_test", "y_test", shuffle=False),
    }


def run_smoke_train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Run one small training epoch to verify the pipeline."""

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(train_loader), 1)
