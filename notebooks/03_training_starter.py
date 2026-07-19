"""Simple training starter for the processed CAN IDS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.processed_dataset import load_processed_dataset
from src.model.cqmlp import CQMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def to_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: CQMLP, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    split_data = load_processed_dataset(args.dataset)

    X_train = torch.from_numpy(split_data["X_train"]).float()
    y_train = torch.from_numpy(split_data["y_train"]).long()
    X_val = torch.from_numpy(split_data["X_val"]).float()
    y_val = torch.from_numpy(split_data["y_val"]).long()

    train_loader = to_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = to_loader(X_val, y_val, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CQMLP(bit_width=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            if batch_idx + 1 >= args.max_batches:
                break

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch + 1} smoke_batches={args.max_batches} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
