"""Training starter for the saved CQMLP dataset (.npz)."""

from __future__ import annotations

import argparse

from src.data.processed_dataset import load_processed_dataset
from src.model.cqmlp_starter import CQMLPStarter, create_dataloaders, run_smoke_train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to split dataset .npz file")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--bit-width", type=int, default=2, help="Brevitas quantization bit width")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_processed_dataset(args.dataset)
    loaders = create_dataloaders(dataset, batch_size=args.batch_size)

    model = CQMLPStarter(bit_width=args.bit_width, num_classes=4)
    loss = run_smoke_train_epoch(model, loaders["train"], device=args.device)

    print("Smoke train loss:", f"{loss:.6f}")
    print("Train batches:", len(loaders["train"]))
    print("Val batches:", len(loaders["val"]))
    print("Test batches:", len(loaders["test"]))


if __name__ == "__main__":
    main()
