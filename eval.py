import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data import make_dataloader
from train import evaluate as run_eval, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch TransNetV2 on a manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest to evaluate.")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights (.pth).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mean-pooling", action="store_true")
    parser.add_argument("--no-many-hot", action="store_true")
    parser.add_argument("--many-hot-weight", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Construct a lightweight namespace to satisfy load_model signature
    model_args = argparse.Namespace(
        no_many_hot=args.no_many_hot,
        mean_pooling=args.mean_pooling,
        dropout=args.dropout,
        weights=args.weights,
    )
    model = load_model(model_args, device)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = make_dataloader(
        manifest=args.manifest,
        window=args.window,
        stride=args.stride,
        resize_hw=(27, 48),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    metrics = run_eval(model, dataloader, device, criterion, args.many_hot_weight)
    print(
        f"[Eval] loss={metrics['loss']:.4f} "
        f"f1={metrics['f1']:.4f} p={metrics['precision']:.4f} r={metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
