import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    train_steps = []
    train_loss = []
    val_epochs = []
    val_loss = []
    val_f1 = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row["phase"]
            if phase == "train":
                try:
                    train_steps.append(int(row["global_step"]))
                    train_loss.append(float(row["loss"]))
                except ValueError:
                    continue
            elif phase == "val":
                try:
                    val_epochs.append(int(row["epoch"]))
                    val_loss.append(float(row["loss"]))
                    val_f1.append(float(row["f1"]))
                except ValueError:
                    continue
    return (train_steps, train_loss), (val_epochs, val_loss, val_f1)


def plot_train_loss(train_data: Tuple[List[int], List[float]], out_path: Path):
    steps, loss = train_data
    if not steps:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(steps, loss, label="train loss")
    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.title("Train loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_val_metrics(val_data: Tuple[List[int], List[float], List[float]], out_path: Path):
    epochs, loss, f1 = val_data
    if not epochs:
        return
    plt.figure(figsize=(8, 4))
    if loss:
        plt.plot(epochs, loss, label="val loss")
    if f1:
        plt.plot(epochs, f1, label="val f1")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title("Validation metrics")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="从 metrics.csv 生成训练/验证曲线。")
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.csv 路径（由 train.py 生成）")
    parser.add_argument("--output-dir", type=Path, help="输出目录，默认与 metrics.csv 同级")
    args = parser.parse_args()

    if not args.metrics.is_file():
        raise FileNotFoundError(f"未找到 metrics 文件: {args.metrics}")
    out_dir = args.output_dir if args.output_dir else args.metrics.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = load_metrics(args.metrics)
    plot_train_loss(train_data, out_dir / "train_loss.png")
    plot_val_metrics(val_data, out_dir / "val_metrics.png")
    print(f"已生成曲线到 {out_dir}")


if __name__ == "__main__":
    main()
