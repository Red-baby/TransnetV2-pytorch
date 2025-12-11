import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
# Ensure the script directory is on sys.path so local imports like `from data import ...` work
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data import make_dataloader
from transnetv2_pytorch import TransNetV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch TransNetV2.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Path to JSONL manifest for training.")
    parser.add_argument("--val-manifest", type=Path, help="Optional JSONL manifest for validation.")
    parser.add_argument("--weights", type=Path, help="Pretrained PyTorch weights (from convert_weights.py).")
    # By default save outputs inside the `training_pytorch` folder so the script can be
    # executed from that folder (or from elsewhere) without requiring changing cwd.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "pytorch",
        help="Directory to save checkpoints (default: training_pytorch/runs/pytorch).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window", type=int, default=100, help="Sliding window length (frames).")
    parser.add_argument("--stride", type=int, default=25, help="Stride between windows (frames).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--log-every", type=int, default=10, help="Logging interval (steps).")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mean-pooling", action="store_true", help="Use spatial mean pooling before FC.")
    parser.add_argument("--no-many-hot", action="store_true", help="Disable many_hot auxiliary head.")
    parser.add_argument("--many-hot-weight", type=float, default=0.5, help="Loss weight for many_hot head.")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping (L2 norm).")
    parser.add_argument("--auto-split", action="store_true", help="Automatically split train-manifest 9:1 into train/val.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for auto split.")
    return parser.parse_args()


def ensure_metrics_logger(path: Path, fieldnames: List[str]) -> None:
    """Create CSV file with header if not exists."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_metrics(path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def autosplit_manifest(manifest: Path, output_dir: Path, seed: int, train_ratio: float = 0.9) -> Tuple[Path, Path]:
    """Split a single manifest into train/val by ratio; returns paths to new manifests."""
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    print(f"[Split] Reading manifest {manifest.resolve()}", flush=True)
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Manifest has too few entries to split: {manifest}")
    rnd = random.Random(seed)
    rnd.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train_split.jsonl"
    val_path = split_dir / "val_split.jsonl"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    print(
        f"[Split] Auto-split manifest: {manifest} -> {train_path} ({len(train_lines)} lines), "
        f"{val_path} ({len(val_lines)} lines)",
        flush=True,
    )
    return train_path, val_path


def manifest_line_count(manifest: Path) -> int:
    return sum(1 for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip())


def load_model(args: argparse.Namespace, device: torch.device) -> TransNetV2:
    model = TransNetV2(
        use_many_hot_targets=not args.no_many_hot,
        use_frame_similarity=True,
        use_color_histograms=True,
        use_mean_pooling=args.mean_pooling,
        dropout_rate=args.dropout,
    )
    if args.weights:
        print(f"[Model] Loading weights from {args.weights}", flush=True)
        state = torch.load(args.weights, map_location=device)
        # allow either pure state_dict or checkpoint dict
        state = state["model_state"] if "model_state" in state else state
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {args.weights}")
    model.to(device)
    return model


def compute_loss(
    outputs,
    labels: torch.Tensor,
    many_hot: torch.Tensor,
    criterion: nn.Module,
    many_hot_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if isinstance(outputs, tuple):
        logits, extra = outputs
    else:
        logits, extra = outputs, {}
    logits = logits.squeeze(-1)
    loss = criterion(logits, labels)
    metrics = {"loss_main": loss.item()}
    if "many_hot" in extra:
        aux_loss = criterion(extra["many_hot"].squeeze(-1), many_hot)
        loss = loss + many_hot_weight * aux_loss
        metrics["loss_many_hot"] = aux_loss.item()
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: TransNetV2,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    many_hot_weight: float,
) -> Dict[str, float]:
    print("[Val] Evaluation started", flush=True)
    model.eval()
    total_loss, total_main, total_aux, total_tp, total_fp, total_fn, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    val_bar = tqdm(dataloader, desc="Val", ncols=120)
    for batch in val_bar:
        videos = batch["video"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        many_hot = batch["many_hot"].to(device, non_blocking=True)

        outputs = model(videos)
        loss, metrics = compute_loss(outputs, labels, many_hot, criterion, many_hot_weight)
        total_loss += loss.item()
        total_main += metrics.get("loss_main", 0.0)
        total_aux += metrics.get("loss_many_hot", 0.0)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).float()
        total_tp += (preds * labels).sum().item()
        total_fp += ((preds == 1) & (labels == 0)).sum().item()
        total_fn += ((preds == 0) & (labels == 1)).sum().item()
        n_batches += 1

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print("[Val] Evaluation finished", flush=True)
    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_main": total_main / max(n_batches, 1),
        "loss_many_hot": total_aux / max(n_batches, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train():
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Args] {args}", flush=True)

    # Auto split if requested and no explicit val manifest
    train_manifest_path = args.train_manifest
    val_manifest_path = args.val_manifest
    if args.auto_split:
        if args.val_manifest:
            print("Warning: --auto-split is set but --val-manifest provided; using provided val-manifest.", flush=True)
        else:
            train_manifest_path, val_manifest_path = autosplit_manifest(
                manifest=args.train_manifest, output_dir=args.output_dir, seed=args.split_seed, train_ratio=0.9
            )

    # Manifest stats
    train_count = manifest_line_count(train_manifest_path)
    print(f"[Data] Train manifest: {train_manifest_path.resolve()} ({train_count} entries)", flush=True)
    if val_manifest_path:
        val_count = manifest_line_count(val_manifest_path)
        print(f"[Data] Val manifest:   {val_manifest_path.resolve()} ({val_count} entries)", flush=True)
    else:
        val_count = 0

    train_loader = make_dataloader(
        manifest=train_manifest_path,
        window=args.window,
        stride=args.stride,
        resize_hw=(27, 48),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = (
        make_dataloader(
            manifest=val_manifest_path,
            window=args.window,
            stride=args.stride,
            resize_hw=(27, 48),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if val_manifest_path
        else None
    )
    print(f"[Data] Train loader batches: {len(train_loader)}", flush=True)
    if val_loader:
        print(f"[Data] Val loader batches:   {len(val_loader)}", flush=True)

    model = load_model(args, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_path = args.output_dir / "metrics.csv"
    fieldnames = [
        "phase",
        "epoch",
        "step",
        "global_step",
        "loss",
        "loss_main",
        "loss_many_hot",
        "precision",
        "recall",
        "f1",
        "lr",
    ]
    ensure_metrics_logger(metrics_path, fieldnames)

    # Initial evaluation before training starts
    if val_loader:
        print("[Val] Running initial validation before training...", flush=True)
        init_metrics = evaluate(model, val_loader, device, criterion, args.many_hot_weight)
        append_metrics(
            metrics_path,
            {
                "phase": "val",
                "epoch": 0,
                "step": "",
                "global_step": 0,
                "loss": init_metrics["loss"],
                "loss_main": init_metrics["loss_main"],
                "loss_many_hot": init_metrics["loss_many_hot"],
                "precision": init_metrics["precision"],
                "recall": init_metrics["recall"],
                "f1": init_metrics["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            },
            fieldnames,
        )
        print(
            f"[Val-before-train] loss={init_metrics['loss']:.4f} f1={init_metrics['f1']:.4f} "
            f"p={init_metrics['precision']:.4f} r={init_metrics['recall']:.4f}",
            flush=True,
        )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"[Train] Starting epoch {epoch}/{args.epochs}", flush=True)
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}", ncols=120)
        first_batch_loaded = False
        for step, batch in pbar:
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            many_hot = batch["many_hot"].to(device, non_blocking=True)

            outputs = model(videos)
            loss, metrics = compute_loss(outputs, labels, many_hot, criterion, args.many_hot_weight)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            global_step += 1
            if not first_batch_loaded:
                print(f"[Train] First batch loaded (epoch {epoch}, step {step})", flush=True)
                first_batch_loaded = True
            if step % args.log_every == 0:
                msg = (
                    f"ep {epoch} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.4f} "
                    f"(main {metrics.get('loss_main', 0):.4f}, aux {metrics.get('loss_many_hot', 0):.4f}) "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.set_postfix_str(msg)
                append_metrics(
                    metrics_path,
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": loss.item(),
                        "loss_main": metrics.get("loss_main", 0.0),
                        "loss_many_hot": metrics.get("loss_many_hot", 0.0),
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    fieldnames,
                )

        if val_loader and (epoch % args.val_every == 0):
            val_metrics = evaluate(model, val_loader, device, criterion, args.many_hot_weight)
            print(
                f"[Val] Epoch {epoch} loss={val_metrics['loss']:.4f} "
                f"f1={val_metrics['f1']:.4f} p={val_metrics['precision']:.4f} r={val_metrics['recall']:.4f}",
                flush=True,
            )
            append_metrics(
                metrics_path,
                {
                    "phase": "val",
                    "epoch": epoch,
                    "step": "",
                    "global_step": global_step,
                    "loss": val_metrics["loss"],
                    "loss_main": val_metrics["loss_main"],
                    "loss_many_hot": val_metrics["loss_many_hot"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
                fieldnames,
            )

        if epoch % args.save_every == 0:
            ckpt_path = args.output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            torch.save(model.state_dict(), args.output_dir / "last_state_dict.pt")
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
