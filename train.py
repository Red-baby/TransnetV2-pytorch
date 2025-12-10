import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data import make_dataloader
from transnetv2_pytorch import TransNetV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch TransNetV2.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Path to JSONL manifest for training.")
    parser.add_argument("--val-manifest", type=Path, help="Optional JSONL manifest for validation.")
    parser.add_argument("--weights", type=Path, help="Pretrained PyTorch weights (from convert_weights.py).")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pytorch"), help="Directory to save checkpoints.")
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
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> TransNetV2:
    model = TransNetV2(
        use_many_hot_targets=not args.no_many_hot,
        use_frame_similarity=True,
        use_color_histograms=True,
        use_mean_pooling=args.mean_pooling,
        dropout_rate=args.dropout,
    )
    if args.weights:
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
    model.eval()
    total_loss, total_main, total_aux, total_tp, total_fp, total_fn, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for batch in dataloader:
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

    train_loader = make_dataloader(
        manifest=args.train_manifest,
        window=args.window,
        stride=args.stride,
        resize_hw=(27, 48),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = (
        make_dataloader(
            manifest=args.val_manifest,
            window=args.window,
            stride=args.stride,
            resize_hw=(27, 48),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if args.val_manifest
        else None
    )

    model = load_model(args, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}", ncols=120)
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
            if step % args.log_every == 0:
                msg = (
                    f"ep {epoch} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.4f} "
                    f"(main {metrics.get('loss_main', 0):.4f}, aux {metrics.get('loss_many_hot', 0):.4f}) "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.set_postfix_str(msg)

        if val_loader and (epoch % args.val_every == 0):
            val_metrics = evaluate(model, val_loader, device, criterion, args.many_hot_weight)
            print(
                f"[Val] Epoch {epoch} loss={val_metrics['loss']:.4f} "
                f"f1={val_metrics['f1']:.4f} p={val_metrics['precision']:.4f} r={val_metrics['recall']:.4f}",
                flush=True,
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
