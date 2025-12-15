import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from transnetv2_pytorch import TransNetV2


def _ensure_cv2() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise ImportError("This script requires OpenCV. Install with `pip install opencv-python`.") from exc


def _read_video_frame(cap, resize_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    import cv2

    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width, height = resize_hw[1], resize_hw[0]
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.uint8)


def _read_frames_from_dir(frames_dir: Path, resize_hw: Tuple[int, int], n_frames: int) -> List[np.ndarray]:
    _ensure_cv2()
    from data import _read_image  # reuse resize+RGB

    files = sorted([p for p in frames_dir.glob("*") if p.is_file()])
    if len(files) < n_frames:
        raise ValueError(f"frames_dir {frames_dir} has {len(files)} frames < n_frames={n_frames}")
    return [_read_image(p, resize_hw) for p in files[:n_frames]]


def _get_padding(n_frames: int, window: int, stride: int) -> Tuple[int, int, int, int]:
    """
    Return (start_pad, end_pad, crop_start, crop_end).

    Matches the TF evaluation logic where each window yields `stride` predictions from the center.
    """
    if window % stride != 0:
        raise ValueError(f"window must be divisible by stride (got window={window}, stride={stride})")
    if (window - stride) % 2 != 0:
        raise ValueError(f"(window - stride) must be even (got window={window}, stride={stride})")

    crop_start = (window - stride) // 2
    crop_end = crop_start + stride
    start_pad = crop_start
    remainder = n_frames % stride
    end_pad = crop_start + ((stride - remainder) if remainder != 0 else 0)
    return start_pad, end_pad, crop_start, crop_end


@torch.no_grad()
def _predict_probs_for_windows(
    model: TransNetV2,
    windows: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    windows: uint8 [B, T, H, W, 3]
    returns:
      one_hot_probs: [B, T]
      many_hot_probs: [B, T] or None
    """
    x = torch.from_numpy(windows).to(device)
    outputs = model(x)
    if isinstance(outputs, tuple):
        logits, extra = outputs
    else:
        logits, extra = outputs, {}

    one_hot_probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
    many_hot_probs: Optional[np.ndarray] = None
    if isinstance(extra, dict) and "many_hot" in extra:
        many_hot_probs = torch.sigmoid(extra["many_hot"].squeeze(-1)).detach().cpu().numpy()
    return one_hot_probs, many_hot_probs


def predict_video(
    model: TransNetV2,
    entry: Dict[str, Any],
    *,
    device: torch.device,
    resize_hw: Tuple[int, int],
    window: int,
    stride: int,
    batch_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    labels = entry.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Manifest entry missing non-empty 'labels'.")
    n_frames = len(labels)

    start_pad, end_pad, crop_start, crop_end = _get_padding(n_frames, window=window, stride=stride)

    buf: deque[np.ndarray] = deque()
    all_probs: List[np.ndarray] = []
    all_many_probs: List[np.ndarray] = []
    have_many = False

    pending_windows: List[np.ndarray] = []

    def flush_windows() -> None:
        nonlocal have_many
        if not pending_windows:
            return
        batch = np.stack(pending_windows, axis=0).astype(np.uint8)
        one_hot_probs, many_hot_probs = _predict_probs_for_windows(model, batch, device)

        all_probs.append(one_hot_probs[:, crop_start:crop_end])
        if many_hot_probs is not None:
            have_many = True
            all_many_probs.append(many_hot_probs[:, crop_start:crop_end])
        pending_windows.clear()

    def push_frame(frame: np.ndarray) -> None:
        buf.append(frame)
        if len(buf) != window:
            return
        pending_windows.append(np.stack(list(buf), axis=0))
        for _ in range(stride):
            buf.popleft()
        if len(pending_windows) >= batch_size:
            flush_windows()

    if entry.get("frames_dir"):
        frames_dir = Path(entry["frames_dir"])
        frames_list = _read_frames_from_dir(frames_dir, resize_hw, n_frames=n_frames)
        first_frame = frames_list[0]
        last_frame = frames_list[-1]

        for _ in range(start_pad):
            push_frame(first_frame)
        for frame in frames_list:
            push_frame(frame)
        for _ in range(end_pad):
            push_frame(last_frame)
    else:
        video_path = entry.get("video_path")
        if not video_path:
            raise ValueError("Manifest entry must have either 'video_path' or 'frames_dir'.")

        _ensure_cv2()
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        first_frame = _read_video_frame(cap, resize_hw)
        if first_frame is None:
            cap.release()
            raise ValueError(f"Video contained no readable frames: {video_path}")

        for _ in range(start_pad):
            push_frame(first_frame)

        push_frame(first_frame)
        last_frame = first_frame
        for _ in range(n_frames - 1):
            frame = _read_video_frame(cap, resize_hw)
            if frame is None:
                cap.release()
                raise ValueError(f"Video ended before n_frames={n_frames}: {video_path}")
            last_frame = frame
            push_frame(frame)
        cap.release()

        for _ in range(end_pad):
            push_frame(last_frame)

    flush_windows()

    if not all_probs:
        raise ValueError("No windows were generated for this video; check window/stride settings and input length.")

    # all_probs contains chunks shaped [batch, stride]; last chunk may have smaller batch,
    # so concatenate on the window axis, then flatten to a single per-frame sequence.
    probs = np.concatenate(all_probs, axis=0).reshape(-1)[:n_frames]
    many_probs = np.concatenate(all_many_probs, axis=0).reshape(-1)[:n_frames] if have_many else None
    return probs, many_probs


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    predictions = (predictions > 0).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(
    gt_scenes: np.ndarray,
    pred_scenes: np.ndarray,
    *,
    n_frames_miss_tolerance: int = 2,
) -> Tuple[float, float, float, Tuple[int, int, int]]:
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            i += 1
        elif i == len(gt_trans):
            fp += 1
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1, (tp, fp, fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch TransNetV2 model on BBC JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to BBC manifest JSONL.")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights (.pth/.pt).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size (number of windows).")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for transition.")
    parser.add_argument("--tolerance", type=int, default=2, help="Frame miss tolerance for scene-based F1.")
    parser.add_argument("--max-videos", type=int, default=0, help="Stop after N videos (0 = all).")
    parser.add_argument("--save-probs-dir", type=Path, default=None, help="If set, save per-video probs as .npy.")
    return parser.parse_args()


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = torch.device(args.device)

    model = TransNetV2()
    try:
        state = torch.load(args.weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location=device)
    state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    entries = _load_manifest(args.manifest)
    if args.max_videos:
        entries = entries[: args.max_videos]

    total_tp, total_fp, total_fn = 0, 0, 0
    total_tp_mh, total_fp_mh, total_fn_mh = 0, 0, 0
    have_many_hot = False

    scene_tp, scene_fp, scene_fn = 0, 0, 0

    for idx, entry in enumerate(entries, start=1):
        src = entry.get("video_path") or entry.get("frames_dir") or f"entry#{idx}"
        print(f"[{idx}/{len(entries)}] {src}", flush=True)

        probs, many_probs = predict_video(
            model,
            entry,
            device=device,
            resize_hw=(27, 48),
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        if args.save_probs_dir:
            args.save_probs_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(str(src)).stem
            np.save(args.save_probs_dir / f"{stem}.one_hot.npy", probs)
            if many_probs is not None:
                np.save(args.save_probs_dir / f"{stem}.many_hot.npy", many_probs)

        gt_one = np.asarray(entry.get("labels", []), dtype=np.uint8)
        pred_one = (probs >= args.thr).astype(np.uint8)
        if len(gt_one) != len(pred_one):
            raise ValueError(f"Label length mismatch for {src}: len(gt)={len(gt_one)} len(pred)={len(pred_one)}")

        total_tp += int(np.sum((pred_one == 1) & (gt_one == 1)))
        total_fp += int(np.sum((pred_one == 1) & (gt_one == 0)))
        total_fn += int(np.sum((pred_one == 0) & (gt_one == 1)))

        gt_many = np.asarray(entry.get("many_hot_labels", entry.get("labels", [])), dtype=np.uint8)
        if many_probs is not None and len(gt_many) == len(many_probs):
            have_many_hot = True
            pred_many = (many_probs >= args.thr).astype(np.uint8)
            total_tp_mh += int(np.sum((pred_many == 1) & (gt_many == 1)))
            total_fp_mh += int(np.sum((pred_many == 1) & (gt_many == 0)))
            total_fn_mh += int(np.sum((pred_many == 0) & (gt_many == 1)))

        # Scene-based metrics (derived from many_hot_labels when available, else labels)
        gt_scenes = predictions_to_scenes(gt_many)
        pred_scenes = predictions_to_scenes(pred_one)
        _, _, _, (tp, fp, fn) = evaluate_scenes(
            gt_scenes, pred_scenes, n_frames_miss_tolerance=args.tolerance
        )
        scene_tp += tp
        scene_fp += fp
        scene_fn += fn

    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    print("")
    print("[Per-frame one_hot]")
    print(f"Precision: {p * 100:5.2f}%")
    print(f"Recall:    {r * 100:5.2f}%")
    print(f"F1 Score:  {f1 * 100:5.2f}%")

    if have_many_hot:
        p_mh = total_tp_mh / (total_tp_mh + total_fp_mh + 1e-8)
        r_mh = total_tp_mh / (total_tp_mh + total_fn_mh + 1e-8)
        f1_mh = 2 * p_mh * r_mh / (p_mh + r_mh + 1e-8)
        print("")
        print("[Per-frame many_hot]")
        print(f"Precision: {p_mh * 100:5.2f}%")
        print(f"Recall:    {r_mh * 100:5.2f}%")
        print(f"F1 Score:  {f1_mh * 100:5.2f}%")

    scene_p = scene_tp / (scene_tp + scene_fp + 1e-8)
    scene_r = scene_tp / (scene_tp + scene_fn + 1e-8)
    scene_f1 = 2 * scene_p * scene_r / (scene_p + scene_r + 1e-8)
    print("")
    print(f"[Scene-based] tolerance={args.tolerance} frames")
    print(f"Precision: {scene_p * 100:5.2f}%")
    print(f"Recall:    {scene_r * 100:5.2f}%")
    print(f"F1 Score:  {scene_f1 * 100:5.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
