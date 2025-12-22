import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data import _ensure_cv2  # type: ignore
from transnetv2_pytorch import TransNetV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference/validation with PyTorch TransNetV2.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-path", type=Path, help="Path to a video file.")
    src.add_argument("--frames-dir", type=Path, help="Directory with RGB frames (sorted).")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights (.pth).")
    parser.add_argument("--window", type=int, default=100, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=25, help="Stride between windows.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for boundaries.")
    parser.add_argument("--max-frames", type=int, help="Only process the first N frames.")
    parser.add_argument("--save-npy", type=Path, help="Optional path to save per-frame probabilities (.npy).")
    return parser.parse_args()


def _read_all_frames_from_dir(
    frames_dir: Path, resize_hw: Tuple[int, int], max_frames: int | None
) -> np.ndarray:
    # Lazy import from local module `data` (same directory) so this file can be
    # executed directly from inside the training_pytorch folder or from elsewhere.
    from data import _read_image  # lazy import to reuse resize+RGB

    files = sorted(frames_dir.glob("*"))
    if max_frames is not None:
        files = files[:max_frames]
    if not files:
        raise ValueError(f"No frames found in {frames_dir}")
    frames = [_read_image(p, resize_hw) for p in files]
    return np.stack(frames, axis=0).astype(np.uint8)


def _read_all_frames_from_video(
    video_path: Path, resize_hw: Tuple[int, int], max_frames: int | None
) -> np.ndarray:
    _ensure_cv2()
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    width, height = resize_hw[1], resize_hw[0]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and total > 0:
        total = min(total, max_frames)
    print(f"[Decode] Start reading video: {video_path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
        if len(frames) % 500 == 0:
            if total > 0:
                print(f"[Decode] {len(frames)}/{total} frames")
            else:
                print(f"[Decode] {len(frames)} frames")
    cap.release()
    if not frames:
        raise ValueError(f"Video {video_path} contained no readable frames.")
    print(f"[Decode] Done. Total frames: {len(frames)}")
    return np.stack(frames, axis=0).astype(np.uint8)


def load_frames(args: argparse.Namespace, resize_hw: Tuple[int, int]) -> np.ndarray:
    if args.frames_dir:
        return _read_all_frames_from_dir(args.frames_dir, resize_hw, args.max_frames)
    return _read_all_frames_from_video(args.video_path, resize_hw, args.max_frames)


def sliding_window_predict(
    model: TransNetV2,
    frames: torch.Tensor,
    window: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    num_frames = frames.shape[0]
    if num_frames < window:
        raise ValueError(f"Not enough frames ({num_frames}) for window size {window}.")
    scores = torch.zeros(num_frames, device=device)
    counts = torch.zeros(num_frames, device=device)
    total_windows = max(1, ((num_frames - window) // stride) + 1)
    for idx, start in enumerate(range(0, num_frames - window + 1, stride), start=1):
        window_frames = frames[start : start + window].unsqueeze(0).to(device)
        outputs = model(window_frames)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(logits.squeeze(0).squeeze(-1))  # [window]
        scores[start : start + window] += probs
        counts[start : start + window] += 1
        if idx % 50 == 0 or idx == total_windows:
            print(f"[Infer] window {idx}/{total_windows}")
    counts = counts.clamp(min=1)
    return scores / counts


def main():
    args = parse_args()
    device = torch.device(args.device)
    frames_np = load_frames(args, resize_hw=(27, 48))
    frames = torch.from_numpy(frames_np)  # [T, H, W, 3], uint8

    model = TransNetV2()
    state = torch.load(args.weights, map_location=device)
    state = state["model_state"] if "model_state" in state else state
    model.load_state_dict(state, strict=False)
    model.to(device)

    probs = sliding_window_predict(model, frames, args.window, args.stride, device).cpu().numpy()
    boundaries = [i for i, p in enumerate(probs) if p >= args.threshold]
    print(f"Detected {len(boundaries)} boundaries at frames: {boundaries}")

    if args.save_npy:
        args.save_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy, probs)
        print(f"Saved per-frame probabilities to {args.save_npy}")


if __name__ == "__main__":
    main()
