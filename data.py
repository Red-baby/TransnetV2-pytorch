import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime check for optional dep
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None


def _ensure_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for reading frames/videos. "
            "Install with `pip install opencv-python`."
        ) from _cv2_import_error


def _read_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    _ensure_cv2()
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read frame: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width, height = size[1], size[0]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def _read_video_window(video_path: Path, start: int, window: int, size: Tuple[int, int]) -> np.ndarray:
    _ensure_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: List[np.ndarray] = []
    for _ in range(window):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = size[1], size[0]
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if len(frames) != window:
        raise ValueError(f"Video {video_path} ended before fetching {window} frames at {start}.")
    return np.stack(frames, axis=0)


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines


class ShotDataset(Dataset):
    """
    Minimal dataset for TransNetV2 training with PyTorch.

    Manifest format (JSON lines):
        {
          "video_path": "/path/to/video.mp4" OR "frames_dir": "/path/to/frames_dir",
          "labels": [0, 0, 1, ...]  # per-frame binary labels, length == num frames
        }
    """

    def __init__(
        self,
        manifest: Path,
        window: int = 100,
        stride: int = 25,
        resize_hw: Tuple[int, int] = (27, 48),
        shuffle: bool = False,
        seed: int = 13,
    ):
        super().__init__()
        self.entries = load_manifest(manifest)
        self.window = window
        self.stride = stride
        self.resize_hw = resize_hw

        rng = random.Random(seed)
        self.index: List[Tuple[int, int]] = []  # (entry_idx, start_idx)
        for entry_idx, entry in enumerate(self.entries):
            labels = entry["labels"]
            if len(labels) < window:
                continue
            # 沿用 TF 版策略：以转场锚点为中心取窗口，不足窗口的样本直接跳过
            anchors = [i for i, v in enumerate(labels) if v == 1]
            starts: List[int]
            if anchors:
                starts = []
                for a in anchors:
                    start = max(0, a - window // 2)
                    if start + window > len(labels):
                        continue  # 无法取满窗口则跳过
                    starts.append(start)
                # 去重，避免多次命中相同 start
                starts = list(dict.fromkeys(starts))
            else:
                # 无锚点时退回滑窗策略
                starts = list(range(0, len(labels) - window + 1, stride))
            if shuffle:
                rng.shuffle(starts)
            self.index.extend((entry_idx, s) for s in starts)

    def __len__(self) -> int:
        return len(self.index)

    def _load_frames(self, entry: Dict[str, Any], start: int) -> np.ndarray:
        if "frames_dir" in entry:
            frames_dir = Path(entry["frames_dir"])
            frame_files = sorted(frames_dir.glob("*"))
            window_files = frame_files[start : start + self.window]
            if len(window_files) != self.window:
                raise ValueError(f"Not enough frames in {frames_dir} for window starting at {start}.")
            frames = [_read_image(p, self.resize_hw) for p in window_files]
            return np.stack(frames, axis=0)
        if "video_path" in entry:
            return _read_video_window(Path(entry["video_path"]), start, self.window, self.resize_hw)
        raise ValueError("Each manifest entry must contain either `frames_dir` or `video_path`.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry_idx, start = self.index[idx]
        entry = self.entries[entry_idx]
        frames = self._load_frames(entry, start).astype(np.uint8)
        labels = np.asarray(entry["labels"][start : start + self.window], dtype=np.float32)
        many_hot = np.asarray(
            entry.get("many_hot_labels", entry["labels"])[start : start + self.window], dtype=np.float32
        )
        return {
            "video": torch.from_numpy(frames),  # [T, H, W, 3] uint8
            "labels": torch.from_numpy(labels),  # [T] float32
            "many_hot": torch.from_numpy(many_hot),  # [T] float32
        }


def make_dataloader(
    manifest: Path,
    window: int,
    stride: int,
    resize_hw: Tuple[int, int],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 13,
) -> DataLoader:
    dataset = ShotDataset(
        manifest=manifest,
        window=window,
        stride=stride,
        resize_hw=resize_hw,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
