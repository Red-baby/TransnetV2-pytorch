import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


ICUT_LOG_LINE_RE = re.compile(r"POC=(\d+).*?shot-cut=(\d+)", re.IGNORECASE)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    """
    Convert per-frame transition flags (1=transition, 0=scene) to scene segments [start, end] inclusive.
    Same logic as TransNetV2's predictions_to_scenes.
    """
    if predictions.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

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

    return np.asarray(scenes, dtype=np.int32)


def _transitions_from_scenes(scenes: np.ndarray) -> np.ndarray:
    if len(scenes) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([scenes[:-1, 1], scenes[1:, 0]], 1).astype(np.float32)


def evaluate_scenes(
    gt_scenes: np.ndarray,
    pred_scenes: np.ndarray,
    *,
    n_frames_miss_tolerance: int = 2,
) -> Tuple[float, float, float, Tuple[int, int, int]]:
    """
    Scene-based evaluation (precision/recall/F1) with tolerance in frames.
    Adapted from training/metrics_utils.py but without TensorFlow dependency.
    """
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = _transitions_from_scenes(gt_scenes)
    pred_trans = _transitions_from_scenes(pred_scenes)

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


def parse_icut_log(log_path: Path) -> Tuple[List[int], int]:
    """
    Parse icut logfile lines:
      POC=<int>, ..., shot-cut=<0/1>;
    Returns (cut_frames, n_frames_in_log).
    """
    cut_frames: List[int] = []
    max_poc = -1

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ICUT_LOG_LINE_RE.search(line)
            if not m:
                continue
            poc = int(m.group(1))
            shotcut = int(m.group(2))
            max_poc = max(max_poc, poc)
            if shotcut != 0:
                cut_frames.append(poc)

    if max_poc < 0:
        raise ValueError(f"No icut log lines found in {log_path}")

    cut_frames = sorted(set(cut_frames))
    return cut_frames, max_poc + 1


def build_icut_command(
    icutcli: str,
    *,
    video_path: Path,
    log_path: Path,
    width: int,
    height: int,
    bitdepth: int,
    shotcut_threshold: int,
    log_level: int,
    threads: int,
    preset: int,
    keyint: Optional[int],
    min_keyint: Optional[int],
    bframes: Optional[int],
) -> List[str]:
    cmd = [
        icutcli,
        "--input",
        str(video_path),
        "--logfile",
        str(log_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--bitdepth",
        str(bitdepth),
        "--shotcut",
        str(shotcut_threshold),
        "--log",
        str(log_level),
        "--threads",
        str(threads),
        "--icut-preset",
        str(preset),
    ]
    if keyint is not None:
        cmd += ["--keyint", str(keyint)]
    if min_keyint is not None:
        cmd += ["--min-keyint", str(min_keyint)]
    if bframes is not None:
        cmd += ["--bframes", str(bframes)]
    return cmd


def run_icut(cmd: List[str], *, log_path: Optional[Path] = None) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "icutcli failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"exit: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
    if log_path is not None and not log_path.exists():
        raise RuntimeError(
            "icutcli completed successfully but logfile was not created.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"log_path: {log_path}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )


def decode_mp4_to_yuv420p(
    video_path: Path,
    *,
    yuv_path: Path,
) -> Tuple[int, int, int]:
    """
    Decode a video with OpenCV and write raw YUV420p (I420) frames to yuv_path.

    Returns (width, height, n_frames_written).
    """
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for --decode-yuv but is not available. "
            "Install opencv-python (or opencv-python-headless)."
        ) from e

    def _get_resolution(cap: "cv2.VideoCapture") -> Tuple[int, int]:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Could not determine video resolution for: {video_path}")
        return width, height

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    width, height = _get_resolution(cap)

    if (width % 2) != 0 or (height % 2) != 0:
        raise RuntimeError(
            f"Video resolution must be even for YUV420p (I420), got {width}x{height}: {video_path}"
        )

    n_frames = 0
    yuv_path.parent.mkdir(parents=True, exist_ok=True)
    with yuv_path.open("wb") as f:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            yuv_i420 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
            if yuv_i420.shape[1] != width or yuv_i420.shape[0] != (height * 3) // 2:
                raise RuntimeError(
                    f"Unexpected I420 buffer shape {yuv_i420.shape} for {width}x{height} ({video_path})"
                )
            f.write(yuv_i420.tobytes())
            n_frames += 1

    cap.release()
    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    return width, height, n_frames


def get_video_resolution_opencv(video_path: Path) -> Tuple[int, int]:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for --decode-yuv but is not available. "
            "Install opencv-python (or opencv-python-headless)."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video resolution for: {video_path}")
    return width, height


def _positions_of_ones(arr: np.ndarray) -> List[int]:
    return np.nonzero(arr.astype(np.uint8) == 1)[0].astype(int).tolist()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run icut (traditional intra-cut detector) on BBC videos from bbc_manifest.jsonl "
            "and evaluate scene-based precision/recall/F1 against manifest ground truth."
        )
    )
    p.add_argument("--manifest", type=Path, required=True, help="Path to BBC manifest JSONL (must contain video_path).")
    p.add_argument("--icutcli", type=str, default="icutcli", help="Path to icutcli binary (default: icutcli on PATH).")
    p.add_argument("--log-dir", type=Path, default=Path("icut_logs"), help="Where to store icut logfiles.")
    p.add_argument("--force", action="store_true", help="Re-run icut even if logfile already exists.")
    p.add_argument("--max-videos", type=int, default=0, help="Stop after N videos (0 = all).")
    p.add_argument("--match", type=str, default=None, help="Only run entries whose video_path contains this substring.")
    p.add_argument("--keep-going", action="store_true", help="Skip videos that error and continue.")

    # icut options (see icut/source/options.inl)
    p.add_argument("--icut-width", type=int, default=320, help="icut analysis width (video will be scaled).")
    p.add_argument("--icut-height", type=int, default=180, help="icut analysis height (video will be scaled).")
    p.add_argument("--icut-bitdepth", type=int, default=8, help="Target bitdepth for icut input conversion (8/10/12).")
    p.add_argument("--icut-shotcut", type=int, default=40, help="icut shotcut aggressiveness threshold (0-100).")
    p.add_argument("--icut-log-level", type=int, default=-1, help="icut console log level (-1..4).")
    p.add_argument("--icut-threads", type=int, default=0, help="icut analysis threads (0=auto).")
    p.add_argument("--icut-preset", type=int, default=5, help="icut preset [0..9].")
    p.add_argument("--icut-keyint", type=int, default=None, help="Override --keyint (max intra period).")
    p.add_argument("--icut-min-keyint", type=int, default=None, help="Override --min-keyint.")
    p.add_argument("--icut-bframes", type=int, default=None, help="Override --bframes (mini-GOP size).")

    # optional decode path (mp4 -> raw yuv) to work around icut builds without avcodec demux/decoder support.
    p.add_argument(
        "--decode-yuv",
        action="store_true",
        help="Decode mp4 via OpenCV and feed raw YUV420p (I420) to icutcli instead of the original video file.",
    )
    p.add_argument(
        "--yuv-dir",
        type=Path,
        default=Path("icut_yuv"),
        help="Temporary directory to store decoded *.yuv files (used with --decode-yuv).",
    )
    p.add_argument(
        "--force-yuv",
        action="store_true",
        help="Force re-decode YUV even if the target *.yuv already exists (used with --decode-yuv).",
    )

    # evaluation options
    p.add_argument("--tolerance", type=int, default=2, help="Frame miss tolerance for scene-based evaluation.")
    p.add_argument(
        "--cut-offset",
        type=int,
        default=0,
        help="Shift icut cut frame index by this offset (e.g. -1 if icut marks start_next but GT anchors end_prev).",
    )
    p.add_argument(
        "--print-indices",
        action="store_true",
        help="Print per-video indices of predicted/GT transition frames (can be long).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    entries = load_jsonl(args.manifest)
    if args.match:
        entries = [e for e in entries if args.match in str(e.get("video_path", ""))]
    if args.max_videos:
        entries = entries[: args.max_videos]

    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.decode_yuv:
        if args.icut_bitdepth != 8:
            raise ValueError("--decode-yuv currently only supports --icut-bitdepth 8 (OpenCV writes 8-bit I420).")
        args.yuv_dir.mkdir(parents=True, exist_ok=True)

    # totals for per-frame one_hot evaluation against manifest "labels"
    total_tp, total_fp, total_fn = 0, 0, 0
    # totals for scene-based transitions evaluation
    scene_tp, scene_fp, scene_fn = 0, 0, 0

    processed = 0

    for idx, entry in enumerate(entries, start=1):
        video_path_str = entry.get("video_path")
        if not video_path_str:
            msg = f"[{idx}/{len(entries)}] skip: manifest entry missing video_path"
            if args.keep_going:
                print(msg, file=sys.stderr)
                continue
            raise ValueError(msg)

        video_path = Path(video_path_str)
        log_path = args.log_dir / f"{video_path.stem}.icut.log"

        try:
            icut_input_path = video_path
            icut_width = args.icut_width
            icut_height = args.icut_height

            if args.decode_yuv:
                yuv_path = args.yuv_dir / f"{video_path.stem}.yuv"
                if args.force_yuv or not yuv_path.exists():
                    width, height, _ = decode_mp4_to_yuv420p(video_path, yuv_path=yuv_path)
                else:
                    width, height = get_video_resolution_opencv(video_path)

                icut_input_path = yuv_path
                icut_width = width
                icut_height = height

            if args.force or not log_path.exists():
                cmd = build_icut_command(
                    args.icutcli,
                    video_path=icut_input_path,
                    log_path=log_path,
                    width=icut_width,
                    height=icut_height,
                    bitdepth=args.icut_bitdepth,
                    shotcut_threshold=args.icut_shotcut,
                    log_level=args.icut_log_level,
                    threads=args.icut_threads,
                    preset=args.icut_preset,
                    keyint=args.icut_keyint,
                    min_keyint=args.icut_min_keyint,
                    bframes=args.icut_bframes,
                )
                run_icut(cmd, log_path=log_path)

            cut_frames, n_frames_in_log = parse_icut_log(log_path)

            gt_one = np.asarray(entry.get("labels", []), dtype=np.uint8)
            if gt_one.size == 0:
                raise ValueError("Manifest entry missing non-empty 'labels'.")
            gt_many = np.asarray(entry.get("many_hot_labels", entry.get("labels", [])), dtype=np.uint8)
            if gt_many.size != gt_one.size:
                raise ValueError(f"len(labels)={gt_one.size} != len(many_hot_labels)={gt_many.size}")

            n_frames_gt = int(gt_one.size)
            if n_frames_in_log != n_frames_gt:
                print(
                    f"[warn] frame count mismatch for {video_path}: icut_log={n_frames_in_log} vs gt={n_frames_gt}",
                    file=sys.stderr,
                )

            pred_one = np.zeros((n_frames_gt,), dtype=np.uint8)
            for poc in cut_frames:
                frame_idx = poc + args.cut_offset
                if frame_idx <= 0:  # ignore cut on first frame (and negative after offset)
                    continue
                if frame_idx >= n_frames_gt:
                    continue
                pred_one[frame_idx] = 1

            # Per-frame (anchor) metrics
            total_tp += int(np.sum((pred_one == 1) & (gt_one == 1)))
            total_fp += int(np.sum((pred_one == 1) & (gt_one == 0)))
            total_fn += int(np.sum((pred_one == 0) & (gt_one == 1)))

            # Scene-based metrics (GT from many_hot_labels; pred from icut cut frames)
            gt_scenes = predictions_to_scenes(gt_many)
            pred_scenes = predictions_to_scenes(pred_one)
            _, _, _, (tp, fp, fn) = evaluate_scenes(
                gt_scenes, pred_scenes, n_frames_miss_tolerance=args.tolerance
            )
            scene_tp += tp
            scene_fp += fp
            scene_fn += fn

            processed += 1

            if args.print_indices:
                print(f"[{idx}/{len(entries)}] {video_path}")
                print(f"  gt_one idx:   {_positions_of_ones(gt_one)}")
                print(f"  pred_one idx: {_positions_of_ones(pred_one)}")
                print(f"  icut cuts:    {cut_frames}")
            else:
                print(f"[{idx}/{len(entries)}] {video_path} cuts={len(cut_frames)}", flush=True)

        except Exception as e:
            if args.keep_going:
                print(f"[error] {video_path}: {e}", file=sys.stderr)
                continue
            raise

    # Per-frame one_hot
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    # Scene-based transitions
    scene_p = scene_tp / (scene_tp + scene_fp + 1e-8)
    scene_r = scene_tp / (scene_tp + scene_fn + 1e-8)
    scene_f1 = 2 * scene_p * scene_r / (scene_p + scene_r + 1e-8)

    print("")
    print(f"Processed videos: {processed}/{len(entries)}")
    print("")
    print("[Per-frame one_hot] (icut cut frames vs manifest labels)")
    print(f"Precision: {p * 100:5.2f}%")
    print(f"Recall:    {r * 100:5.2f}%")
    print(f"F1 Score:  {f1 * 100:5.2f}%")
    print("")
    print(f"[Scene-based] tolerance={args.tolerance} frames")
    print(f"Precision: {scene_p * 100:5.2f}%")
    print(f"Recall:    {scene_r * 100:5.2f}%")
    print(f"F1 Score:  {scene_f1 * 100:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
