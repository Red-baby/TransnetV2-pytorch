#!/usr/bin/env python3
"""
dual_scene_detect.py - Dual Scene Detection using TransNetV2 and icut

This script runs both TransNetV2 (deep learning) and icut (traditional algorithm) 
scene detection on the same video, producing two independent sets of results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Import from existing scripts (they must be in the same directory)
try:
    from infer import load_frames, sliding_window_predict, build_keyframes, TransNetV2
    from icut_infer import detect_video_params, run_icutcli, parse_icut_log
except ImportError as e:
    print(f"Error: Cannot import required modules: {e}")
    print("Make sure infer.py and icut_infer.py are in the same directory.")
    sys.exit(1)

import torch
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual scene detection using TransNetV2 and icut."
    )
    
    # Input
    parser.add_argument(
        "--video-path", 
        type=Path, 
        required=True, 
        help="Path to the video file (MP4)."
    )
    
    # Model/Tool paths
    parser.add_argument(
        "--transnetv2-weights",
        type=Path,
        required=True,
        help="Path to TransNetV2 model weights (.pth)."
    )
    parser.add_argument(
        "--icutcli-path",
        type=Path,
        default=Path("icutcli.exe") if sys.platform == "win32" else Path("icutcli"),
        help="Path to icutcli executable (default: icutcli.exe)."
    )
    
    # Detection parameters
    parser.add_argument(
        "--transnetv2-threshold",
        type=float,
        default=0.5,
        help="TransNetV2 decision threshold (default: 0.5)."
    )
    parser.add_argument(
        "--icut-threshold",
        type=int,
        default=40,
        help="icut scene detection threshold 0-100 (default: 40)."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="TransNetV2 sliding window size (default: 100)."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="TransNetV2 stride between windows (default: 25)."
    )
    
    # Keyframe constraints (applied to both methods)
    parser.add_argument(
        "--min-keyframe",
        type=int,
        help="Minimum keyframe interval (frames)."
    )
    parser.add_argument(
        "--max-keyframe",
        type=int,
        help="Maximum keyframe interval (frames)."
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Frame rate (for PTS calculation, auto-detected if not specified)."
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for keyframe files (default: current directory)."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison report (JSON format)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Only process the first N frames (for testing)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for TransNetV2 (default: cuda if available, else cpu)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output."
    )
    
    return parser.parse_args()


def get_total_frames(video_path: Path) -> int:
    """Get total number of frames in video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def run_transnetv2_detection(
    video_path: Path,
    weights_path: Path,
    threshold: float,
    window: int,
    stride: int,
    device: str,
    max_frames: Optional[int],
    verbose: bool
) -> List[int]:
    """
    Run TransNetV2 scene detection.
    
    Returns:
        List of scene start frame indices.
    """
    print("=" * 60)
    print("TRANSNETV2 DETECTION")
    print("=" * 60)
    
    # Load frames
    print(f"Loading frames from {video_path}...")
    frames_np = load_frames(
        type('Args', (), {
            'video_path': video_path,
            'frames_dir': None,
            'max_frames': max_frames
        })(),
        resize_hw=(27, 48)
    )
    frames = torch.from_numpy(frames_np)
    print(f"Loaded {len(frames)} frames.")
    
    # Load model
    print(f"Loading TransNetV2 model from {weights_path}...")
    model = TransNetV2()
    state = torch.load(weights_path, map_location=device)
    state = state["model_state"] if "model_state" in state else state
    model.load_state_dict(state, strict=False)
    model.to(device)
    print("Model loaded successfully.")
    
    # Run prediction
    print(f"Running prediction (window={window}, stride={stride})...")
    probs = sliding_window_predict(
        model, frames, window, stride, torch.device(device)
    ).detach().cpu().numpy()
    
    # Extract boundaries
    boundaries = [i for i, p in enumerate(probs) if p >= threshold]
    print(f"Detected {len(boundaries)} boundaries at threshold {threshold}")
    
    # Convert to scene starts
    scene_starts = [0]
    if boundaries:
        current = boundaries[0]
        for idx in boundaries[1:]:
            if idx != current + 1:
                next_start = current + 1
                if next_start < len(probs):
                    scene_starts.append(next_start)
            current = idx
        next_start = current + 1
        if next_start < len(probs):
            scene_starts.append(next_start)
    
    print(f"Scene start frames: {scene_starts[:10]}{'...' if len(scene_starts) > 10 else ''}")
    print(f"Total scenes detected: {len(scene_starts)}\n")
    
    return scene_starts


def run_icut_detection(
    video_path: Path,
    icutcli_path: Path,
    width: int,
    height: int,
    bitdepth: int,
    fps_num: int,
    fps_denom: int,
    threshold: int,
    max_frames: Optional[int],
    verbose: bool
) -> List[int]:
    """
    Run icut scene detection.
    
    Returns:
        List of scene start frame indices.
    """
    print("=" * 60)
    print("ICUT DETECTION")
    print("=" * 60)
    
    # Debug: Print icutcli path and check if executable
    print(f"icutcli path: {icutcli_path}")
    print(f"icutcli absolute path: {icutcli_path.absolute()}")
    print(f"icutcli exists: {icutcli_path.exists()}")
    print(f"icutcli is file: {icutcli_path.is_file()}")
    if icutcli_path.exists():
        import os
        print(f"icutcli is executable: {os.access(icutcli_path, os.X_OK)}")
    print()
    
    # Run icutcli
    print(f"Running icutcli on {video_path}...")
    log_path = run_icutcli(
        icutcli_path,
        video_path,
        width,
        height,
        bitdepth,
        fps_num,
        fps_denom,
        None,  # max_keyframe (not used here, applied later)
        None,  # min_keyframe (not used here, applied later)
        threshold,
        max_frames,
        verbose
    )

    
    # Parse log
    print(f"Parsing icut log...")
    scene_starts = parse_icut_log(log_path, verbose)
    
    if not scene_starts:
        print("Warning: No scenes detected by icut, adding frame 0.")
        scene_starts = [0]
    
    print(f"Scene start frames: {scene_starts[:10]}{'...' if len(scene_starts) > 10 else ''}")
    print(f"Total scenes detected: {len(scene_starts)}\n")
    
    # Clean up log file
    if not verbose:
        try:
            log_path.unlink()
        except:
            pass
    
    return scene_starts


def write_keyframe_file(path: Path, keyframes: List[Tuple[int, int]]) -> None:
    """Write keyframe list to file in POC/PTS format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"#{len(keyframes):09d}"] + [f"{frame} {flag}" for frame, flag in keyframes]
    path.write_text("\n".join(lines), encoding="ascii")


def output_results(
    output_dir: Path, 
    method_name: str, 
    keyframes: List[Tuple[int, int]], 
    fps: Optional[float]
) -> None:
    """
    Output keyframe results for one method.
    
    Args:
        output_dir: Output directory
        method_name: "transnetv2" or "icut"
        keyframes: List of (frame, flag) tuples
        fps: Frame rate for PTS calculation
    """
    # Write POC file
    poc_path = output_dir / f"keyframe_POC_{method_name}.txt"
    write_keyframe_file(poc_path, keyframes)
    print(f"✓ Saved {len(keyframes)} keyframes to {poc_path}")
    
    # Write PTS file if fps is specified
    if fps:
        ms_per_frame = 1000.0 / fps
        pts_keyframes = [
            (int(round(frame * ms_per_frame)), flag) for frame, flag in keyframes
        ]
        pts_path = output_dir / f"keyframe_PTS_{method_name}.txt"
        write_keyframe_file(pts_path, pts_keyframes)
        print(f"✓ Saved PTS keyframes to {pts_path}")


def generate_comparison_report(
    transnetv2_keyframes: List[Tuple[int, int]],
    icut_keyframes: List[Tuple[int, int]],
    output_dir: Path,
    video_path: Path,
    total_frames: int
) -> None:
    """Generate a JSON comparison report."""
    transnetv2_frames = {frame for frame, _ in transnetv2_keyframes}
    icut_frames = {frame for frame, _ in icut_keyframes}
    
    common = transnetv2_frames & icut_frames
    transnetv2_only = transnetv2_frames - icut_frames
    icut_only = icut_frames - transnetv2_frames
    
    # Calculate agreement rate
    total_unique = len(transnetv2_frames | icut_frames)
    agreement_rate = len(common) / total_unique if total_unique > 0 else 0
    
    report = {
        "video": str(video_path.name),
        "total_frames": total_frames,
        "transnetv2": {
            "scenes": len(transnetv2_keyframes),
            "keyframes": list(sorted(transnetv2_frames))
        },
        "icut": {
            "scenes": len(icut_keyframes),
            "keyframes": list(sorted(icut_frames))
        },
        "comparison": {
            "common_keyframes": list(sorted(common)),
            "common_count": len(common),
            "transnetv2_only": list(sorted(transnetv2_only)),
            "transnetv2_only_count": len(transnetv2_only),
            "icut_only": list(sorted(icut_only)),
            "icut_only_count": len(icut_only),
            "agreement_rate": round(agreement_rate, 4)
        }
    }
    
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Comparison report saved to {report_path}")
    print(f"  Agreement rate: {agreement_rate:.1%}")
    print(f"  Common keyframes: {len(common)}")
    print(f"  TransNetV2 only: {len(transnetv2_only)}")
    print(f"  icut only: {len(icut_only)}")


def main():
    args = parse_args()
    
    # Validate inputs
    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not args.transnetv2_weights.exists():
        print(f"Error: TransNetV2 weights not found: {args.transnetv2_weights}")
        sys.exit(1)
    
    if not args.icutcli_path.exists():
        print(f"Error: icutcli not found: {args.icutcli_path}")
        print("Please specify the correct path with --icutcli-path")
        sys.exit(1)
    
    # Detect video parameters
    print("Detecting video parameters...")
    width, height, fps = detect_video_params(args.video_path)
    if args.fps:
        fps = args.fps
    if not width or not height or not fps:
        print("Error: Could not detect video parameters.")
        sys.exit(1)
    
    print(f"Video: {args.video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.3f}\n")
    
    # Get total frames
    try:
        total_frames = get_total_frames(args.video_path)
        if args.max_frames:
            total_frames = min(total_frames, args.max_frames)
        print(f"Total frames to process: {total_frames}\n")
    except Exception as e:
        print(f"Warning: Could not get total frames: {e}")
        total_frames = args.max_frames or 10000
    
    # Run TransNetV2 detection
    transnetv2_scenes = run_transnetv2_detection(
        args.video_path,
        args.transnetv2_weights,
        args.transnetv2_threshold,
        args.window,
        args.stride,
        args.device,
        args.max_frames,
        args.verbose
    )
    
    # Run icut detection
    fps_num = int(fps * 1000)
    fps_denom = 1000
    icut_scenes = run_icut_detection(
        args.video_path,
        args.icutcli_path,
        width,
        height,
        8,  # bitdepth
        fps_num,
        fps_denom,
        args.icut_threshold,
        args.max_frames,
        args.verbose
    )
    
    # Apply keyframe constraints to both methods
    print("=" * 60)
    print("APPLYING KEYFRAME CONSTRAINTS")
    print("=" * 60)
    
    if args.min_keyframe or args.max_keyframe:
        print(f"min-keyframe: {args.min_keyframe or 'None'}")
        print(f"max-keyframe: {args.max_keyframe or 'None'}\n")
    else:
        print("No constraints specified.\n")
    
    # TransNetV2 keyframes
    print("TransNetV2:")
    transnetv2_keyframes, transnetv2_dropped = build_keyframes(
        transnetv2_scenes,
        total_frames,
        args.min_keyframe,
        args.max_keyframe
    )
    if transnetv2_dropped:
        print(f"  Dropped {len(transnetv2_dropped)} scenes due to min-keyframe constraint")
    print(f"  Final keyframes: {len(transnetv2_keyframes)}\n")
    
    # icut keyframes
    print("icut:")
    icut_keyframes, icut_dropped = build_keyframes(
        icut_scenes,
        total_frames,
        args.min_keyframe,
        args.max_keyframe
    )
    if icut_dropped:
        print(f"  Dropped {len(icut_dropped)} scenes due to min-keyframe constraint")
    print(f"  Final keyframes: {len(icut_keyframes)}\n")
    
    # Output results
    print("=" * 60)
    print("WRITING OUTPUT FILES")
    print("=" * 60)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_results(args.output_dir, "transnetv2", transnetv2_keyframes, fps)
    output_results(args.output_dir, "icut", icut_keyframes, fps)
    
    # Generate comparison report
    if args.compare:
        print("\n" + "=" * 60)
        print("COMPARISON REPORT")
        print("=" * 60)
        generate_comparison_report(
            transnetv2_keyframes,
            icut_keyframes,
            args.output_dir,
            args.video_path,
            total_frames
        )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
