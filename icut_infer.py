#!/usr/bin/env python3
"""
icut_infer.py - Wrapper script for icutcli scene detection tool

This script provides a similar interface to infer.py but uses icutcli for scene detection.
It supports min/max keyframe constraints and PTS output.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scene detection using icutcli with keyframe constraints."
    )
    
    # Input source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-path", type=Path, help="Path to a video file.")
    src.add_argument("--yuv-path", type=Path, help="Path to a YUV raw file.")
    
    # Required parameters for YUV input
    parser.add_argument("--width", type=int, help="Video width (required for YUV).")
    parser.add_argument("--height", type=int, help="Video height (required for YUV).")
    parser.add_argument("--bitdepth", type=int, default=8, choices=[8, 10], 
                       help="Bit depth (required for YUV, default: 8).")
    
    # icutcli executable
    parser.add_argument(
        "--icutcli-path",
        type=Path,
        default=Path("icutcli.exe") if sys.platform == "win32" else Path("icutcli"),
        help="Path to icutcli executable.",
    )
    
    # Detection parameters
    parser.add_argument("--fps", type=float, help="Frame rate (for PTS calculation).")
    parser.add_argument(
        "--threshold",
        type=int,
        default=40,
        help="Scene detection threshold (0-100, default: 40).",
    )
    parser.add_argument(
        "--min-keyframe",
        type=int,
        help="Drop scene starts closer than this distance to the previous keyframe.",
    )
    parser.add_argument(
        "--max-keyframe",
        type=int,
        help="Insert keyframes if scene starts are farther apart than this value.",
    )
    
    # Output files
    parser.add_argument(
        "--keyframe-poc",
        type=Path,
        default=Path("keyframe_POC.txt"),
        help="Output keyframe POC list path.",
    )
    parser.add_argument(
        "--keyframe-pts",
        type=Path,
        default=Path("keyframe_PTS.txt"),
        help="Output keyframe PTS list path (only used with --fps).",
    )
    
    # Additional options
    parser.add_argument("--max-frames", type=int, help="Only process the first N frames.")
    parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output from icutcli."
    )
    
    return parser.parse_args()


def detect_video_params(video_path: Path) -> Tuple[int, int, float]:
    """
    Detect video parameters using ffprobe.
    
    Returns:
        (width, height, fps)
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        
        width = stream["width"]
        height = stream["height"]
        
        # Parse frame rate (e.g., "30000/1001" or "30/1")
        fps_str = stream["r_frame_rate"]
        num, denom = map(int, fps_str.split("/"))
        fps = num / denom
        
        return width, height, fps
    except (subprocess.CalledProcessError, KeyError, ValueError, FileNotFoundError) as e:
        print(f"Warning: Failed to detect video parameters: {e}")
        print("Please specify --width, --height, and --fps manually.")
        return None, None, None


def run_icutcli(
    icutcli_path: Path,
    input_path: Path,
    width: int,
    height: int,
    bitdepth: int,
    fps_num: int,
    fps_denom: int,
    max_keyframe: Optional[int],
    min_keyframe: Optional[int],
    threshold: int,
    max_frames: Optional[int],
    verbose: bool,
) -> Path:
    """
    Run icutcli and return the path to the log file.
    """
    # Create a temporary log file
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="icut_", text=True)
    os.close(log_fd)
    log_path = Path(log_path)
    
    # Build icutcli command
    # Use absolute path to avoid issues with relative paths on Linux
    icutcli_absolute = icutcli_path.absolute()
    cmd = [
        str(icutcli_absolute),
        "--input", str(input_path),
        "--width", str(width),
        "--height", str(height),
        "--bitdepth", str(bitdepth),
        "--fps", str(fps_num),
        "--fps-denom", str(fps_denom),
        "--logfile", str(log_path),
        "--shotcut", str(threshold),
    ]
    
    if max_keyframe:
        cmd.extend(["--keyint", str(max_keyframe)])
    if min_keyframe:
        cmd.extend(["--min-keyint", str(min_keyframe)])
    if max_frames:
        cmd.extend(["--frames", str(max_frames)])
    
    # Run icutcli
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        
        if verbose or result.returncode != 0:
            print("=== icutcli stdout ===")
            print(result.stdout)
            if result.stderr:
                print("=== icutcli stderr ===")
                print(result.stderr)
        
        if result.returncode != 0:
            print(f"Warning: icutcli exited with code {result.returncode}")
        
        return log_path
        
    except Exception as e:
        print(f"Error running icutcli: {e}")
        raise


def parse_icut_log(log_path: Path, verbose: bool = False) -> List[int]:
    """
    Parse icutcli log file to extract scene start frames.
    
    Log format (from analyses.c line 448):
        POC=%d, type=%d, COI=%d, GOP=%d, order=%d, layer=%d, shot-cut=%d;
    
    Where:
        POC: Picture Order Count (frame index)
        type: Frame type (1=IDR, 2=I, 3=P, 4=BREF, 5=B)
        shot-cut: 1 if scene cut detected, 0 otherwise
    
    Returns:
        List of frame indices where scenes start (frames with type=IDR/I or shot-cut=1).
    """
    scene_starts = []
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Skip comment lines
                if line.startswith("#"):
                    continue
                
                # Match the exact format: POC=%d, type=%d, COI=%d, GOP=%d, order=%d, layer=%d, shot-cut=%d;
                match = re.match(
                    r'POC=(\d+),\s*type=(\d+),\s*COI=(\d+),\s*GOP=(\d+),\s*order=(\d+),\s*layer=(\d+),\s*shot-cut=(\d+);',
                    line.strip()
                )
                
                if match:
                    poc = int(match.group(1))
                    frame_type = int(match.group(2))
                    shot_cut = int(match.group(7))
                    
                    # Extract scene starts:
                    # - Type 1 (IDR) or Type 2 (I) frames
                    # - Or any frame with shot-cut=1
                    is_intra = (frame_type == 1 or frame_type == 2)  # IDR or I
                    is_scene_cut = (shot_cut == 1)
                    
                    if is_intra and is_scene_cut:
                        # This is a scene cut with I/IDR frame
                        if poc not in scene_starts:
                            scene_starts.append(poc)
                            if verbose:
                                print(f"Scene cut at POC {poc}: type={frame_type}, shot-cut={shot_cut}")
                    elif is_intra:
                        # Regular keyframe (not a scene cut, but still important)
                        # We may want to include these depending on use case
                        # For now, include all intra frames
                        if poc not in scene_starts:
                            scene_starts.append(poc)
                            if verbose:
                                print(f"Keyframe at POC {poc}: type={frame_type} (no scene cut)")
        
        scene_starts.sort()
        return scene_starts
        
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return []


def build_keyframes(
    scene_starts: List[int],
    num_frames: int,
    min_keyframe: Optional[int],
    max_keyframe: Optional[int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Build keyframe list with min/max constraints.
    
    This function is adapted from infer.py.
    
    Returns:
        (keyframes, dropped)
        keyframes: List of (frame_idx, flag) tuples where flag=1 for scene cuts, 0 for inserted
        dropped: List of frame indices that were dropped due to min_keyframe constraint
    """
    if min_keyframe is not None and min_keyframe < 0:
        raise ValueError("--min-keyframe must be >= 0")
    if max_keyframe is not None and max_keyframe <= 0:
        raise ValueError("--max-keyframe must be > 0")
    
    min_keyframe = min_keyframe or 0
    keyframes: List[Tuple[int, int]] = []
    dropped: List[int] = []
    last: Optional[int] = None
    
    for start in scene_starts:
        if last is None:
            flag = 0 if start == 0 else 1
            keyframes.append((start, flag))
            last = start
            continue
        if max_keyframe:
            while last + max_keyframe < start:
                last += max_keyframe
                keyframes.append((last, 0))
        if min_keyframe and start - last < min_keyframe:
            dropped.append(start)
            continue
        if start != last:
            keyframes.append((start, 1))
            last = start
    
    if max_keyframe and last is not None:
        end_frame = num_frames - 1
        while last + max_keyframe <= end_frame:
            last += max_keyframe
            if keyframes[-1][0] != last:
                keyframes.append((last, 0))
    
    return keyframes, dropped


def write_keyframe_file(path: Path, keyframes: List[Tuple[int, int]]) -> None:
    """Write keyframe list to file in POC/PTS format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"#{len(keyframes):09d}"] + [f"{frame} {flag}" for frame, flag in keyframes]
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    args = parse_args()
    
    # Validate input
    if args.yuv_path:
        if not args.width or not args.height:
            print("Error: --width and --height are required for YUV input.")
            sys.exit(1)
        input_path = args.yuv_path
        width = args.width
        height = args.height
        fps = args.fps or 25.0  # Default FPS
    else:
        input_path = args.video_path
        # Try to detect video parameters
        detected_width, detected_height, detected_fps = detect_video_params(input_path)
        width = args.width or detected_width
        height = args.height or detected_height
        fps = args.fps or detected_fps
        
        if not width or not height:
            print("Error: Could not detect video dimensions. Please specify --width and --height.")
            sys.exit(1)
        if not fps:
            print("Warning: Could not detect FPS. Using default 25.0")
            fps = 25.0
    
    # Convert FPS to numerator/denominator
    fps_num = int(fps * 1000)
    fps_denom = 1000
    
    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height}, Bit depth: {args.bitdepth}, FPS: {fps:.3f}")
    
    # Check icutcli exists
    if not args.icutcli_path.exists():
        print(f"Error: icutcli not found at {args.icutcli_path}")
        print("Please specify the correct path with --icutcli-path")
        sys.exit(1)
    
    # Run icutcli
    print("Running icutcli...")
    log_path = run_icutcli(
        args.icutcli_path,
        input_path,
        width,
        height,
        args.bitdepth,
        fps_num,
        fps_denom,
        args.max_keyframe,
        args.min_keyframe,
        args.threshold,
        args.max_frames,
        args.verbose,
    )
    
    # Parse results
    print(f"Parsing log: {log_path}")
    scene_starts = parse_icut_log(log_path, args.verbose)
    
    if not scene_starts:
        print("Warning: No scene cuts detected. Adding frame 0 as the first scene.")
        scene_starts = [0]
    
    print(f"Detected {len(scene_starts)} scene starts: {scene_starts[:10]}{'...' if len(scene_starts) > 10 else ''}")
    
    # Estimate total frames (this is a rough estimate)
    # In production, we'd need to get this from icutcli or video metadata
    if scene_starts:
        num_frames = max(scene_starts) + 100  # Add buffer
    else:
        num_frames = args.max_frames or 1000
    
    # Build keyframes with constraints
    if args.min_keyframe is not None or args.max_keyframe is not None:
        keyframes, dropped = build_keyframes(
            scene_starts,
            num_frames,
            args.min_keyframe,
            args.max_keyframe,
        )
        
        if dropped:
            print(f"Dropped {len(dropped)} scene starts due to --min-keyframe: {dropped}")
        
        # Write POC file
        write_keyframe_file(args.keyframe_poc, keyframes)
        print(f"Saved POC keyframes ({len(keyframes)}) to {args.keyframe_poc}")
        
        # Write PTS file if FPS is specified
        if args.fps:
            ms_per_frame = 1000.0 / fps
            pts_keyframes = [
                (int(round(frame * ms_per_frame)), flag) for frame, flag in keyframes
            ]
            write_keyframe_file(args.keyframe_pts, pts_keyframes)
            print(f"Saved PTS keyframes ({len(pts_keyframes)}) to {args.keyframe_pts}")
    else:
        # No constraints, just output scene starts
        keyframes = [(frame, 1) for frame in scene_starts]
        write_keyframe_file(args.keyframe_poc, keyframes)
        print(f"Saved POC keyframes ({len(keyframes)}) to {args.keyframe_poc}")
    
    # Cleanup temp log file (optional - keep for debugging)
    if not args.verbose:
        try:
            log_path.unlink()
        except:
            pass
    else:
        print(f"Log file kept at: {log_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
