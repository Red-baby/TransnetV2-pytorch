import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def segments_to_labels(segments: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    """Convert list of [start, end] (inclusive) segments to labels/many_hot."""
    if not segments:
        return [], []
    max_idx = max(end for _, end in segments)
    length = max_idx + 1
    labels = [0] * length
    many_hot = [0] * length

    # transitions between consecutive segments
    for (s_prev, e_prev), (s_next, e_next) in zip(segments[:-1], segments[1:]):
        gap_start = e_prev + 1
        gap_end = s_next - 1
        if gap_start > gap_end:  # hard cut, contiguous
            anchor = e_prev  # previous segment last frame
            labels[anchor] = 1
            many_hot[anchor] = 1
        else:  # dissolve/gradual: gap frames belong to transition
            anchor = (gap_start + gap_end) // 2  # center frame of gap
            labels[anchor] = 1
            for idx in range(gap_start, gap_end + 1):
                many_hot[idx] = 1

    return labels, many_hot


def parse_blocks(txt_path: Path) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, Optional[int]]]:
    """
    Parse txt into:
      boundaries[video_stem] = [(end_prev, start_next), ...]
      declared_len[video_stem] = int or None
    """
    boundaries: Dict[str, List[Tuple[int, int]]] = {}
    declared_len: Dict[str, Optional[int]] = {}

    current_stem: Optional[str] = None
    current_bounds: List[Tuple[int, int]] = []

    def flush():
        nonlocal current_stem, current_bounds
        if current_stem is not None:
            boundaries[current_stem] = current_bounds
        current_stem = None
        current_bounds = []

    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                flush()
                continue

            if current_stem is None:
                # header line: "<video>.mp4 <something_like_length>"
                parts = line.split()
                video_name = parts[0]
                stem = Path(video_name).stem
                current_stem = stem
                current_bounds = []
                dlen: Optional[int] = None
                if len(parts) > 1:
                    try:
                        dlen = int(parts[1])
                    except ValueError:
                        dlen = None
                declared_len[stem] = dlen
                continue

            # boundary line: "end_prev,start_next"
            pair_text = line.split()[0]
            if "," not in pair_text:
                continue
            a_str, b_str = pair_text.split(",", 1)
            try:
                end_prev = int(a_str)
                start_next = int(b_str)
            except ValueError:
                continue
            current_bounds.append((end_prev, start_next))

    flush()
    return boundaries, declared_len


def boundaries_to_segments(
    bounds: List[Tuple[int, int]], declared_len: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Convert [(end_prev, start_next), ...] to segments [(start, end), ...].

    Example bounds:
      130,131
      254,255
    -> segments:
      [0,130], [131,254], ...

    The last bound's second number is ignored (often out-of-range).
    This function also tolerates unsorted/noisy bounds by sorting by end_prev
    and dropping any bound that would create a backward segment.

    Returns (segments, num_dropped_bounds).
    """
    if not bounds:
        if declared_len is None:
            return [], 0
        return [(0, declared_len - 1)], 0

    sorted_bounds = sorted(bounds, key=lambda x: (x[0], x[1]))
    terminal = sorted_bounds[-1]
    others = sorted_bounds[:-1]

    cleaned: List[Tuple[int, int]] = []
    dropped = 0
    start = 0
    for end_prev, start_next in others:
        if end_prev < start:
            dropped += 1
            continue
        if start_next <= start:
            # avoid moving backward; fall back to contiguous next start
            start_next = end_prev + 1
        cleaned.append((end_prev, start_next))
        start = start_next

    segments: List[Tuple[int, int]] = []
    start = 0
    for end_prev, start_next in cleaned:
        segments.append((start, end_prev))
        start = start_next

    last_end = terminal[0]
    if declared_len is not None:
        last_end = min(last_end, declared_len - 1)
    if start <= last_end:
        segments.append((start, last_end))

    return segments, dropped


def main():
    parser = argparse.ArgumentParser(description="Convert transition-boundaries txt to JSONL manifest.")
    parser.add_argument("--txt", required=True, help="Path to txt file")
    parser.add_argument("--output", required=True, help="Path to write JSONL")
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="If set, frames_dir will be frames_root/<video_stem>; otherwise just <video_stem>.",
    )
    parser.add_argument(
        "--frames-filter",
        type=Path,
        default=None,
        help="If set, only keep entries whose stem exists as subfolder under this path.",
    )
    args = parser.parse_args()

    txt_path = Path(args.txt)
    if not txt_path.is_file():
        raise FileNotFoundError(f"txt not found: {txt_path}")

    frames_filter: Optional[Set[str]] = None
    if args.frames_filter:
        frames_filter = {p.name for p in args.frames_filter.iterdir() if p.is_dir()}

    bounds_map, declared_map = parse_blocks(txt_path)

    entries = []
    for video_stem, bounds in bounds_map.items():
        if frames_filter is not None and video_stem not in frames_filter:
            continue
        segments, dropped = boundaries_to_segments(bounds, declared_map.get(video_stem))
        if dropped:
            print(f"[warn] {video_stem}: dropped {dropped} out-of-order bounds")
        labels, many_hot = segments_to_labels(segments)
        frames_dir = str(args.frames_root / video_stem) if args.frames_root else video_stem
        entries.append({"frames_dir": frames_dir, "labels": labels, "many_hot_labels": many_hot})

    with open(args.output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
