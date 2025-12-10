import argparse
import json
import pickle
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


def main():
    parser = argparse.ArgumentParser(description="Convert gt_scenes_dict_baseline_v2.pickle to JSONL manifest.")
    parser.add_argument("--pickle", required=True, help="Path to gt_scenes_dict_baseline_v2.pickle")
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

    frames_filter: Optional[Set[str]] = None
    if args.frames_filter:
        frames_filter = {p.name for p in args.frames_filter.iterdir() if p.is_dir()}

    with open(args.pickle, "rb") as f:
        data: Dict[str, List[Tuple[int, int]]] = pickle.load(f)

    entries = []
    for video_stem, seg_array in data.items():
        segments = sorted([(int(s), int(e)) for s, e in seg_array], key=lambda x: x[0])
        if frames_filter is not None and video_stem not in frames_filter:
            continue
        labels, many_hot = segments_to_labels(segments)
        frames_dir = str(args.frames_root / video_stem) if args.frames_root else video_stem
        entries.append({"frames_dir": frames_dir, "labels": labels, "many_hot_labels": many_hot})

    with open(args.output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
