import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

def parse_transition(token: str) -> Tuple[int, int]:
    nums = re.findall(r"\d+", token)
    if len(nums) < 2:
        raise ValueError(f"Cannot parse transition token: {token}")
    return int(nums[0]), int(nums[1])

def build_labels(frame_count_hint: int, cuts: List[int], dissolves: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    max_idx = -1
    if cuts:
        max_idx = max(max_idx, max(cuts))
    if dissolves:
        max_idx = max(max_idx, max(b for _, b in dissolves))
    length = max(frame_count_hint, max_idx + 1)
    labels = [0] * length
    many_hot = [0] * length
    for anchor in cuts:
        if anchor < length:
            labels[anchor] = 1
            many_hot[anchor] = 1
    for start, end in dissolves:
        if start < length:
            labels[start] = 1  # 锚点取过渡区间起点
            end_clamped = min(end, length - 1)
            for i in range(start, end_clamped + 1):
                many_hot[i] = 1
    return labels, many_hot

def main():
    parser = argparse.ArgumentParser(description="Convert kuaishou_v2.txt style annotations to JSONL manifest.")
    parser.add_argument("--input", required=True, help="Path to kuaishou_v2.txt")
    parser.add_argument("--output", required=True, help="Path to write JSONL manifest")
    parser.add_argument("--frames-root", type=Path, default=None,
                        help="Root directory that contains per-video frame folders; frames_dir will be frames_root/<stem> if set.")
    parser.add_argument("--frames-filter", type=Path, default=None,
                        help="If set, only keep entries whose stem matches a subfolder under this path.")
    args = parser.parse_args()

    frames_filter: Optional[Set[str]] = None
    if args.frames_filter:
        frames_filter = {p.name for p in args.frames_filter.iterdir() if p.is_dir()}

    entries = []
    current_name: Optional[str] = None
    current_count = 0
    current_transitions: List[str] = []

    def flush_current():
        nonlocal current_name, current_count, current_transitions
        if current_name is None:
            return
        cuts: List[int] = []
        dissolves: List[Tuple[int, int]] = []
        for tok in current_transitions:
            if "," not in tok:
                continue
            try:
                a, b = parse_transition(tok)
            except ValueError:
                continue
            if b == a + 1:
                cuts.append(a)
            elif b > a:
                dissolves.append((a, b))
        labels, many_hot = build_labels(current_count, cuts, dissolves)
        stem = Path(current_name).stem
        frames_dir = str(args.frames_root / stem) if args.frames_root else stem
        if frames_filter is not None and stem not in frames_filter:
            current_name = None
            current_count = 0
            current_transitions = []
            return
        entries.append({"frames_dir": frames_dir, "labels": labels, "many_hot_labels": many_hot})
        current_name = None
        current_count = 0
        current_transitions = []

    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0].endswith(".mp4"):
                flush_current()
                current_name = parts[0]
                current_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                current_transitions = parts[2:]
            else:
                current_transitions.extend(parts)
        flush_current()

    with Path(args.output).open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} entries to {args.output}")

if __name__ == "__main__":
    main()
