import argparse
import json
from typing import List, Tuple


def extract_transitions(labels: List[int], many_hot: List[int]) -> List[Tuple[int, int]]:
    """
    Convert frame-level labels into pairs of (prev_frame_idx, next_frame_idx).
    - Hard cut: many_hot run length == 1 (or many_hot missing => labels used), emit (idx, idx + 1).
    - Dissolve: many_hot run length > 1, emit (start, end) as过渡区间的起止帧（均为 0-based）。
    """
    mh = many_hot if many_hot is not None else labels
    n = len(mh)
    transitions = []
    i = 0
    while i < n:
        if mh[i] == 0:
            i += 1
            continue
        start = i
        while i + 1 < n and mh[i + 1] == 1:
            i += 1
        end = i  # inclusive
        if end == start:  # hard cut
            prev_idx = start
            next_idx = start + 1
            if next_idx < n:
                transitions.append((prev_idx, next_idx))
        else:  # dissolve
            transitions.append((start, end))
        i += 1
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Extract indices of transitions from a JSON file.")
    parser.add_argument(
        "--json-path",
        type=str,
        required=True,
        help="Path to a JSON file containing `labels` and optional `many_hot_labels` arrays.",
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        entry = json.load(f)
    labels = entry["labels"]
    many_hot = entry.get("many_hot_labels")

    pairs = extract_transitions(labels, many_hot)
    for a, b in pairs:
        print(f"{a},{b}")


if __name__ == "__main__":
    main()
