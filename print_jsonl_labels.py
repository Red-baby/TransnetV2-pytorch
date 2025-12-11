import argparse
import json
from pathlib import Path


def positions_of_ones(arr):
    return [idx for idx, val in enumerate(arr) if val == 1]


def frames_id_matches(frames_dir: str, target: str) -> bool:
    # 允许匹配完整路径、末尾文件夹名或简单子串
    frames_dir_path = Path(frames_dir)
    return (
        frames_dir == target
        or frames_dir_path.name == target
        or frames_dir.replace("\\", "/").endswith(target)
    )


def main():
    parser = argparse.ArgumentParser(
        description="按 frames_dir 中的 ID 查找 JSONL 行，打印 labels/many_hot_labels 中为 1 的位置（0-based）。"
    )
    parser.add_argument("--jsonl", required=True, help="JSONL 文件路径")
    parser.add_argument("--id", required=True, help="要匹配的 ID，如 33338203782（匹配 frames_dir 最末级目录或结尾部分）")
    parser.add_argument("--labels-key", default="labels", help="labels 字段名称，默认为 labels")
    parser.add_argument("--manyhot-key", default="many_hot_labels", help="many_hot_labels 字段名称，默认为 many_hot_labels")
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"未找到文件: {path}")

    found = False
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            frames_dir = entry.get("frames_dir", "")
            if not frames_id_matches(frames_dir, args.id):
                continue

            found = True
            labels = entry.get(args.labels_key, [])
            many_hot = entry.get(args.manyhot_key, [])

            labels_pos = positions_of_ones(labels)
            many_hot_pos = positions_of_ones(many_hot)

            print(f"文件: {path}")
            print(f"匹配 ID: {args.id}")
            print(f"行号: {line_no}")
            print(f"frames_dir: {frames_dir}")
            print(f"{args.labels_key} 为 1 的位置: {labels_pos}")
            print(f"{args.manyhot_key} 为 1 的位置: {many_hot_pos}")
            print("-" * 40)

    if not found:
        print(f"未找到包含 ID '{args.id}' 的记录。")


if __name__ == "__main__":
    main()
