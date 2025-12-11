import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="将 JSONL 中的 frames_dir 更新为指定根目录下的绝对路径。"
    )
    parser.add_argument("--jsonl", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output-jsonl", required=True, help="输出 JSONL 路径")
    parser.add_argument(
        "--root",
        default="/lhdata/lh_data/Autoshot/frames",
        help="目标根目录，默认为 /lhdata/lh_data/Autoshot/frames",
    )
    parser.add_argument(
        "--keep-relative",
        action="store_true",
        help="默认仅保留最后一级目录名；若开启，则将原 frames_dir 的相对路径附在 root 之后。",
    )
    args = parser.parse_args()

    src = Path(args.jsonl)
    dst = Path(args.output_jsonl)
    root = Path(args.root)
    if not src.is_file():
        raise FileNotFoundError(f"未找到 JSONL：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    updated = []
    for line in lines:
        entry = json.loads(line)
        orig = entry.get("frames_dir", "")
        if args.keep_relative:
            rel = Path(orig).as_posix().lstrip("/\\")
            new_dir = root / rel
        else:
            new_dir = root / Path(orig).name
        entry["frames_dir"] = str(new_dir)
        updated.append(json.dumps(entry, ensure_ascii=False))

    dst.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"已写入更新后的 JSONL：{dst}")


if __name__ == "__main__":
    main()
