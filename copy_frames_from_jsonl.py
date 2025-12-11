import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="将 JSONL 中的 frames_dir 对应文件夹拷贝到指定目录，并更新 JSONL 中的 frames_dir。"
    )
    parser.add_argument("--jsonl", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output-jsonl", required=True, help="更新 frames_dir 后的输出 JSONL 路径")
    parser.add_argument(
        "--dst-root",
        required=True,
        help="目标根目录，frames 目录将拷贝到此目录下（子目录名取原 frames_dir 的末级文件夹名）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="目标子目录已存在时先删除再拷贝（默认遇到已存在则跳过拷贝但仍更新 JSONL）",
    )
    args = parser.parse_args()

    src_jsonl = Path(args.jsonl)
    dst_jsonl = Path(args.output_jsonl)
    dst_root = Path(args.dst_root)

    if not src_jsonl.is_file():
        raise FileNotFoundError(f"未找到 JSONL：{src_jsonl}")
    dst_root.mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in src_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    updated = []
    copied, skipped_missing, skipped_exists = 0, 0, 0

    for line in lines:
        entry = json.loads(line)
        frames_dir = Path(entry["frames_dir"])
        src_dir = frames_dir
        target_dir = dst_root / frames_dir.name

        if not src_dir.is_dir():
            print(f"[missing] 源目录不存在，跳过拷贝：{src_dir}")
            skipped_missing += 1
        else:
            if target_dir.exists():
                if args.overwrite:
                    shutil.rmtree(target_dir)
                else:
                    print(f"[exists] 目标已存在，跳过拷贝：{target_dir}")
                    skipped_exists += 1
                # 如果存在且未覆盖，仍然更新 JSONL 指向目标路径
            if not target_dir.exists():
                shutil.copytree(src_dir, target_dir)
                copied += 1
                print(f"[ok] {src_dir} -> {target_dir}")

        # 更新 frames_dir 指向目标路径
        entry["frames_dir"] = str(target_dir)
        updated.append(json.dumps(entry, ensure_ascii=False))

    dst_jsonl.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(
        f"完成：拷贝 {copied}，缺失 {skipped_missing}，已存在 {skipped_exists}（未覆盖时跳过拷贝）。"
        f"\n输出 JSONL：{dst_jsonl}"
    )


if __name__ == "__main__":
    main()
