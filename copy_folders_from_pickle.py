import argparse
import pickle
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="根据 pickle 文件中的键名，将对应目录从源路径拷贝到目标路径。"
    )
    parser.add_argument("--pickle", required=True, help="pickle 文件路径（应为字典，键为文件夹名）")
    parser.add_argument(
        "--src-root",
        default="/lhdata/lh_data/Autoshot/frames",
        help="源根目录，默认 /lhdata/lh_data/Autoshot/frames",
    )
    parser.add_argument(
        "--dst-root",
        default="Autoshot/frames",
        help="目标根目录，默认 ./Autoshot/frames（相对当前工作目录）",
    )
    parser.add_argument("--overwrite", action="store_true", help="目标已存在时先删除再拷贝")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    keys = list(data.keys())
    copied, missing = 0, 0
    for key in keys:
        key_str = str(key)
        src_dir = src_root / key_str
        dst_dir = dst_root / key_str
        if not src_dir.is_dir():
            print(f"[skip] 源目录不存在: {src_dir}")
            missing += 1
            continue
        if dst_dir.exists():
            if args.overwrite:
                shutil.rmtree(dst_dir)
            else:
                print(f"[skip] 目标已存在: {dst_dir}（使用 --overwrite 可覆盖）")
                continue
        shutil.copytree(src_dir, dst_dir)
        print(f"[ok] {src_dir} -> {dst_dir}")
        copied += 1

    print(f"完成：拷贝 {copied} 个，缺失 {missing} 个。")


if __name__ == "__main__":
    main()
