#!/usr/bin/env python3
"""
批量视频编码脚本
对 bbc_01 到 bbc_11 使用 transnetv2 和 icut 的 keyframe 列表分别编码
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def create_directories():
    """创建必要的目录"""
    Path("./tmp").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    print("✓ 目录已创建/确认存在: ./tmp, ./logs\n")


def run_encode_command(
    video_num: int,
    method: str,
    input_video: str,
    keyframe_file: str,
    output_file: str,
    log_file: str
) -> bool:
    """
    执行单个编码命令
    
    Args:
        video_num: 视频编号 (1-11)
        method: 方法名称 (transnetv2/icut)
        input_video: 输入视频路径
        keyframe_file: keyframe 列表文件路径
        output_file: 输出文件路径
        log_file: 日志文件路径
    
    Returns:
        True if success, False if failed
    """
    # 构建命令
    cmd = [
        "./vtcoder_new",
        "-i", input_video,
        "-w", "360",
        "-b", "300",
        "-r", "25",
        "--maxWidth", "1280",
        "--maxHeight", "720",
        "--no-dvdnav",
        "--cloud_mode",
        "--subtitle-enhance",
        "--large-file",
        "-v",
        "-O",
        "--modulus", "8:8",
        "--cpu-num", "8",
        "--crop", "0:0:0:0",
        "--outside-iframe-read", keyframe_file,
        "--disable-adaptive-decomb",
        "-e", "q265",
        "-2",
        "-T",
        "--scene-mode", "2",
        "--rotate",
        "-x", "bframes=16:keyint=120:qp-min=10:qp-max=44:frame-threads=8:threads=32:wpp:scenecut=40:min-keyint=24:refresh=2:rd-level=2:pass-type=2:b-adapt=1:mctf=1",
        "-o", output_file
    ]
    
    # 显示当前任务
    print("=" * 70)
    print(f"视频: bbc_{video_num:02d}")
    print(f"方法: {method}")
    print(f"Keyframe: {keyframe_file}")
    print(f"输出: {output_file}")
    print(f"日志: {log_file}")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 执行命令并保存日志
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            # 写入命令信息到日志
            log.write(f"命令执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"视频编号: bbc_{video_num:02d}\n")
            log.write(f"检测方法: {method}\n")
            log.write(f"命令: {' '.join(cmd)}\n")
            log.write("=" * 70 + "\n\n")
            log.flush()
            
            # 执行命令，实时输出到终端和日志
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出
            for line in process.stdout:
                print(line, end='')  # 输出到终端
                log.write(line)      # 写入日志
                log.flush()
            
            # 等待进程结束
            return_code = process.wait()
            
            # 写入结束信息
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log.write(f"\n{'=' * 70}\n")
            log.write(f"结束时间: {end_time}\n")
            log.write(f"返回码: {return_code}\n")
            
            if return_code == 0:
                print(f"\n✓ 完成时间: {end_time}")
                print(f"✓ 成功完成: bbc_{video_num:02d} ({method})\n")
                return True
            else:
                print(f"\n✗ 失败时间: {end_time}")
                print(f"✗ 编码失败: bbc_{video_num:02d} ({method})")
                print(f"✗ 返回码: {return_code}")
                print(f"✗ 详细日志: {log_file}\n")
                return False
                
    except FileNotFoundError:
        error_msg = f"错误: 找不到 vtcoder_new 可执行文件"
        print(f"\n✗ {error_msg}\n")
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n{error_msg}\n")
        return False
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        print(f"\n✗ {error_msg}\n")
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n{error_msg}\n")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("BBC 视频批量编码脚本")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 创建必要的目录
    create_directories()
    
    # 配置
    video_nums = range(1, 12)  # 1 到 11
    methods = ["transnetv2", "icut"]
    base_video_path = "/mnt/ec-data2/lh/bbc"
    
    # 统计
    total_tasks = len(video_nums) * len(methods)
    completed_tasks = 0
    
    print(f"总任务数: {total_tasks} (11 个视频 × 2 种方法)\n")
    
    # 循环处理每个视频
    for video_num in video_nums:
        video_name = f"bbc_{video_num:02d}"
        
        # 对每个视频使用两种方法
        for method in methods:
            completed_tasks += 1
            
            print(f"\n[进度: {completed_tasks}/{total_tasks}]")
            
            # 构建路径
            input_video = f"{base_video_path}/{video_name}.mp4"
            keyframe_file = f"{base_video_path}/{video_name}_keyframe_PTS_{method}.txt"
            output_file = f"./tmp/{video_name}_{method}.265"
            log_file = f"./logs/{video_name}_{method}.log"
            
            # 检查 keyframe 文件是否存在
            if not Path(keyframe_file).exists():
                print(f"✗ 错误: Keyframe 文件不存在: {keyframe_file}")
                print(f"✗ 脚本终止于任务 {completed_tasks}/{total_tasks}\n")
                sys.exit(1)
            
            # 执行编码
            success = run_encode_command(
                video_num,
                method,
                input_video,
                keyframe_file,
                output_file,
                log_file
            )
            
            # 如果失败，停止整个脚本
            if not success:
                print("=" * 70)
                print("✗ 编码失败，脚本已终止")
                print(f"✗ 失败任务: {video_name} ({method})")
                print(f"✗ 已完成: {completed_tasks - 1}/{total_tasks}")
                print(f"✗ 详细日志: {log_file}")
                print("=" * 70)
                sys.exit(1)
    
    # 全部完成
    print("\n" + "=" * 70)
    print("✓ 所有任务已完成!")
    print(f"✓ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ 总计: {completed_tasks}/{total_tasks} 个任务")
    print(f"✓ 输出目录: ./tmp/")
    print(f"✓ 日志目录: ./logs/")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ 用户中断 (Ctrl+C)")
        print("✗ 脚本已终止")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ 未预料的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
