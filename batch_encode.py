# -*- coding: utf-8 -*-
"""
批量视频编码脚本
对 bbc_01 到 bbc_11 使用 transnetv2 和 icut 的 keyframe 列表分别编码
兼容 Python 2.7
"""

from __future__ import print_function
import subprocess
import sys
import os
from datetime import datetime


def create_directories():
    """创建必要的目录"""
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    print("OK 目录已创建/确认存在: ./tmp, ./logs\n")


def run_encode_command(video_num, method, input_video, keyframe_file, output_file, log_file):
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
    print("视频: bbc_{:02d}".format(video_num))
    print("方法: {}".format(method))
    print("Keyframe: {}".format(keyframe_file))
    print("输出: {}".format(output_file))
    print("日志: {}".format(log_file))
    print("=" * 70)
    print("开始时间: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print()
    
    # 执行命令并保存日志
    try:
        with open(log_file, "w") as log:
            # 写入命令信息到日志
            log.write("命令执行时间: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            log.write("视频编号: bbc_{:02d}\n".format(video_num))
            log.write("检测方法: {}\n".format(method))
            log.write("命令: {}\n".format(' '.join(cmd)))
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
            for line in iter(process.stdout.readline, ''):
                if line == '':
                    break
                print(line, end='')  # 输出到终端
                log.write(line)      # 写入日志
                log.flush()
            
            # 等待进程结束
            return_code = process.wait()
            
            # 写入结束信息
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log.write("\n" + "=" * 70 + "\n")
            log.write("结束时间: {}\n".format(end_time))
            log.write("返回码: {}\n".format(return_code))
            
            if return_code == 0:
                print("\nOK 完成时间: {}".format(end_time))
                print("OK 成功完成: bbc_{:02d} ({})\n".format(video_num, method))
                return True
            else:
                print("\nERROR 失败时间: {}".format(end_time))
                print("ERROR 编码失败: bbc_{:02d} ({})".format(video_num, method))
                print("ERROR 返回码: {}".format(return_code))
                print("ERROR 详细日志: {}\n".format(log_file))
                return False
                
    except OSError as e:
        if e.errno == 2:  # File not found
            error_msg = "错误: 找不到 vtcoder_new 可执行文件"
        else:
            error_msg = "错误: {}".format(str(e))
        print("\nERROR {}\n".format(error_msg))
        with open(log_file, "a") as log:
            log.write("\n{}\n".format(error_msg))
        return False
    except Exception as e:
        error_msg = "错误: {}".format(str(e))
        print("\nERROR {}\n".format(error_msg))
        with open(log_file, "a") as log:
            log.write("\n{}\n".format(error_msg))
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("BBC 视频批量编码脚本")
    print("=" * 70)
    print("开始时间: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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
    
    print("总任务数: {} (11 个视频 x 2 种方法)\n".format(total_tasks))
    
    # 循环处理每个视频
    for video_num in video_nums:
        video_name = "bbc_{:02d}".format(video_num)
        
        # 对每个视频使用两种方法
        for method in methods:
            completed_tasks += 1
            
            print("\n[进度: {}/{}]".format(completed_tasks, total_tasks))
            
            # 构建路径
            input_video = "{}/{}.mp4".format(base_video_path, video_name)
            keyframe_file = "{}/{}_keyframe_PTS_{}.txt".format(base_video_path, video_name, method)
            output_file = "./tmp/{}_{}.265".format(video_name, method)
            log_file = "./logs/{}_{}.log".format(video_name, method)
            
            # 检查 keyframe 文件是否存在
            if not os.path.exists(keyframe_file):
                print("ERROR Keyframe 文件不存在: {}".format(keyframe_file))
                print("ERROR 脚本终止于任务 {}/{}\n".format(completed_tasks, total_tasks))
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
                print("ERROR 编码失败，脚本已终止")
                print("ERROR 失败任务: {} ({})".format(video_name, method))
                print("ERROR 已完成: {}/{}".format(completed_tasks - 1, total_tasks))
                print("ERROR 详细日志: {}".format(log_file))
                print("=" * 70)
                sys.exit(1)
    
    # 全部完成
    print("\n" + "=" * 70)
    print("OK 所有任务已完成!")
    print("OK 完成时间: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("OK 总计: {}/{} 个任务".format(completed_tasks, total_tasks))
    print("OK 输出目录: ./tmp/")
    print("OK 日志目录: ./logs/")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nERROR 用户中断 (Ctrl+C)")
        print("ERROR 脚本已终止")
        sys.exit(130)
    except Exception as e:
        print("\nERROR 未预料的错误: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
