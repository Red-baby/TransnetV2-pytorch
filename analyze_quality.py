# -*- coding: utf-8 -*-
"""
编码质量对比分析脚本
解析 q265 编码日志，对比 transnetv2 和 icut 两种场景检测方法的编码效果
兼容 Python 2.7
"""

from __future__ import print_function
import re
import os
import sys
import json
from collections import defaultdict


def parse_q265_log(log_file):
    """
    解析 q265 日志文件
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        list of dict: 包含每帧信息的列表
        格式: [{'poc': 7, 'type': 'b', 'qp': 37.0, 'bits': 41, 'psnr_y': 42.9368}, ...]
    """
    frames = []
    
    # 正则表达式匹配 q265 [info]: 行
    # 格式: q265 [info]:    7  b  37.00       41   42.9368 46.4839 47.1132 ...
    pattern = re.compile(
        r'q265 \[info\]:\s+'
        r'(\d+)\s+'           # POC
        r'([bBPI])\s+'        # 帧类型
        r'([\d.]+)\s+'        # QP
        r'(\d+)\s+'           # 比特数
        r'([\d.]+)\s+'        # PSNR_Y
        r'([\d.]+)\s+'        # PSNR_U
        r'([\d.]+)'           # PSNR_V
    )
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    frames.append({
                        'poc': int(match.group(1)),
                        'type': match.group(2),
                        'qp': float(match.group(3)),
                        'bits': int(match.group(4)),
                        'psnr_y': float(match.group(5)),
                        'psnr_u': float(match.group(6)),
                        'psnr_v': float(match.group(7))
                    })
    except IOError as e:
        print("ERROR 无法读取日志文件: {}".format(log_file))
        print("ERROR {}".format(e))
        return []
    
    return frames


def calculate_statistics(frames):
    """
    计算统计信息
    
    Args:
        frames: 帧信息列表
    
    Returns:
        dict: 统计信息
    """
    if not frames:
        return None
    
    # 按 POC 排序（播放顺序）
    frames_sorted = sorted(frames, key=lambda x: x['poc'])
    
    psnr_y_values = [f['psnr_y'] for f in frames_sorted]
    qp_values = [f['qp'] for f in frames_sorted]
    bits_values = [f['bits'] for f in frames_sorted]
    
    # 计算相邻帧波动（POC 顺序）
    psnr_y_fluctuations = []
    for i in range(len(frames_sorted) - 1):
        fluctuation = abs(frames_sorted[i+1]['psnr_y'] - frames_sorted[i]['psnr_y'])
        psnr_y_fluctuations.append(fluctuation)
    
    # 按帧类型分组
    frames_by_type = defaultdict(list)
    for f in frames_sorted:
        frames_by_type[f['type']].append(f['psnr_y'])
    
    # 计算方差（标准差的平方）
    def variance(values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    # 计算标准差
    def std_dev(values):
        return variance(values) ** 0.5
    
    stats = {
        'total_frames': len(frames_sorted),
        'total_bits': sum(bits_values),
        'psnr_y': {
            'min': min(psnr_y_values),
            'max': max(psnr_y_values),
            'mean': sum(psnr_y_values) / len(psnr_y_values),
            'std': std_dev(psnr_y_values),
            'variance': variance(psnr_y_values)
        },
        'psnr_y_fluctuation': {
            'mean': sum(psnr_y_fluctuations) / len(psnr_y_fluctuations) if psnr_y_fluctuations else 0.0,
            'max': max(psnr_y_fluctuations) if psnr_y_fluctuations else 0.0,
            'std': std_dev(psnr_y_fluctuations) if psnr_y_fluctuations else 0.0,
            'total_fluctuations': len(psnr_y_fluctuations)
        },
        'qp': {
            'min': min(qp_values),
            'max': max(qp_values),
            'mean': sum(qp_values) / len(qp_values),
            'std': std_dev(qp_values)
        },
        'bits': {
            'min': min(bits_values),
            'max': max(bits_values),
            'mean': sum(bits_values) / len(bits_values),
            'total': sum(bits_values)
        },
        'frame_types': {}
    }
    
    # 每种帧类型的统计
    for ftype, psnr_values in frames_by_type.items():
        stats['frame_types'][ftype] = {
            'count': len(psnr_values),
            'psnr_y_mean': sum(psnr_values) / len(psnr_values),
            'psnr_y_std': std_dev(psnr_values)
        }
    
    return stats


def compare_videos(video_name, transnetv2_log, icut_log):
    """
    对比一个视频的两种方法
    
    Args:
        video_name: 视频名称
        transnetv2_log: transnetv2 日志路径
        icut_log: icut 日志路径
    
    Returns:
        dict: 对比结果
    """
    print("\n" + "=" * 70)
    print("分析视频: {}".format(video_name))
    print("=" * 70)
    
    # 解析日志
    print("解析 TransNetV2 日志...")
    transnetv2_frames = parse_q265_log(transnetv2_log)
    print("  提取 {} 帧".format(len(transnetv2_frames)))
    
    print("解析 icut 日志...")
    icut_frames = parse_q265_log(icut_log)
    print("  提取 {} 帧".format(len(icut_frames)))
    
    if not transnetv2_frames or not icut_frames:
        print("ERROR 日志解析失败，跳过此视频")
        return None
    
    # 计算统计
    print("计算统计信息...")
    transnetv2_stats = calculate_statistics(transnetv2_frames)
    icut_stats = calculate_statistics(icut_frames)
    
    # 对比结果
    comparison = {
        'video_name': video_name,
        'transnetv2': {
            'log_file': transnetv2_log,
            'stats': transnetv2_stats
        },
        'icut': {
            'log_file': icut_log,
            'stats': icut_stats
        },
        'comparison': {
            'psnr_y_diff_mean': transnetv2_stats['psnr_y']['mean'] - icut_stats['psnr_y']['mean'],
            'psnr_y_diff_std': transnetv2_stats['psnr_y']['std'] - icut_stats['psnr_y']['std'],
            'psnr_y_fluctuation_diff_mean': (transnetv2_stats['psnr_y_fluctuation']['mean'] - 
                                             icut_stats['psnr_y_fluctuation']['mean']),
            'bits_diff_total': transnetv2_stats['bits']['total'] - icut_stats['bits']['total'],
            'bits_diff_percent': ((transnetv2_stats['bits']['total'] - icut_stats['bits']['total']) / 
                                   float(icut_stats['bits']['total']) * 100)
        }
    }
    
    # 打印对比
    print("\n质量对比 (PSNR_Y):")
    print("  TransNetV2: 平均={:.4f}, 标准差={:.4f}, 方差={:.4f}".format(
        transnetv2_stats['psnr_y']['mean'],
        transnetv2_stats['psnr_y']['std'],
        transnetv2_stats['psnr_y']['variance']
    ))
    print("  icut:       平均={:.4f}, 标准差={:.4f}, 方差={:.4f}".format(
        icut_stats['psnr_y']['mean'],
        icut_stats['psnr_y']['std'],
        icut_stats['psnr_y']['variance']
    ))
    print("  差异:       平均={:+.4f}, 标准差={:+.4f}".format(
        comparison['comparison']['psnr_y_diff_mean'],
        comparison['comparison']['psnr_y_diff_std']
    ))
    
    print("\n相邻帧波动 (POC 顺序，越小越平滑):")
    print("  TransNetV2: 平均波动={:.4f}, 最大波动={:.4f}".format(
        transnetv2_stats['psnr_y_fluctuation']['mean'],
        transnetv2_stats['psnr_y_fluctuation']['max']
    ))
    print("  icut:       平均波动={:.4f}, 最大波动={:.4f}".format(
        icut_stats['psnr_y_fluctuation']['mean'],
        icut_stats['psnr_y_fluctuation']['max']
    ))
    print("  差异:       平均波动={:+.4f}".format(
        comparison['comparison']['psnr_y_fluctuation_diff_mean']
    ))
    
    print("\n比特率对比:")
    print("  TransNetV2: 总计={} bits".format(transnetv2_stats['bits']['total']))
    print("  icut:       总计={} bits".format(icut_stats['bits']['total']))
    print("  差异:       {:+} bits ({:+.2f}%)".format(
        comparison['comparison']['bits_diff_total'],
        comparison['comparison']['bits_diff_percent']
    ))
    
    # 质量平滑度对比（相邻帧波动越小越平滑）
    print("\n播放质量平滑度 (基于相邻帧波动):")
    if transnetv2_stats['psnr_y_fluctuation']['mean'] < icut_stats['psnr_y_fluctuation']['mean']:
        print("  TransNetV2 更平滑 (平均波动小 {:.4f})".format(
            icut_stats['psnr_y_fluctuation']['mean'] - transnetv2_stats['psnr_y_fluctuation']['mean']
        ))
    else:
        print("  icut 更平滑 (平均波动小 {:.4f})".format(
            transnetv2_stats['psnr_y_fluctuation']['mean'] - icut_stats['psnr_y_fluctuation']['mean']
        ))
    
    return comparison


def generate_summary_report(all_comparisons, output_file):
    """
    生成汇总报告
    
    Args:
        all_comparisons: 所有视频的对比结果列表
        output_file: 输出文件路径
    """
    if not all_comparisons:
        print("\nERROR 没有有效的对比结果")
        return
    
    print("\n" + "=" * 70)
    print("生成汇总报告")
    print("=" * 70)
    
    # 计算总体统计
    transnetv2_wins_quality = 0
    icut_wins_quality = 0
    transnetv2_wins_smoothness = 0
    icut_wins_smoothness = 0
    transnetv2_wins_bits = 0
    icut_wins_bits = 0
    
    for comp in all_comparisons:
        # 质量对比（PSNR_Y 平均值）
        if comp['comparison']['psnr_y_diff_mean'] > 0:
            transnetv2_wins_quality += 1
        else:
            icut_wins_quality += 1
        
        # 平滑度对比（相邻帧平均波动，越小越好）
        if comp['comparison']['psnr_y_fluctuation_diff_mean'] < 0:
            transnetv2_wins_smoothness += 1
        else:
            icut_wins_smoothness += 1
        
        # 比特率对比（越小越好）
        if comp['comparison']['bits_diff_total'] < 0:
            transnetv2_wins_bits += 1
        else:
            icut_wins_bits += 1
    
    summary = {
        'total_videos': len(all_comparisons),
        'quality_comparison': {
            'transnetv2_wins': transnetv2_wins_quality,
            'icut_wins': icut_wins_quality
        },
        'smoothness_comparison': {
            'transnetv2_wins': transnetv2_wins_smoothness,
            'icut_wins': icut_wins_smoothness
        },
        'bitrate_comparison': {
            'transnetv2_wins': transnetv2_wins_bits,
            'icut_wins': icut_wins_bits
        },
        'details': all_comparisons
    }
    
    # 保存 JSON 报告
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("报告已保存: {}".format(output_file))
    
    # 打印汇总
    print("\n总体对比 (共 {} 个视频):".format(len(all_comparisons)))
    print("\n1. 平均质量 (PSNR_Y):")
    print("   TransNetV2 胜: {} 个".format(transnetv2_wins_quality))
    print("   icut 胜:       {} 个".format(icut_wins_quality))
    
    print("\n2. 播放质量平滑度 (相邻帧波动，越小越好):")
    print("   TransNetV2 胜: {} 个".format(transnetv2_wins_smoothness))
    print("   icut 胜:       {} 个".format(icut_wins_smoothness))
    
    print("\n3. 比特率 (越低越好):")
    print("   TransNetV2 胜: {} 个".format(transnetv2_wins_bits))
    print("   icut 胜:       {} 个".format(icut_wins_bits))


def main():
    """主函数"""
    print("=" * 70)
    print("编码质量对比分析")
    print("=" * 70)
    
    # 配置
    logs_dir = "./logs"
    output_report = "./quality_comparison_report.json"
    
    # 检查日志目录
    if not os.path.exists(logs_dir):
        print("ERROR 日志目录不存在: {}".format(logs_dir))
        sys.exit(1)
    
    # 收集所有对比结果
    all_comparisons = []
    
    # 处理 bbc_01 到 bbc_11
    for video_num in range(1, 12):
        video_name = "bbc_{:02d}".format(video_num)
        
        transnetv2_log = os.path.join(logs_dir, "{}_transnetv2.log".format(video_name))
        icut_log = os.path.join(logs_dir, "{}_icut.log".format(video_name))
        
        # 检查日志文件是否存在
        if not os.path.exists(transnetv2_log):
            print("\nWARN TransNetV2 日志不存在: {}".format(transnetv2_log))
            continue
        if not os.path.exists(icut_log):
            print("\nWARN icut 日志不存在: {}".format(icut_log))
            continue
        
        # 对比
        comparison = compare_videos(video_name, transnetv2_log, icut_log)
        if comparison:
            all_comparisons.append(comparison)
    
    # 生成汇总报告
    if all_comparisons:
        generate_summary_report(all_comparisons, output_report)
        print("\n" + "=" * 70)
        print("分析完成!")
        print("=" * 70)
    else:
        print("\nERROR 没有找到有效的日志文件对")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nERROR 用户中断")
        sys.exit(130)
    except Exception as e:
        print("\nERROR {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
