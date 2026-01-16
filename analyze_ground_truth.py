# -*- coding: utf-8 -*-
"""
基于真实转场标签的质量分析
对比 TransNetV2 和 icut 在真实转场后的编码质量
兼容 Python 2.7
"""

from __future__ import print_function
import re
import os
import sys
import json


def read_ground_truth(gt_file):
    """
    读取真实转场标签文件
    
    格式：
    0    632
    650  770
    771  891
    ...
    
    Returns:
        list of int: 转场位置（每个镜头的起始帧）
    """
    scene_starts = []
    
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    start = int(parts[0])
                    # 第一个镜头从0开始，不算转场
                    if start > 0:
                        scene_starts.append(start)
        
        return sorted(scene_starts)
        
    except IOError as e:
        print("ERROR 无法读取真实标签文件: {}".format(gt_file))
        print("ERROR {}".format(e))
        return []


def parse_q265_log(log_file):
    """
    解析 q265 日志文件，返回按 POC 索引的帧信息
    
    Returns:
        dict: {poc: {'psnr_y': 42.9368, ...}, ...}
    """
    frames = {}
    
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
                    poc = int(match.group(1))
                    frames[poc] = {
                        'poc': poc,
                        'type': match.group(2),
                        'qp': float(match.group(3)),
                        'bits': int(match.group(4)),
                        'psnr_y': float(match.group(5))
                    }
    except IOError as e:
        print("ERROR 无法读取日志文件: {}".format(log_file))
        return {}
    
    return frames


def calculate_scene_quality(frames_dict, scene_start, window=20):
    """
    计算转场后指定帧数的平均 PSNR
    
    Args:
        frames_dict: 帧信息字典
        scene_start: 转场起始 POC
        window: 窗口大小（默认20帧）
    
    Returns:
        dict: 统计信息
    """
    psnr_values = []
    
    # 提取转场后的帧 [scene_start, scene_start + window - 1]
    for poc in range(scene_start, scene_start + window):
        if poc in frames_dict:
            psnr_values.append(frames_dict[poc]['psnr_y'])
    
    if not psnr_values:
        return None
    
    return {
        'count': len(psnr_values),
        'psnr_mean': sum(psnr_values) / len(psnr_values),
        'psnr_min': min(psnr_values),
        'psnr_max': max(psnr_values)
    }


def analyze_video(video_name, gt_dir, logs_dir):
    """
    分析一个视频的真实转场质量
    """
    print("\n" + "=" * 70)
    print("分析视频: {}".format(video_name))
    print("=" * 70)
    
    # 读取真实标签
    gt_file = os.path.join(gt_dir, "{}.txt".format(video_name))
    print("读取真实转场标签: {}".format(gt_file))
    
    true_scenes = read_ground_truth(gt_file)
    if not true_scenes:
        print("ERROR 真实标签读取失败")
        return None
    
    print("  真实转场数: {}".format(len(true_scenes)))
    print("  转场位置: {}".format(true_scenes[:10] + (['...'] if len(true_scenes) > 10 else [])))
    
    # 读取编码日志
    tn_log = os.path.join(logs_dir, "{}_transnetv2.log".format(video_name))
    ic_log = os.path.join(logs_dir, "{}_icut.log".format(video_name))
    
    print("\n解析编码日志...")
    tn_frames = parse_q265_log(tn_log)
    ic_frames = parse_q265_log(ic_log)
    
    if not tn_frames or not ic_frames:
        print("ERROR 日志解析失败")
        return None
    
    print("  TransNetV2: {} 帧".format(len(tn_frames)))
    print("  icut:       {} 帧".format(len(ic_frames)))
    
    # 分析每个真实转场
    print("\n分析真实转场后的质量...")
    scene_analyses = []
    
    for scene_poc in true_scenes:
        tn_stats = calculate_scene_quality(tn_frames, scene_poc, window=20)
        ic_stats = calculate_scene_quality(ic_frames, scene_poc, window=20)
        
        if tn_stats and ic_stats:
            analysis = {
                'scene_poc': scene_poc,
                'transnetv2': tn_stats,
                'icut': ic_stats,
                'psnr_diff': tn_stats['psnr_mean'] - ic_stats['psnr_mean']
            }
            scene_analyses.append(analysis)
    
    print("  成功分析 {} 个转场".format(len(scene_analyses)))
    
    # 统计结果
    if scene_analyses:
        tn_psnr_list = [a['transnetv2']['psnr_mean'] for a in scene_analyses]
        ic_psnr_list = [a['icut']['psnr_mean'] for a in scene_analyses]
        psnr_diffs = [a['psnr_diff'] for a in scene_analyses]
        
        tn_avg = sum(tn_psnr_list) / len(tn_psnr_list)
        ic_avg = sum(ic_psnr_list) / len(ic_psnr_list)
        
        print("\n真实转场后20帧质量统计:")
        print("  TransNetV2: 平均 PSNR = {:.4f} dB".format(tn_avg))
        print("  icut:       平均 PSNR = {:.4f} dB".format(ic_avg))
        print("  差异:       {:.4f} dB (TransNetV2 - icut)".format(tn_avg - ic_avg))
        
        # 统计谁在更多转场处质量更高
        tn_wins = sum(1 for d in psnr_diffs if d > 0)
        ic_wins = sum(1 for d in psnr_diffs if d < 0)
        ties = len(psnr_diffs) - tn_wins - ic_wins
        
        print("\n  单转场对比:")
        print("    TransNetV2 胜: {} 个转场".format(tn_wins))
        print("    icut 胜:       {} 个转场".format(ic_wins))
        if ties > 0:
            print("    平局:          {} 个转场".format(ties))
        
        return {
            'video_name': video_name,
            'scene_count': len(scene_analyses),
            'transnetv2_avg_psnr': tn_avg,
            'icut_avg_psnr': ic_avg,
            'psnr_diff': tn_avg - ic_avg,
            'transnetv2_wins': tn_wins,
            'icut_wins': ic_wins,
            'scene_analyses': scene_analyses
        }
    
    return None


def main():
    """主函数"""
    print("=" * 70)
    print("基于真实标签的转场质量分析")
    print("=" * 70)
    
    # 配置
    gt_dir = "/mnt/ec-data2/lh_data/bbc"
    logs_dir = "./logs"
    output_file = "./ground_truth_quality_analysis.json"
    
    # 检查目录
    if not os.path.exists(gt_dir):
        print("ERROR 真实标签目录不存在: {}".format(gt_dir))
        sys.exit(1)
    
    if not os.path.exists(logs_dir):
        print("ERROR 日志目录不存在: {}".format(logs_dir))
        sys.exit(1)
    
    # 分析所有视频
    all_results = []
    
    for video_num in range(1, 12):
        video_name = "bbc_{:02d}".format(video_num)
        
        result = analyze_video(video_name, gt_dir, logs_dir)
        if result:
            all_results.append(result)
    
    # 生成总体统计
    if all_results:
        print("\n" + "=" * 70)
        print("总体统计")
        print("=" * 70)
        
        total_scenes = sum(r['scene_count'] for r in all_results)
        total_tn_wins = sum(r['transnetv2_wins'] for r in all_results)
        total_ic_wins = sum(r['icut_wins'] for r in all_results)
        
        # 计算所有视频的平均 PSNR
        all_tn_psnr = sum(r['transnetv2_avg_psnr'] * r['scene_count'] for r in all_results) / total_scenes
        all_ic_psnr = sum(r['icut_avg_psnr'] * r['scene_count'] for r in all_results) / total_scenes
        
        print("  分析视频数: {}".format(len(all_results)))
        print("  真实转场总数: {}".format(total_scenes))
        print("\n  转场后20帧平均 PSNR:")
        print("    TransNetV2: {:.4f} dB".format(all_tn_psnr))
        print("    icut:       {:.4f} dB".format(all_ic_psnr))
        print("    差异:       {:.4f} dB".format(all_tn_psnr - all_ic_psnr))
        
        print("\n  单转场质量对比:")
        print("    TransNetV2 胜: {} / {} ({:.1f}%)".format(
            total_tn_wins, total_scenes, 100.0 * total_tn_wins / total_scenes
        ))
        print("    icut 胜:       {} / {} ({:.1f}%)".format(
            total_ic_wins, total_scenes, 100.0 * total_ic_wins / total_scenes
        ))
        
        # 按视频统计
        tn_video_wins = sum(1 for r in all_results if r['psnr_diff'] > 0)
        ic_video_wins = sum(1 for r in all_results if r['psnr_diff'] < 0)
        
        print("\n  视频级别对比:")
        print("    TransNetV2 胜: {} / {} 个视频".format(tn_video_wins, len(all_results)))
        print("    icut 胜:       {} / {} 个视频".format(ic_video_wins, len(all_results)))
        
        # 保存结果
        summary = {
            'total_videos': len(all_results),
            'total_scenes': total_scenes,
            'overall_transnetv2_psnr': all_tn_psnr,
            'overall_icut_psnr': all_ic_psnr,
            'overall_psnr_diff': all_tn_psnr - all_ic_psnr,
            'transnetv2_scene_wins': total_tn_wins,
            'icut_scene_wins': total_ic_wins,
            'transnetv2_video_wins': tn_video_wins,
            'icut_video_wins': ic_video_wins,
            'video_results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n结果已保存: {}".format(output_file))
        print("=" * 70)
    else:
        print("\nERROR 没有有效的分析结果")
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
