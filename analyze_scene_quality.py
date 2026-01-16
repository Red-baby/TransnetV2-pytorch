# -*- coding: utf-8 -*-
"""
转场质量分析脚本
分析 transnetv2 独有转场（icut 未检测到）附近的编码质量
兼容 Python 2.7
"""

from __future__ import print_function
import re
import os
import sys
import json
from collections import defaultdict


def pts_to_poc(pts, fps=25):
    """
    将 PTS (毫秒) 转换为 POC (帧序号)
    
    Args:
        pts: Presentation timestamp (毫秒)
        fps: 帧率
    
    Returns:
        POC (帧序号)
    """
    # POC = PTS / (1000/fps) = PTS * fps / 1000
    return int(round(pts * fps / 1000.0))


def read_keyframe_list(keyframe_file, fps=25):
    """
    读取 keyframe 文件，返回 POC 列表
    
    Args:
        keyframe_file: keyframe PTS 文件路径
        fps: 帧率
    
    Returns:
        list of int: POC 列表
    """
    pocs = []
    
    try:
        with open(keyframe_file, 'r') as f:
            lines = f.readlines()
            
            # 跳过第一行（帧数统计）
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 1:
                    pts = int(parts[0])
                    poc = pts_to_poc(pts, fps)
                    pocs.append(poc)
        
        return sorted(set(pocs))  # 去重并排序
        
    except IOError as e:
        print("ERROR 无法读取 keyframe 文件: {}".format(keyframe_file))
        print("ERROR {}".format(e))
        return []


def find_unique_scenes(transnetv2_pocs, icut_pocs, tolerance=3):
    """
    找出 transnetv2 独有的转场（icut 未检测到）
    
    Args:
        transnetv2_pocs: transnetv2 检测的 POC 列表
        icut_pocs: icut 检测的 POC 列表
        tolerance: 容忍度（±帧数）
    
    Returns:
        list of int: transnetv2 独有的转场 POC
    """
    unique_scenes = []
    
    for tn_poc in transnetv2_pocs:
        # 检查是否在 icut 中有匹配（容忍度范围内）
        matched = False
        for ic_poc in icut_pocs:
            if abs(tn_poc - ic_poc) <= tolerance:
                matched = True
                break
        
        if not matched:
            unique_scenes.append(tn_poc)
    
    return unique_scenes


def parse_q265_log(log_file):
    """
    解析 q265 日志文件，返回按 POC 索引的帧信息
    
    Returns:
        dict: {poc: {'type': 'b', 'qp': 37.0, 'psnr_y': 42.9368, ...}, ...}
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
                        'psnr_y': float(match.group(5)),
                        'psnr_u': float(match.group(6)),
                        'psnr_v': float(match.group(7))
                    }
    except IOError as e:
        print("ERROR 无法读取日志文件: {}".format(log_file))
        return {}
    
    return frames


def calculate_region_stats(frames_dict, poc_start, poc_end):
    """
    计算指定 POC 范围的统计信息
    
    Args:
        frames_dict: 帧信息字典 {poc: frame_info}
        poc_start: 起始 POC
        poc_end: 结束 POC (包含)
    
    Returns:
        dict: 统计信息
    """
    psnr_values = []
    fluctuations = []
    
    # 提取范围内的帧
    for poc in range(poc_start, poc_end + 1):
        if poc in frames_dict:
            psnr_values.append(frames_dict[poc]['psnr_y'])
    
    if not psnr_values:
        return None
    
    # 计算相邻帧波动
    for i in range(len(psnr_values) - 1):
        fluctuation = abs(psnr_values[i+1] - psnr_values[i])
        fluctuations.append(fluctuation)
    
    return {
        'count': len(psnr_values),
        'psnr_mean': sum(psnr_values) / len(psnr_values),
        'psnr_min': min(psnr_values),
        'psnr_max': max(psnr_values),
        'fluctuation_mean': sum(fluctuations) / len(fluctuations) if fluctuations else 0.0,
        'fluctuation_max': max(fluctuations) if fluctuations else 0.0
    }


def analyze_scene(scene_poc, transnetv2_frames, icut_frames):
    """
    分析单个转场附近的质量跳变
    
    对比转场前10帧和后10帧的平均 PSNR，衡量质量是否突然变差
    
    Args:
        scene_poc: 转场 POC
        transnetv2_frames: transnetv2 编码的帧信息字典
        icut_frames: icut 编码的帧信息字典
    
    Returns:
        dict: 分析结果
    """
    # 定义区域（前后各10帧）
    # 前10帧：[scene_poc - 10, scene_poc - 1]
    # 后10帧（含转场）：[scene_poc, scene_poc + 9]
    
    before_start = scene_poc - 10
    before_end = scene_poc - 1
    after_start = scene_poc
    after_end = scene_poc + 9
    
    # 计算 transnetv2 的统计
    tn_before = calculate_region_stats(transnetv2_frames, before_start, before_end)
    tn_after = calculate_region_stats(transnetv2_frames, after_start, after_end)
    
    # 计算 icut 的统计
    ic_before = calculate_region_stats(icut_frames, before_start, before_end)
    ic_after = calculate_region_stats(icut_frames, after_start, after_end)
    
    if not tn_before or not tn_after or not ic_before or not ic_after:
        return None
    
    # 计算质量跳变（后10帧 - 前10帧）
    tn_quality_drop = tn_after['psnr_mean'] - tn_before['psnr_mean']
    ic_quality_drop = ic_after['psnr_mean'] - ic_before['psnr_mean']
    
    result = {
        'scene_poc': scene_poc,
        'transnetv2': {
            'before_10': tn_before,
            'after_10': tn_after,
            'quality_drop': tn_quality_drop  # 负值表示质量下降
        },
        'icut': {
            'before_10': ic_before,
            'after_10': ic_after,
            'quality_drop': ic_quality_drop
        },
        'comparison': {
            # TransNetV2 和 icut 在这个转场处的质量跳变差异
            'quality_drop_diff': tn_quality_drop - ic_quality_drop,
            # 后10帧的质量对比
            'psnr_diff_after': tn_after['psnr_mean'] - ic_after['psnr_mean'],
            # 后10帧的波动对比
            'fluctuation_diff_after': tn_after['fluctuation_mean'] - ic_after['fluctuation_mean']
        }
    }
    
    return result


def analyze_video(video_name, keyframe_dir, logs_dir, fps=25):
    """
    分析一个视频
    """
    print("\n" + "=" * 70)
    print("分析视频: {}".format(video_name))
    print("=" * 70)
    
    # 读取 keyframe 列表
    tn_keyframe_file = os.path.join(keyframe_dir, "{}_keyframe_PTS_transnetv2.txt".format(video_name))
    ic_keyframe_file = os.path.join(keyframe_dir, "{}_keyframe_PTS_icut.txt".format(video_name))
    
    print("读取 keyframe 列表...")
    tn_pocs = read_keyframe_list(tn_keyframe_file, fps)
    ic_pocs = read_keyframe_list(ic_keyframe_file, fps)
    
    if not tn_pocs or not ic_pocs:
        print("ERROR keyframe 列表读取失败")
        return None
    
    print("  TransNetV2: {} 个转场".format(len(tn_pocs)))
    print("  icut:       {} 个转场".format(len(ic_pocs)))
    
    # 找出 transnetv2 独有的转场
    unique_scenes = find_unique_scenes(tn_pocs, ic_pocs, tolerance=3)
    print("  TransNetV2 独有转场: {} 个".format(len(unique_scenes)))
    
    if not unique_scenes:
        print("  无独有转场，跳过分析")
        return None
    
    print("  独有转场 POC: {}".format(unique_scenes[:10] + (['...'] if len(unique_scenes) > 10 else [])))
    
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
    
    # 分析每个独有转场
    print("\n分析独有转场的质量跳变...")
    scene_analyses = []
    
    for scene_poc in unique_scenes:
        analysis = analyze_scene(scene_poc, tn_frames, ic_frames)
        if analysis:
            scene_analyses.append(analysis)
    
    print("  成功分析 {} 个转场".format(len(scene_analyses)))
    
    # 打印统计
    if scene_analyses:
        print("\n转场质量跳变分析:")
        
        tn_drops = [a['transnetv2']['quality_drop'] for a in scene_analyses]
        ic_drops = [a['icut']['quality_drop'] for a in scene_analyses]
        drop_diffs = [a['comparison']['quality_drop_diff'] for a in scene_analyses]
        
        tn_avg_drop = sum(tn_drops) / len(tn_drops)
        ic_avg_drop = sum(ic_drops) / len(ic_drops)
        
        print("\n  转场前后质量变化 (后10帧 - 前10帧):")
        print("    TransNetV2: {:.4f} dB {}".format(
            tn_avg_drop,
            "(下降)" if tn_avg_drop < 0 else "(提升)"
        ))
        print("    icut:       {:.4f} dB {}".format(
            ic_avg_drop,
            "(下降)" if ic_avg_drop < 0 else "(提升)"
        ))
        print("    差异:       {:.4f} dB".format(sum(drop_diffs) / len(drop_diffs)))
        
        # 统计明显质量下降的转场（下降超过0.5dB）
        tn_bad_scenes = sum(1 for d in tn_drops if d < -0.5)
        ic_bad_scenes = sum(1 for d in ic_drops if d < -0.5)
        
        print("\n  明显质量下降的转场数 (>0.5dB):")
        print("    TransNetV2: {} / {}".format(tn_bad_scenes, len(tn_drops)))
        print("    icut:       {} / {}".format(ic_bad_scenes, len(ic_drops)))
        
        # 后10帧的质量和波动对比
        psnr_diffs = [a['comparison']['psnr_diff_after'] for a in scene_analyses]
        fluct_diffs = [a['comparison']['fluctuation_diff_after'] for a in scene_analyses]
        
        print("\n  后10帧质量对比:")
        print("    平均 PSNR 差异: {:.4f} dB (TransNetV2 - icut)".format(
            sum(psnr_diffs) / len(psnr_diffs)
        ))
        print("    平均波动差异:   {:.4f} (TransNetV2 - icut)".format(
            sum(fluct_diffs) / len(fluct_diffs)
        ))
    
    return {
        'video_name': video_name,
        'unique_scene_count': len(unique_scenes),
        'unique_scenes': unique_scenes,
        'scene_analyses': scene_analyses
    }


def main():
    """主函数"""
    print("=" * 70)
    print("转场质量分析")
    print("=" * 70)
    
    # 配置
    keyframe_dir = "/mnt/ec-data2/lh_data/bbc"
    logs_dir = "./logs"
    output_file = "./scene_quality_analysis.json"
    fps = 25
    
    # 检查目录
    if not os.path.exists(keyframe_dir):
        print("ERROR keyframe 目录不存在: {}".format(keyframe_dir))
        sys.exit(1)
    
    if not os.path.exists(logs_dir):
        print("ERROR 日志目录不存在: {}".format(logs_dir))
        sys.exit(1)
    
    # 分析所有视频
    all_results = []
    
    for video_num in range(1, 12):
        video_name = "bbc_{:02d}".format(video_num)
        
        result = analyze_video(video_name, keyframe_dir, logs_dir, fps)
        if result:
            all_results.append(result)
    
    # 保存结果
    if all_results:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 70)
        print("分析完成!")
        print("结果已保存: {}".format(output_file))
        print("=" * 70)
        
        # 总体统计
        total_unique = sum(r['unique_scene_count'] for r in all_results)
        print("\n总体统计:")
        print("  分析视频数: {}".format(len(all_results)))
        print("  独有转场总数: {}".format(total_unique))
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
