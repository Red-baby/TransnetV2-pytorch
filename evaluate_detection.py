# -*- coding: utf-8 -*-
"""
场景检测准确性评估
基于真实标签计算 Precision、Recall、F1-Score
兼容 Python 2.7
"""

from __future__ import print_function
import os
import sys
import json


def read_ground_truth(gt_file):
    """
    读取真实转场标签文件
    
    Returns:
        list of int: 真实转场位置（POC）
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
        print("ERROR 无法读取真实标签: {}".format(gt_file))
        return []


def pts_to_poc(pts, fps=25):
    """
    将 PTS (毫秒) 转换为 POC (帧序号)
    """
    return int(round(pts * fps / 1000.0))


def read_keyframe_list(keyframe_file, fps=25):
    """
    读取检测到的 keyframe 文件
    
    Returns:
        list of int: 检测到的转场 POC 列表
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
        return []


def match_scenes(predicted, ground_truth, tolerance=3):
    """
    匹配预测的转场和真实转场
    
    Args:
        predicted: 预测的转场列表
        ground_truth: 真实转场列表
        tolerance: 容忍度（±帧数）
    
    Returns:
        tuple: (true_positives, false_positives, false_negatives)
    """
    matched_gt = set()  # 已匹配的真实转场
    matched_pred = set()  # 已匹配的预测转场
    
    # 对每个预测的转场，寻找最近的真实转场
    for pred_poc in predicted:
        matched = False
        for gt_poc in ground_truth:
            if abs(pred_poc - gt_poc) <= tolerance:
                matched_gt.add(gt_poc)
                matched_pred.add(pred_poc)
                matched = True
                break
        
    true_positives = len(matched_pred)  # 正确检测到的转场
    false_positives = len(predicted) - true_positives  # 误检的转场
    false_negatives = len(ground_truth) - len(matched_gt)  # 漏检的转场
    
    return true_positives, false_positives, false_negatives


def calculate_metrics(tp, fp, fn):
    """
    计算评估指标
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
    
    Returns:
        dict: 包含 precision, recall, f1 的字典
    """
    # Precision = TP / (TP + FP)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def evaluate_video(video_name, gt_dir, keyframe_dir, fps=25, tolerance=3):
    """
    评估一个视频的检测准确性
    """
    print("\n" + "=" * 70)
    print("评估视频: {}".format(video_name))
    print("=" * 70)
    
    # 读取真实标签
    gt_file = os.path.join(gt_dir, "{}.txt".format(video_name))
    ground_truth = read_ground_truth(gt_file)
    
    if not ground_truth:
        print("ERROR 真实标签读取失败")
        return None
    
    print("  真实转场数: {}".format(len(ground_truth)))
    
    # 读取检测结果
    tn_file = os.path.join(keyframe_dir, "{}_keyframe_PTS_transnetv2.txt".format(video_name))
    ic_file = os.path.join(keyframe_dir, "{}_keyframe_PTS_icut.txt".format(video_name))
    
    tn_predicted = read_keyframe_list(tn_file, fps)
    ic_predicted = read_keyframe_list(ic_file, fps)
    
    if not tn_predicted or not ic_predicted:
        print("ERROR 检测结果读取失败")
        return None
    
    print("  TransNetV2 检测: {}".format(len(tn_predicted)))
    print("  icut 检测:       {}".format(len(ic_predicted)))
    
    # 匹配和评估
    print("\n匹配转场 (容忍度 ±{} 帧)...".format(tolerance))
    
    tn_tp, tn_fp, tn_fn = match_scenes(tn_predicted, ground_truth, tolerance)
    ic_tp, ic_fp, ic_fn = match_scenes(ic_predicted, ground_truth, tolerance)
    
    tn_metrics = calculate_metrics(tn_tp, tn_fp, tn_fn)
    ic_metrics = calculate_metrics(ic_tp, ic_fp, ic_fn)
    
    # 打印结果
    print("\nTransNetV2:")
    print("  TP={}, FP={}, FN={}".format(tn_tp, tn_fp, tn_fn))
    print("  Precision: {:.4f} ({}/{})".format(
        tn_metrics['precision'], tn_tp, tn_tp + tn_fp
    ))
    print("  Recall:    {:.4f} ({}/{})".format(
        tn_metrics['recall'], tn_tp, tn_tp + tn_fn
    ))
    print("  F1-Score:  {:.4f}".format(tn_metrics['f1']))
    
    print("\nicut:")
    print("  TP={}, FP={}, FN={}".format(ic_tp, ic_fp, ic_fn))
    print("  Precision: {:.4f} ({}/{})".format(
        ic_metrics['precision'], ic_tp, ic_tp + ic_fp
    ))
    print("  Recall:    {:.4f} ({}/{})".format(
        ic_metrics['recall'], ic_tp, ic_tp + ic_fn
    ))
    print("  F1-Score:  {:.4f}".format(ic_metrics['f1']))
    
    # 对比
    print("\n对比:")
    print("  F1-Score 差异: {:.4f} (TransNetV2 - icut)".format(
        tn_metrics['f1'] - ic_metrics['f1']
    ))
    
    winner = "TransNetV2" if tn_metrics['f1'] > ic_metrics['f1'] else "icut"
    if abs(tn_metrics['f1'] - ic_metrics['f1']) < 0.01:
        winner = "平局"
    print("  本视频胜者: {}".format(winner))
    
    return {
        'video_name': video_name,
        'ground_truth_count': len(ground_truth),
        'transnetv2': {
            'predicted_count': len(tn_predicted),
            'metrics': tn_metrics
        },
        'icut': {
            'predicted_count': len(ic_predicted),
            'metrics': ic_metrics
        }
    }


def main():
    """主函数"""
    print("=" * 70)
    print("场景检测准确性评估")
    print("=" * 70)
    
    # 配置
    gt_dir = "/mnt/ec-data2/lh_data/bbc"
    keyframe_dir = "/mnt/ec-data2/lh_data/bbc"
    output_file = "./detection_accuracy.json"
    fps = 25
    tolerance = 3  # ±3 帧容忍度
    
    print("配置:")
    print("  真实标签目录: {}".format(gt_dir))
    print("  检测结果目录: {}".format(keyframe_dir))
    print("  容忍度: ±{} 帧".format(tolerance))
    print("  帧率: {} fps".format(fps))
    
    # 检查目录
    if not os.path.exists(gt_dir):
        print("\nERROR 目录不存在: {}".format(gt_dir))
        sys.exit(1)
    
    # 评估所有视频
    all_results = []
    
    for video_num in range(1, 12):
        video_name = "bbc_{:02d}".format(video_num)
        
        result = evaluate_video(video_name, gt_dir, keyframe_dir, fps, tolerance)
        if result:
            all_results.append(result)
    
    # 总体统计
    if all_results:
        print("\n" + "=" * 70)
        print("总体统计")
        print("=" * 70)
        
        # 累计 TP, FP, FN
        tn_total_tp = sum(r['transnetv2']['metrics']['tp'] for r in all_results)
        tn_total_fp = sum(r['transnetv2']['metrics']['fp'] for r in all_results)
        tn_total_fn = sum(r['transnetv2']['metrics']['fn'] for r in all_results)
        
        ic_total_tp = sum(r['icut']['metrics']['tp'] for r in all_results)
        ic_total_fp = sum(r['icut']['metrics']['fp'] for r in all_results)
        ic_total_fn = sum(r['icut']['metrics']['fn'] for r in all_results)
        
        # 计算总体指标
        tn_overall = calculate_metrics(tn_total_tp, tn_total_fp, tn_total_fn)
        ic_overall = calculate_metrics(ic_total_tp, ic_total_fp, ic_total_fn)
        
        print("\nTransNetV2 总体:")
        print("  Precision: {:.4f}".format(tn_overall['precision']))
        print("  Recall:    {:.4f}".format(tn_overall['recall']))
        print("  F1-Score:  {:.4f}".format(tn_overall['f1']))
        
        print("\nicut 总体:")
        print("  Precision: {:.4f}".format(ic_overall['precision']))
        print("  Recall:    {:.4f}".format(ic_overall['recall']))
        print("  F1-Score:  {:.4f}".format(ic_overall['f1']))
        
        print("\n总体对比:")
        print("  F1-Score: TransNetV2 {:.4f} vs icut {:.4f}".format(
            tn_overall['f1'], ic_overall['f1']
        ))
        print("  差异: {:.4f}".format(tn_overall['f1'] - ic_overall['f1']))
        
        # 视频级别胜负
        tn_video_wins = sum(1 for r in all_results 
                           if r['transnetv2']['metrics']['f1'] > r['icut']['metrics']['f1'])
        ic_video_wins = sum(1 for r in all_results 
                           if r['icut']['metrics']['f1'] > r['transnetv2']['metrics']['f1'])
        
        print("\n视频级别胜负:")
        print("  TransNetV2: {} / {} 个视频".format(tn_video_wins, len(all_results)))
        print("  icut:       {} / {} 个视频".format(ic_video_wins, len(all_results)))
        
        # 保存结果
        summary = {
            'tolerance': tolerance,
            'fps': fps,
            'total_videos': len(all_results),
            'transnetv2_overall': tn_overall,
            'icut_overall': ic_overall,
            'transnetv2_video_wins': tn_video_wins,
            'icut_video_wins': ic_video_wins,
            'video_results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n结果已保存: {}".format(output_file))
        print("=" * 70)
    else:
        print("\nERROR 没有有效的评估结果")
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
