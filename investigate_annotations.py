#!/usr/bin/env python3
"""
调查 KITS23 数据集的真实标注格式和肾脏覆盖情况
"""

import nibabel as nib
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

def analyze_annotation_format(data_dir="."):
    """分析标注文件的真实格式"""
    
    print("=" * 60)
    print("KITS23 标注格式分析")
    print("=" * 60)
    
    segmentation_files = glob.glob(os.path.join(data_dir, "**", "segmentation.nii.gz"), recursive=True)
    
    if not segmentation_files:
        print("没有找到 segmentation.nii.gz 文件")
        return
    
    # 分析前10个文件的标注格式
    unique_labels_all = []
    label_distributions = []
    
    for seg_file in segmentation_files[:10]:
        try:
            seg = nib.load(seg_file)
            seg_data = seg.get_fdata()
            
            # 找到所有唯一的标签值
            unique_labels = np.unique(seg_data)
            unique_labels_all.extend(unique_labels)
            
            print(f"\n{os.path.basename(os.path.dirname(seg_file))}:")
            print(f"  形状: {seg_data.shape}")
            print(f"  唯一标签: {sorted(unique_labels)}")
            
            # 计算每个标签的体素数量
            label_counts = {}
            for label in unique_labels:
                if label > 0:  # 忽略背景
                    count = np.sum(seg_data == label)
                    label_counts[int(label)] = count
                    print(f"    标签 {int(label)}: {count:>8,} 体素")
            
            label_distributions.append(label_counts)
            
        except Exception as e:
            print(f"处理文件 {seg_file} 时出错: {e}")
    
    # 分析所有标签
    all_unique_labels = set(unique_labels_all)
    print(f"\n=== 全局标签分析 ===")
    print(f"所有发现的标签: {sorted(all_unique_labels)}")
    
    # 推断标签含义（基于KITS23数据集文档）
    label_mapping = {
        0: "背景",
        1: "肾脏",
        2: "肿瘤", 
        3: "囊肿"
    }
    
    print(f"\n=== 推断的标签含义 ===")
    for label in sorted(all_unique_labels):
        if label in label_mapping:
            print(f"  标签 {label}: {label_mapping[label]}")
        else:
            print(f"  标签 {label}: 未知")
    
    return all_unique_labels, label_distributions

def analyze_kidney_coverage_correct(segmentation_files, kidney_label=1):
    """正确分析肾脏覆盖情况"""
    
    print(f"\n" + "=" * 60)
    print(f"肾脏覆盖分析 (标签 {kidney_label})")
    print("=" + "=" * 60)
    
    results = []
    
    for seg_file in segmentation_files[:20]:  # 分析前20个文件
        try:
            seg = nib.load(seg_file)
            seg_data = seg.get_fdata()
            
            total_slices = seg_data.shape[0]
            
            # 检查每个切片是否有肾脏
            kidney_slices = []
            for slice_idx in range(total_slices):
                slice_data = seg_data[slice_idx, :, :]
                if np.any(slice_data == kidney_label):
                    kidney_slices.append(slice_idx)
            
            kidney_slice_count = len(kidney_slices)
            coverage_ratio = kidney_slice_count / total_slices if total_slices > 0 else 0
            
            # 计算肾脏的连续区域
            if kidney_slices:
                kidney_start = kidney_slices[0]
                kidney_end = kidney_slices[-1]
                kidney_span = kidney_end - kidney_start + 1
            else:
                kidney_start = kidney_end = kidney_span = 0
            
            results.append({
                'case_id': os.path.basename(os.path.dirname(seg_file)),
                'total_slices': total_slices,
                'kidney_slices': kidney_slice_count,
                'coverage_ratio': coverage_ratio,
                'kidney_start': kidney_start,
                'kidney_end': kidney_end,
                'kidney_span': kidney_span
            })
            
            print(f"{results[-1]['case_id']:15s} | "
                  f"总切片: {total_slices:3d} | "
                  f"肾切片: {kidney_slice_count:3d} | "
                  f"覆盖: {coverage_ratio:5.1%} | "
                  f"肾区域: [{kidney_start:3d}-{kidney_end:3d}] = {kidney_span:3d}切片")
                  
        except Exception as e:
            print(f"处理文件 {seg_file} 时出错: {e}")
    
    return results

def visualize_kidney_analysis(results):
    """可视化肾脏分析结果"""
    
    if not results:
        return
    
    total_slices = [r['total_slices'] for r in results]
    kidney_spans = [r['kidney_span'] for r in results]
    coverage_ratios = [r['coverage_ratio'] for r in results]
    
    plt.figure(figsize=(15, 5))
    
    # 总切片 vs 肾脏跨度
    plt.subplot(1, 3, 1)
    plt.scatter(total_slices, kidney_spans, alpha=0.6)
    plt.xlabel('总切片数量')
    plt.ylabel('肾脏区域跨度')
    plt.title('体积大小 vs 肾脏跨度')
    plt.grid(True, alpha=0.3)
    
    # 肾脏覆盖比例分布
    plt.subplot(1, 3, 2)
    plt.hist(coverage_ratios, bins=20, alpha=0.7, color='lightgreen')
    plt.xlabel('肾脏覆盖比例')
    plt.ylabel('病例数量')
    plt.title('肾脏覆盖比例分布')
    plt.grid(True, alpha=0.3)
    
    # 肾脏跨度分布
    plt.subplot(1, 3, 3)
    plt.hist(kidney_spans, bins=20, alpha=0.7, color='lightcoral')
    plt.xlabel('肾脏区域跨度(切片数)')
    plt.ylabel('病例数量')
    plt.title('肾脏区域大小分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kidney_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 肾脏分析图表已保存为: kidney_analysis.png")

def provide_roi_recommendation(results):
    """基于分析结果提供ROI推荐"""
    
    if not results:
        return
    
    kidney_spans = [r['kidney_span'] for r in results if r['kidney_span'] > 0]
    coverage_ratios = [r['coverage_ratio'] for r in results]
    
    if kidney_spans:
        avg_span = np.mean(kidney_spans)
        max_span = max(kidney_spans)
        min_span = min(kidney_spans)
        
        print(f"\n=== ROI推荐 ===")
        print(f"肾脏区域统计:")
        print(f"  平均跨度: {avg_span:.1f} 切片")
        print(f"  最小跨度: {min_span} 切片")
        print(f"  最大跨度: {max_span} 切片")
        print(f"  建议ROI大小: {int(avg_span + 20)} 切片 (平均值+缓冲)")
        
        if max(coverage_ratios) < 0.4:
            print(f"\n💡 关键发现: 肾脏只占总体积的很小部分!")
            print("✅ 强烈推荐使用肾脏ROI检测 + 裁剪")
            print("   可以显著减少计算量并保持细节")
        else:
            print(f"\n💡 发现: 肾脏分布较广")
            print("✅ 考虑使用较大的ROI区域或2.5D处理")

if __name__ == "__main__":
    # 首先分析标注格式
    all_labels, label_dists = analyze_annotation_format()
    
    # 然后分析肾脏覆盖
    segmentation_files = glob.glob(os.path.join(".", "**", "segmentation.nii.gz"), recursive=True)
    if segmentation_files:
        results = analyze_kidney_coverage_correct(segmentation_files, kidney_label=1)
        
        # 可视化结果
        visualize_kidney_analysis(results)
        
        # 提供推荐
        provide_roi_recommendation(results)
        
        print(f"\n🎯 下一步:")
        print("1. 查看 kidney_analysis.png 了解肾脏分布")
        print("2. 根据肾脏跨度确定ROI目标大小") 
        print("3. 实现基于真实标注的肾脏ROI检测")
    else:
        print("没有找到分割文件")
