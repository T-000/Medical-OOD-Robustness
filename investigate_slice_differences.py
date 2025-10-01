#!/usr/bin/env python3
"""
调查 KITS23 数据集中切片数量的差异并分析解决方案
"""

import nibabel as nib
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def analyze_slice_distribution(data_dir="."):
    """分析切片数量的分布情况"""
    
    print("=" * 60)
    print("KITS23 数据集切片数量分析")
    print("=" * 60)
    
    # 查找所有 imaging 文件
    imaging_files = glob.glob(os.path.join(data_dir, "**", "imaging.nii.gz"), recursive=True)
    segmentation_files = glob.glob(os.path.join(data_dir, "**", "segmentation.nii.gz"), recursive=True)
    
    print(f"找到 {len(imaging_files)} 个影像文件")
    print(f"找到 {len(segmentation_files)} 个分割文件")
    
    # 收集切片数量信息
    imaging_slices = []
    segmentation_slices = []
    kidney_coverage = []  # 肾脏覆盖的切片比例
    
    for img_file in imaging_files[:50]:  # 先分析前50个文件避免太慢
        try:
            # 加载影像
            img = nib.load(img_file)
            img_data = img.get_fdata()
            imaging_slices.append(img_data.shape[0])
            
            # 找到对应的分割文件
            seg_file = img_file.replace("imaging.nii.gz", "segmentation.nii.gz")
            if os.path.exists(seg_file):
                seg = nib.load(seg_file)
                seg_data = seg.get_fdata()
                segmentation_slices.append(seg_data.shape[0])
                
                # 计算肾脏覆盖情况
                kidney_present = np.any(seg_data == 1, axis=(1, 2))  # 假设1是肾脏标签
                kidney_coverage.append(np.sum(kidney_present) / seg_data.shape[0])
                
                print(f"{os.path.basename(os.path.dirname(img_file)):15s} | "
                      f"影像切片: {img_data.shape[0]:3d} | "
                      f"分割切片: {seg_data.shape[0]:3d} | "
                      f"肾脏覆盖: {np.sum(kidney_present):3d} slices ({kidney_coverage[-1]:.1%})")
            
        except Exception as e:
            print(f"处理文件 {img_file} 时出错: {e}")
    
    return imaging_slices, segmentation_slices, kidney_coverage

def suggest_solutions(imaging_slices, kidney_coverage):
    """基于分析结果提供解决方案建议"""
    
    print("\n" + "=" * 60)
    print("解决方案分析")
    print("=" * 60)
    
    imaging_slices = np.array(imaging_slices)
    
    print(f"切片数量统计:")
    print(f"  最小值: {imaging_slices.min()}")
    print(f"  最大值: {imaging_slices.max()}")
    print(f"  平均值: {imaging_slices.mean():.1f}")
    print(f"  中位数: {np.median(imaging_slices)}")
    print(f"  标准差: {imaging_slices.std():.1f}")
    
    # 分析分布
    slice_counts = Counter(imaging_slices)
    common_sizes = slice_counts.most_common(5)
    print(f"\n最常见的切片数量: {common_sizes}")
    
    # 根据数据特征推荐解决方案
    max_slices = imaging_slices.max()
    min_slices = imaging_slices.min()
    
    print(f"\n=== 推荐解决方案 ===")
    
    if max_slices - min_slices > 200:
        print("❌ 情况: 切片数量差异极大 (>200 slices)")
        print("✅ 推荐方案 1: 肾脏ROI检测 + 自动裁剪")
        print("   - 使用肾脏检测算法定位肾脏区域")
        print("   - 只裁剪包含肾脏的切片进行训练")
        print("   - 可以保持一致的 ~80-120 切片数量")
        
        print("\n✅ 推荐方案 2: 2.5D切片处理")
        print("   - 按2D切片处理，但提供相邻切片作为上下文")
        print("   - 适合CNN架构，内存效率高")
        print("   - 可以处理任意长度的体积")
        
    elif max_slices - min_slices > 50:
        print("⚠️  情况: 切片数量差异中等 (50-200 slices)")
        print("✅ 推荐方案: 智能填充/裁剪")
        print("   - 基于肾脏位置的中心裁剪")
        print("   - 或使用百分位数选择目标尺寸")
        
    else:
        print("✅ 情况: 切片数量差异较小 (<50 slices)")
        print("✅ 推荐方案: 标准填充/裁剪")
        print("   - 简单的填充或中心裁剪到固定尺寸")
    
    # 肾脏覆盖分析
    if kidney_coverage:
        avg_coverage = np.mean(kidney_coverage)
        print(f"\n肾脏覆盖分析: 平均 {avg_coverage:.1%} 的切片包含肾脏")
        
        if avg_coverage < 0.3:
            print("💡 洞察: 肾脏只占总体积的小部分，ROI裁剪会很有效!")
        else:
            print("💡 洞察: 肾脏分布较广，需要考虑更大的裁剪区域")

def create_visualization(imaging_slices, kidney_coverage):
    """创建可视化图表"""
    
    plt.figure(figsize=(15, 5))
    
    # 切片数量分布
    plt.subplot(1, 3, 1)
    plt.hist(imaging_slices, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('切片数量')
    plt.ylabel('频次')
    plt.title('切片数量分布')
    plt.grid(True, alpha=0.3)
    
    # 肾脏覆盖分布
    plt.subplot(1, 3, 2)
    plt.hist(kidney_coverage, bins=20, alpha=0.7, color='lightcoral')
    plt.xlabel('肾脏覆盖比例')
    plt.ylabel('频次')
    plt.title('肾脏覆盖比例分布')
    plt.grid(True, alpha=0.3)
    
    # 切片数量 vs 肾脏覆盖
    plt.subplot(1, 3, 3)
    plt.scatter(imaging_slices[:len(kidney_coverage)], kidney_coverage, alpha=0.6)
    plt.xlabel('总切片数量')
    plt.ylabel('肾脏覆盖比例')
    plt.title('体积大小 vs 肾脏覆盖')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('slice_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 可视化图表已保存为: slice_analysis.png")

def kidney_roi_detection_example():
    """提供肾脏ROI检测的示例代码"""
    
    print("\n" + "=" * 60)
    print("肾脏ROI检测示例代码")
    print("=" * 60)
    
    example_code = '''
def detect_kidney_roi(segmentation_volume):
    """检测肾脏区域的ROI"""
    # 假设标签: 1=肾脏, 2=肿瘤, 3=囊肿
    kidney_mask = (segmentation_volume == 1)
    
    # 找到包含肾脏的切片
    kidney_slices = np.any(kidney_mask, axis=(1, 2))
    kidney_indices = np.where(kidney_slices)[0]
    
    if len(kidney_indices) == 0:
        return None  # 没有找到肾脏
    
    # 计算肾脏边界 (增加一些边界缓冲)
    start_slice = max(0, kidney_indices[0] - 5)
    end_slice = min(segmentation_volume.shape[0], kidney_indices[-1] + 5)
    
    return start_slice, end_slice

def extract_kidney_roi(imaging_volume, segmentation_volume, target_slices=128):
    """提取肾脏ROI区域"""
    roi = detect_kidney_roi(segmentation_volume)
    
    if roi is None:
        #  fallback: 中心裁剪
        center = imaging_volume.shape[0] // 2
        start_slice = center - target_slices // 2
        end_slice = start_slice + target_slices
    else:
        start_slice, end_slice = roi
        # 如果ROI太小，扩展到目标大小
        current_slices = end_slice - start_slice
        if current_slices < target_slices:
            padding = target_slices - current_slices
            start_slice = max(0, start_slice - padding // 2)
            end_slice = min(imaging_volume.shape[0], start_slice + target_slices)
    
    return imaging_volume[start_slice:end_slice, :, :]
'''
    print(example_code)

if __name__ == "__main__":
    # 分析数据
    imaging_slices, segmentation_slices, kidney_coverage = analyze_slice_distribution()
    
    if imaging_slices:
        # 提供解决方案建议
        suggest_solutions(imaging_slices, kidney_coverage)
        
        # 创建可视化
        create_visualization(imaging_slices, kidney_coverage)
        
        # 显示示例代码
        kidney_roi_detection_example()
        
        print(f"\n🎯 下一步建议:")
        print("1. 查看 slice_analysis.png 了解数据分布")
        print("2. 根据推荐方案选择预处理策略") 
        print("3. 实现肾脏ROI检测来标准化输入尺寸")
        print("4. 修改 data_preprocessor.py 使用新的预处理方法")
    else:
        print("没有找到足够的影像文件进行分析")
