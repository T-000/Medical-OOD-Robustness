#!/usr/bin/env python3
"""
KITS23 预处理器 - 最终版本
使用 target_depth=128, 保持512×512分辨率
"""

import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

class KITS23Preprocessor:
    def __init__(self, target_depth=128, ct_min=-100, ct_max=400):
        """
        Args:
            target_depth: 目标深度 = 128切片
            ct_min: CT值下限 = -100 (保留肾脏组织)
            ct_max: CT值上限 = 400 (裁剪高密度区域)
        """
        self.target_depth = target_depth
        self.ct_min = ct_min
        self.ct_max = ct_max
        
        # 官方标签定义
        self.LABELS = {
            0: "background",
            1: "kidney", 
            2: "tumor",
            3: "cyst"
        }
    
    def detect_kidney_roi(self, segmentation_volume):
        """检测肾脏ROI区域（包括肾脏、肿瘤、囊肿）"""
        # 肾脏区域 = 肾脏 + 肿瘤 + 囊肿
        kidney_region = (segmentation_volume == 1) | (segmentation_volume == 2) | (segmentation_volume == 3)
        
        # 找到包含肾脏区域的切片
        kidney_slices = np.any(kidney_region, axis=(1, 2))
        kidney_indices = np.where(kidney_slices)[0]
        
        if len(kidney_indices) == 0:
            print("Warning: No kidney region found in volume")
            return None
        
        # 计算ROI边界（增加10切片缓冲）
        buffer = 10
        start_slice = max(0, kidney_indices[0] - buffer)
        end_slice = min(segmentation_volume.shape[0], kidney_indices[-1] + buffer + 1)
        
        print(f"肾脏区域: 切片 [{kidney_indices[0]}-{kidney_indices[-1]}]")
        print(f"ROI裁剪: [{start_slice}-{end_slice}] (+{buffer}切片缓冲)")
        
        return start_slice, end_slice
    
    def normalize_ct(self, volume):
        """CT值标准化到 [0, 1]"""
        volume = np.clip(volume, self.ct_min, self.ct_max)
        volume = (volume - self.ct_min) / (self.ct_max - self.ct_min)
        return volume
    
    def resize_depth(self, volume, target_depth, is_segmentation=False):
        """调整深度维度到128切片，保持512×512分辨率"""
        current_depth = volume.shape[0]
        
        if current_depth == target_depth:
            return volume
        
        print(f"深度调整: {current_depth} -> {target_depth} 切片")
        
        # 选择插值方法
        mode = 'nearest' if is_segmentation else 'trilinear'
        
        # 转换为Tensor: [D, H, W] -> [1, 1, D, H, W]
        volume_tensor = torch.from_numpy(volume.astype(np.float32))
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        
        # 调整深度，保持512×512不变
        resized = F.interpolate(
            volume_tensor, 
            size=(target_depth, 512, 512),  # 保持512×512分辨率!
            mode=mode,
            align_corners=False if mode == 'trilinear' else None
        )
        
        return resized.squeeze().numpy()
    
    def preprocess(self, imaging_path, segmentation_path):
        """完整的预处理流程"""
        print(f"\n📁 处理病例: {os.path.basename(os.path.dirname(imaging_path))}")
        print("-" * 50)
        
        # 1. 加载数据
        imaging = nib.load(imaging_path)
        segmentation = nib.load(segmentation_path)
        
        imaging_data = imaging.get_fdata()
        segmentation_data = segmentation.get_fdata()
        
        print(f"原始数据:")
        print(f"  影像: {imaging_data.shape} (范围: [{imaging_data.min():.0f}, {imaging_data.max():.0f}])")
        print(f"  分割: {segmentation_data.shape}")
        
        # 2. 检测肾脏ROI并裁剪
        roi = self.detect_kidney_roi(segmentation_data)
        
        if roi:
            start, end = roi
            imaging_roi = imaging_data[start:end, :, :]
            segmentation_roi = segmentation_data[start:end, :, :]
            print(f"ROI裁剪后: {imaging_roi.shape}")
        else:
            imaging_roi = imaging_data
            segmentation_roi = segmentation_data
            print("使用完整体积")
        
        # 3. 调整深度到128切片
        imaging_roi = self.resize_depth(imaging_roi, self.target_depth, is_segmentation=False)
        segmentation_roi = self.resize_depth(segmentation_roi, self.target_depth, is_segmentation=True)
        
        # 4. CT值标准化
        imaging_roi = self.normalize_ct(imaging_roi)
        
        # 5. 转换为Tensor
        imaging_tensor = torch.from_numpy(imaging_roi.astype(np.float32)).unsqueeze(0)  # [1, D, H, W]
        segmentation_tensor = torch.from_numpy(segmentation_roi.astype(np.int64)).unsqueeze(0)
        
        print(f"最终结果:")
        print(f"  影像: {imaging_tensor.shape} (范围: [{imaging_tensor.min():.3f}, {imaging_tensor.max():.3f}])")
        print(f"  分割: {segmentation_tensor.shape} (标签: {torch.unique(segmentation_tensor).tolist()})")
        
        return imaging_tensor, segmentation_tensor

def main():
    """主函数"""
    print("🚀 KITS23 预处理器 - 最终版本")
    print("参数: depth=128, 分辨率=512×512, CT范围=[-100,400]")
    print("=" * 60)
    
    # 初始化预处理器
    preprocessor = KITS23Preprocessor(target_depth=128)
    
    # 在 dataset 目录中查找病例
    dataset_path = "dataset"
    cases = glob.glob(f"{dataset_path}/case_*/imaging.nii.gz", recursive=True)
    
    if not cases:
        print("❌ 在 dataset 目录中找不到病例")
        return
    
    print(f"找到 {len(cases)} 个病例")
    
    # 测试前3个病例
    for case_path in cases[:3]:
        seg_path = case_path.replace("imaging.nii.gz", "segmentation.nii.gz")
        
        if os.path.exists(seg_path):
            try:
                image_tensor, seg_tensor = preprocessor.preprocess(case_path, seg_path)
                print("✅ 预处理成功!\n")
                
            except Exception as e:
                print(f"❌ 错误: {e}\n")
        else:
            print(f"❌ 找不到分割文件: {seg_path}\n")

if __name__ == "__main__":
    main()
