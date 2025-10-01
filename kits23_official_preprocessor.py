#!/usr/bin/env python3
"""
基于 KITS23 官方标签定义的预处理器
标签: 1=肾脏, 2=肿瘤, 3=囊肿
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

class KITS23Preprocessor:
    def __init__(self, target_depth=128, target_size=512, ct_min=-100, ct_max=400):
        """
        Args:
            target_depth: 目标深度（切片数量）
            target_size: 目标平面尺寸（保持512×512高分辨率）
            ct_min: CT值裁剪下限（保留肾脏相关组织）
            ct_max: CT值裁剪上限
        """
        self.target_depth = target_depth
        self.target_size = target_size
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
        
        # 计算ROI边界（增加15%的缓冲）
        kidney_span = kidney_indices[-1] - kidney_indices[0] + 1
        buffer = max(5, int(kidney_span * 0.15))  # 至少5切片缓冲
        
        start_slice = max(0, kidney_indices[0] - buffer)
        end_slice = min(segmentation_volume.shape[0], kidney_indices[-1] + buffer + 1)
        
        print(f"Kidney ROI: slices [{kidney_indices[0]}-{kidney_indices[-1]}] -> "
              f"ROI [{start_slice}-{end_slice}] with {buffer} slice buffer")
        
        return start_slice, end_slice
    
    def normalize_ct(self, volume):
        """CT值标准化"""
        # 裁剪到合理的CT值范围
        volume = np.clip(volume, self.ct_min, self.ct_max)
        # 标准化到 [0, 1]
        volume = (volume - self.ct_min) / (self.ct_max - self.ct_min)
        return volume
    
    def resize_depth(self, volume, target_depth, is_segmentation=False):
        """调整深度维度"""
        current_depth = volume.shape[0]
        
        if current_depth == target_depth:
            return volume
        
        # 使用插值调整深度
        if is_segmentation:
            # 分割标签使用最近邻插值
            mode = 'nearest'
        else:
            # CT影像使用线性插值
            mode = 'trilinear'
        
        # 转换为Tensor并调整维度 [D, H, W] -> [1, 1, D, H, W]
        volume_tensor = torch.from_numpy(volume.astype(np.float32))
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        
        # 调整深度
        resized = F.interpolate(
            volume_tensor, 
            size=(target_depth, self.target_size, self.target_size),
            mode=mode,
            align_corners=False if mode == 'trilinear' else None
        )
        
        return resized.squeeze().numpy()
    
    def preprocess(self, imaging_path, segmentation_path):
        """完整的预处理流程"""
        print(f"Processing: {imaging_path}")
        
        # 1. 加载数据
        imaging = nib.load(imaging_path)
        segmentation = nib.load(segmentation_path)
        
        imaging_data = imaging.get_fdata()
        segmentation_data = segmentation.get_fdata()
        
        print(f"Original - Imaging: {imaging_data.shape}, Segmentation: {segmentation_data.shape}")
        
        # 2. 检测肾脏ROI
        roi = self.detect_kidney_roi(segmentation_data)
        
        if roi:
            start, end = roi
            imaging_roi = imaging_data[start:end, :, :]
            segmentation_roi = segmentation_data[start:end, :, :]
            print(f"ROI crop: {imaging_data.shape} -> {imaging_roi.shape}")
        else:
            imaging_roi = imaging_data
            segmentation_roi = segmentation_data
            print("Using full volume (no ROI detection)")
        
        # 3. 调整深度到目标尺寸
        if imaging_roi.shape[0] != self.target_depth:
            imaging_roi = self.resize_depth(imaging_roi, self.target_depth, is_segmentation=False)
            segmentation_roi = self.resize_depth(segmentation_roi, self.target_depth, is_segmentation=True)
            print(f"Depth resize: -> {imaging_roi.shape}")
        
        # 4. CT值标准化
        imaging_roi = self.normalize_ct(imaging_roi)
        
        # 5. 转换为Tensor
        imaging_tensor = torch.from_numpy(imaging_roi.astype(np.float32))
        segmentation_tensor = torch.from_numpy(segmentation_roi.astype(np.int64))
        
        # 添加通道维度: [D, H, W] -> [1, D, H, W]
        imaging_tensor = imaging_tensor.unsqueeze(0)
        segmentation_tensor = segmentation_tensor.unsqueeze(0)
        
        print(f"Final - Imaging: {imaging_tensor.shape}, Range: [{imaging_tensor.min():.3f}, {imaging_tensor.max():.3f}]")
        print(f"Final - Segmentation: {segmentation_tensor.shape}, Labels: {torch.unique(segmentation_tensor)}")
        
        return imaging_tensor, segmentation_tensor

# 使用示例
if __name__ == "__main__":
    preprocessor = KITS23Preprocessor(target_depth=128, target_size=512)
    
    # 测试一个病例
    import glob
    cases = glob.glob("case_*/imaging.nii.gz")[:1]
    
    for case in cases:
        seg_path = case.replace("imaging.nii.gz", "segmentation.nii.gz")
        if os.path.exists(seg_path):
            image_tensor, seg_tensor = preprocessor.preprocess(case, seg_path)
            print("Preprocessing completed successfully!")
            break
