#!/usr/bin/env python3
"""
批量处理所有 KITS23 数据
"""

import os
import glob
import torch
from run_preprocessor_final import KITS23Preprocessor

def preprocess_all():
    """处理所有病例并保存"""
    
    print("🔄 开始批量处理所有 KITS23 数据...")
    
    # 初始化预处理器
    preprocessor = KITS23Preprocessor(target_depth=128)
    
    # 查找所有病例
    dataset_path = "dataset"
    cases = glob.glob(f"{dataset_path}/case_*/imaging.nii.gz", recursive=True)
    
    print(f"找到 {len(cases)} 个病例")
    
    # 创建输出目录
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    error_cases = []
    
    for i, case_path in enumerate(cases):
        seg_path = case_path.replace("imaging.nii.gz", "segmentation.nii.gz")
        case_name = os.path.basename(os.path.dirname(case_path))
        
        print(f"\n[{i+1}/{len(cases)}] 处理 {case_name}...")
        
        if os.path.exists(seg_path):
            try:
                # 预处理
                image_tensor, seg_tensor = preprocessor.preprocess(case_path, seg_path)
                
                # 保存处理后的数据
                torch.save({
                    'image': image_tensor,
                    'segmentation': seg_tensor,
                    'case_name': case_name,
                    'original_shape': f"{image_tensor.shape}"
                }, f"{output_dir}/{case_name}.pt")
                
                success_count += 1
                print(f"✅ 已保存: {output_dir}/{case_name}.pt")
                
            except Exception as e:
                error_msg = f"{case_name}: {e}"
                print(f"❌ 处理失败: {error_msg}")
                error_cases.append(error_msg)
        else:
            error_msg = f"{case_name}: 分割文件不存在"
            print(f"❌ {error_msg}")
            error_cases.append(error_msg)
    
    # 输出总结
    print(f"\n{'='*50}")
    print(f"🎉 批量处理完成!")
    print(f"✅ 成功: {success_count}/{len(cases)} 个病例")
    print(f"❌ 失败: {len(error_cases)} 个病例")
    print(f"📁 数据保存在: {output_dir}/")
    
    if error_cases:
        print(f"\n失败病例:")
        for error in error_cases[:10]:  # 只显示前10个错误
            print(f"  - {error}")
    
    return success_count, error_cases

if __name__ == "__main__":
    preprocess_all()
