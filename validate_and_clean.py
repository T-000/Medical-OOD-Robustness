#!/usr/bin/env python3
"""
自动验证预处理数据，删除损坏文件
"""

import torch
import glob
import os

def validate_and_clean():
    print("🔍 验证并清理预处理数据...")
    files = glob.glob("preprocessed_data/*.pt")
    valid_files = []
    corrupted_files = []
    
    print(f"📊 总共找到 {len(files)} 个预处理文件")
    
    for file_path in files:
        try:
            # 尝试加载文件
            data = torch.load(file_path)
            
            # 检查必要字段
            required_fields = ['image', 'segmentation', 'case_name']
            if all(field in data for field in required_fields):
                # 检查数据形状
                if (data['image'].shape == torch.Size([1, 128, 512, 512]) and 
                    data['segmentation'].shape == torch.Size([1, 128, 512, 512])):
                    valid_files.append(file_path)
                    print(f"✅ {data['case_name']}: 验证通过")
                else:
                    raise ValueError(f"形状不正确: {data['image'].shape}, {data['segmentation'].shape}")
            else:
                raise ValueError("缺少必要字段")
                
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: 损坏 - {e}")
            corrupted_files.append(file_path)
    
    # 删除损坏文件
    if corrupted_files:
        print(f"\n🗑️  删除 {len(corrupted_files)} 个损坏文件:")
        for corrupted_file in corrupted_files:
            try:
                os.remove(corrupted_file)
                print(f"   已删除: {os.path.basename(corrupted_file)}")
            except Exception as e:
                print(f"   删除失败 {os.path.basename(corrupted_file)}: {e}")
    else:
        print("\n🎉 没有发现损坏文件!")
    
    print(f"\n📊 最终统计:")
    print(f"  有效文件: {len(valid_files)} 个")
    print(f"  删除文件: {len(corrupted_files)} 个")
    print(f"  剩余病例: {len(valid_files)} 个")
    
    return valid_files

if __name__ == "__main__":
    valid_files = validate_and_clean()
    
    # 如果有效文件太少，警告用户
    if len(valid_files) < 10:
        print(f"\n⚠️  警告: 只有 {len(valid_files)} 个有效文件，可能不足以训练")
    else:
        print(f"\n🎉 有 {len(valid_files)} 个高质量病例可用于训练!")
