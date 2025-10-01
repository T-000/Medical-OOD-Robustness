#!/usr/bin/env python3
"""
安全清理不完整的病例数据
"""
import os
import glob
import shutil

def safe_cleanup():
    print("🧹 清理不完整病例数据")
    print("=" * 40)
    
    # 统计清理前后对比
    all_cases = glob.glob("dataset/case_*")
    complete_cases = []
    incomplete_cases = []
    
    for case_dir in all_cases:
        has_imaging = os.path.exists(os.path.join(case_dir, "imaging.nii.gz"))
        has_segmentation = os.path.exists(os.path.join(case_dir, "segmentation.nii.gz"))
        
        if has_imaging and has_segmentation:
            complete_cases.append(case_dir)
        else:
            incomplete_cases.append(case_dir)
    
    print(f"📊 清理前统计:")
    print(f"   总目录数: {len(all_cases)}")
    print(f"   完整病例: {len(complete_cases)}")
    print(f"   不完整病例: {len(incomplete_cases)}")
    
    if not incomplete_cases:
        print("✅ 没有需要清理的不完整病例")
        return
    
    # 显示将要删除的病例
    print(f"\n🗑️  将要删除的不完整病例 (前10个):")
    for case in incomplete_cases[:10]:
        case_name = os.path.basename(case)
        print(f"   - {case_name}")
    
    if len(incomplete_cases) > 10:
        print(f"   ... 还有 {len(incomplete_cases)-10} 个")
    
    # 确认清理
    confirm = input(f"\n⚠️  确认删除 {len(incomplete_cases)} 个不完整病例？ (y/n): ")
    
    if confirm.lower() == 'y':
        deleted_count = 0
        for case_dir in incomplete_cases:
            try:
                shutil.rmtree(case_dir)
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {os.path.basename(case_dir)}: {e}")
        
        print(f"✅ 成功删除 {deleted_count} 个不完整病例")
        
        # 验证清理结果
        remaining_cases = glob.glob("dataset/case_*")
        print(f"📊 清理后统计:")
        print(f"   剩余病例: {len(remaining_cases)}")
        print(f"   全部为完整病例 ✅")
        
    else:
        print("🔒 取消清理操作")

if __name__ == "__main__":
    safe_cleanup()
