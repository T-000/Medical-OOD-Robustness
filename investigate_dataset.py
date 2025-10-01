#!/usr/bin/env python3
"""
深入调查数据集真实状态
"""
import os
import glob
from collections import defaultdict

def analyze_dataset_completeness():
    print("🔍 深入数据集分析")
    print("=" * 50)
    
    all_cases = glob.glob("dataset/case_*")
    all_cases.sort()
    
    # 分类统计
    case_status = defaultdict(list)
    
    for case_dir in all_cases:
        case_name = os.path.basename(case_dir)
        has_imaging = os.path.exists(os.path.join(case_dir, "imaging.nii.gz"))
        has_segmentation = os.path.exists(os.path.join(case_dir, "segmentation.nii.gz"))
        
        # 检查目录内容
        contents = os.listdir(case_dir)
        
        if has_imaging and has_segmentation:
            case_status["complete"].append(case_name)
        elif has_imaging and not has_segmentation:
            case_status["imaging_only"].append(case_name)
        elif not has_imaging and has_segmentation:
            case_status["segmentation_only"].append(case_name)
        else:
            case_status["other_files"].append((case_name, contents))
    
    # 输出统计
    print("📊 病例状态统计:")
    for status, cases in case_status.items():
        print(f"   {status}: {len(cases)} 个病例")
    
    # 显示详细信息
    if case_status["complete"]:
        print(f"\n✅ 完整病例 (有影像和标注): {len(case_status['complete'])}")
        complete_ids = [int(c.replace("case_", "")) for c in case_status["complete"]]
        print(f"   ID范围: {min(complete_ids)} - {max(complete_ids)}")
    
    if case_status["imaging_only"]:
        print(f"\n⚠️  只有影像的病例: {len(case_status['imaging_only'])}")
        print(f"   前5个: {case_status['imaging_only'][:5]}")
    
    if case_status["segmentation_only"]:
        print(f"\n❌ 只有标注的病例: {len(case_status['segmentation_only'])}")
    
    if case_status["other_files"]:
        print(f"\n📁 其他文件情况的病例: {len(case_status['other_files'])}")
        for case_name, files in case_status["other_files"][:3]:
            print(f"   {case_name}: {files}")
    
    return case_status

if __name__ == "__main__":
    status = analyze_dataset_completeness()
    
    complete_cases = len(status["complete"])
    print(f"\n🎯 研究可用性评估:")
    print(f"   完全可用的病例: {complete_cases}")
    
    if complete_cases >= 100:
        print("🎉 优秀！远超最优数据量要求")
    elif complete_cases >= 80:
        print("✅ 很好！达到最优数据量")
    elif complete_cases >= 50:
        print("👍 足够！可以开始核心研究")
    elif complete_cases >= 30:
        print("🔧 可行！可以开始方法开发")
    else:
        print("❌ 需要更多数据")
