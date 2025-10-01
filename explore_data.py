#!/usr/bin/env python3
"""
KiTS19 Data Exploration Script
第一步：验证环境和数据读取
"""
import sys
import numpy as np

def check_environment():
    """检查必要的包是否已安装"""
    required_packages = ['nibabel', 'monai', 'torch', 'numpy']
    
    print("🔍 检查环境...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError as e:
            print(f"❌ {package} 未安装: {e}")
            return False
    return True

def main():
    print("🎯 医疗影像模型稳健性研究 - 环境验证")
    print("=" * 50)
    
    # 检查Python和环境
    print(f"Python路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # 检查环境
    if not check_environment():
        print("\n⚠️ 请先安装必要的包")
        return
    
    print("\n✅ 环境检查完成！")
    print("📚 下一步：下载 KiTS19 数据")
    print("🌐 访问: https://kits19.grand-challenge.org/")

if __name__ == "__main__":
    main()
