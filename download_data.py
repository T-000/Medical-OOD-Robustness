#!/usr/bin/env python3
"""
KiTS23 数据下载和准备脚本
"""
import os
import tarfile
import urllib.request
import time

def main():
    print("🎯 开始 KiTS23 数据准备")
    print("=" * 50)
    
    # 创建数据目录
    data_dir = "data/kits23"
    os.makedirs(data_dir, exist_ok=True)
    
    print("📁 数据将下载到:", os.path.abspath(data_dir))
    print("\n📚 请从 KiTS23 GitHub 仓库下载数据:")
    print("1. 访问你找到的 KiTS23 GitHub 页面")
    print("2. 按照仓库的说明下载数据")
    print("3. 将数据放到 data/kits23/ 目录")
    print("\n💡 确保数据目录结构符合预期")
    print("🕐 下载期间我们可以先优化数据处理代码")

if __name__ == "__main__":
    main()
