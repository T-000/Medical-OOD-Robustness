#!/usr/bin/env python3
import time
import sys

print("开始时间:", time.time())

start = time.time()
print("🔍 检查Python导入...")
import numpy as np
print(f"numpy 导入耗时: {time.time() - start:.2f}秒")

start = time.time()
import nibabel
print(f"nibabel 导入耗时: {time.time() - start:.2f}秒")

start = time.time()  
import monai
print(f"monai 导入耗时: {time.time() - start:.2f}秒")

start = time.time()
import torch
print(f"torch 导入耗时: {time.time() - start:.2f}秒")

print("总运行时间:", time.time() - start, "秒")
