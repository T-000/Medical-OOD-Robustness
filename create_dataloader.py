#!/usr/bin/env python3
import torch
import glob
from torch.utils.data import Dataset, DataLoader

class KITS23Dataset(Dataset):
    def __init__(self, data_dir="preprocessed_data"):
        self.files = glob.glob(f"{data_dir}/*.pt")
        print(f"📁 加载 {len(self.files)} 个预处理病例")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data['image'], data['segmentation']

# 测试
if __name__ == "__main__":
    dataset = KITS23Dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("🚀 DataLoader 测试:")
    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i}: images {images.shape}, masks {masks.shape}")
        if i == 1: break
    print("✅ DataLoader 准备就绪!")
