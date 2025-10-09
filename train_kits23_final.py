#!/usr/bin/env python3
"""
KITS23 专用训练脚本 - 使用修复后的UNet
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from create_dataloader import KITS23Dataset
from kits23_unet_fixed import KITS23UNetFixed

def main():
    print("🚀 启动 KITS23 训练 (修复版UNet)...")
    print("=" * 50)
    
    # 1. 数据加载
    dataset = KITS23Dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"📊 训练病例: {len(dataset)} 个")
    
    # 2. 模型和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KITS23UNetFixed().to(device)
    
    # 3. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🎯 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📱 使用设备: {device}")
    
    # 4. 训练循环
    model.train()
    
    for epoch in range(3):  # 先训练3个epoch
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device).squeeze(1)  # [B, 128, 512, 512]
            
            # 前向传播
            outputs = model(images)  # [B, 4, 128, 512, 512]
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"🎯 Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
        print("-" * 40)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "kits23_trained_model.pth")
    
    print("💾 模型已保存: kits23_trained_model.pth")
    print("✅ 训练完成!")

if __name__ == "__main__":
    main()
