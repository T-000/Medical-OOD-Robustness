#!/usr/bin/env python3
"""
修复的 KITS23 UNet - 立即下采样输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class KITS23UNetFixed(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        
        print("🧠 初始化修复的 KITS23 UNet...")
        
        # 关键修复：在第一个卷积前立即下采样
        self.initial_downsample = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1),  # 立即减半尺寸
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        # 剩下的UNet在 manageable 的尺寸上运行
        self.enc1 = self._block(16, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        
        self.dec1 = self._block(128 + 64, 64)
        self.dec2 = self._block(64 + 32, 32)
        self.dec3 = self._block(32 + 16, 16)
        
        self.final_upsample = nn.ConvTranspose3d(16, out_channels, 2, stride=2)
        
        self.pool = nn.MaxPool3d(2)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        print(f"📥 原始输入: {x.shape}")
        
        # 关键修复：立即处理大尺寸输入
        x = self.initial_downsample(x)  # [B, 16, 64, 256, 256]
        print(f"📤 下采样后: {x.shape}")
        
        # Encoder
        e1 = self.enc1(x)              # [B, 32, 64, 256, 256]
        e2 = self.enc2(self.pool(e1))  # [B, 64, 32, 128, 128]
        e3 = self.enc3(self.pool(e2))  # [B, 128, 16, 64, 64]
        
        # Decoder
        d1 = F.interpolate(e3, size=e2.shape[2:], mode='trilinear')
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)             # [B, 64, 32, 128, 128]
        
        d2 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear')
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)             # [B, 32, 64, 256, 256]
        
        d3 = F.interpolate(d2, size=x.shape[2:], mode='trilinear')
        d3 = torch.cat([d3, x], dim=1)
        d3 = self.dec3(d3)             # [B, 16, 64, 256, 256]
        
        # 上采样回原始尺寸
        output = self.final_upsample(d3)  # [B, 4, 128, 512, 512]
        print(f"📦 最终输出: {output.shape}")
        
        return output

def test_fixed_model():
    print("🔍 测试修复的模型...")
    
    model = KITS23UNetFixed()
    dummy_input = torch.randn(2, 1, 128, 512, 512)
    
    print(f"输入尺寸: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ 成功! 输出尺寸: {output.shape}")
    print(f"📊 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    test_fixed_model()
