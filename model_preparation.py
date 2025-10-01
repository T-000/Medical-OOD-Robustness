#!/usr/bin/env python3
"""
在数据下载期间准备3D分割模型
"""
import torch
import torch.nn as nn
import monai

class Simple3DUNet(nn.Module):
    """简单的3D UNet分割模型"""
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        
        print("🧠 初始化3D UNet模型...")
        
        # 使用MONAI内置的UNet（等数据下载后可以训练）
        self.model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        
    def forward(self, x):
        return self.model(x)

def test_model_setup():
    """测试模型配置"""
    print("🎯 模型准备检查")
    print("=" * 40)
    
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")
    
    # 测试模型
    model = Simple3DUNet()
    
    # 模拟输入（等真实数据下载后替换）
    batch_size, depth, height, width = 2, 128, 128, 128
    dummy_input = torch.randn(batch_size, 1, depth, height, width)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ 模型测试通过!")
    print(f"   输入维度: {dummy_input.shape}")
    print(f"   输出维度: {output.shape}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    test_model_setup()
    print("\n📥 数据下载完成后，我们就可以开始训练!")
