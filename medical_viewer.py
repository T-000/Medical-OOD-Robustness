import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from create_dataloader import KITS23Dataset
from kits23_unet_fixed import KITS23UNetFixed


class MedicalViewer:
    def __init__(self):
        self.dataset = KITS23Dataset()
        self.current_case_idx = 0
        self.current_slice = 64  # 中间切片
        self.window_center = 40   # CT窗位
        self.window_width = 400   # CT窗宽
        self.model = None
        self.model_loaded = False

        self.ai_mask = None  # 存储AI预测结果
        self.has_ai_prediction = False  # 标记是否已预测

        self.current_case_loaded = False  # 标记当前病例是否加载
        self.image = None
        self.mask = None

    def load_case(self, case_idx):
        """加载指定病例"""
        try:
            self.current_case_idx = case_idx
            self.image, self.mask = self.dataset[case_idx]
            self.image = self.image.squeeze().numpy()  # [128,512,512]
            self.mask = self.mask.squeeze().numpy()  # [128,512,512]
            self.current_case_loaded = True
            self.ai_mask = None  # 重置AI预测
            print(f"✅ 成功加载病例 {case_idx}")
            return self.get_case_info()
        except Exception as e:
            print(f"❌ 加载病例失败: {e}")
            self.current_case_loaded = False
            return None

    def apply_ct_window(self, image):
        """应用CT窗宽窗位"""
        min_val = self.window_center - self.window_width // 2
        max_val = self.window_center + self.window_width // 2
        windowed = np.clip(image, min_val, max_val)
        windowed_normalized = (windowed - min_val) / (max_val - min_val)
        return np.clip(windowed_normalized, 0, 1)  # 确保在[0,1]范围内

    def get_slice(self, slice_idx=None):
        """获取指定切片"""
        if not self.current_case_loaded:
            print("⚠️  请先加载病例")
            return None, None

        if slice_idx is None:
            slice_idx = self.current_slice

        # 确保切片在有效范围内
        slice_idx = max(0, min(slice_idx, self.image.shape[0] - 1))
        self.current_slice = slice_idx

        ct_slice = self.apply_ct_window(self.image[slice_idx])
        mask_slice = self.mask[slice_idx]

        return ct_slice, mask_slice

    def get_case_info(self):
        """获取病例信息"""
        if not self.current_case_loaded:
            return {'name': 'No case loaded', 'shape': None, 'slices': 0}

        case_name = f"case_{self.current_case_idx:05d}"
        shape = self.image.shape
        return {
            'name': case_name,
            'shape': shape,
            'slices': shape[0]
        }

    def load_model(self, model_path='models/kits23_trained_model.pth'):
        """加载训练好的模型"""
        print("🧠 加载AI分割模型...")
        try:
            self.model = KITS23UNetFixed()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            print("✅ 模型加载成功!")
            return True
        except FileNotFoundError:
            print(f"❌ 模型文件不存在: {model_path}")
            # 列出models文件夹内容帮助调试
            import os
            if os.path.exists('models'):
                print(f"📁 models文件夹内容: {os.listdir('models')}")
            return False
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def predict_case(self, image_tensor):
        """对病例进行AI预测 - 修复版本"""
        if not self.model_loaded:
            print("⚠️  请先加载模型")
            return None

        if not self.current_case_loaded:
            print("⚠️  请先加载病例")
            return None

        try:
            print(f"🔍 调试信息:")
            print(f"   📊 输入数据范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

            with torch.no_grad():
                image_tensor = image_tensor.float()
                output = self.model(image_tensor.unsqueeze(0))

                print(f"   📊 模型输出范围: [{output.min():.3f}, {output.max():.3f}]")
                print(f"   📊 模型输出形状: {output.shape}")

                # 修复：使用softmax将logits转换为概率
                probabilities = F.softmax(output, dim=1)
                print(f"   📊 Softmax后范围: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
                prediction = torch.argmax(probabilities, dim=1).squeeze(0)
                print(f"   📊 原始预测类别: {torch.unique(prediction)}")

                # 将类别3映射为背景
                prediction = torch.where(prediction == 3, torch.tensor(0), prediction)

                # 统计预测结果
                unique, counts = torch.unique(prediction, return_counts=True)
                class_distribution = dict(zip(unique.numpy(), counts.numpy()))
                print(f"   📊 预测类别分布: {class_distribution}")

                return prediction.numpy()

        except Exception as e:
            print(f"❌ AI预测失败: {e}")
            return None

    def predict_entire_case(self):
        """对整个病例进行AI预测（只调用一次）"""
        if not self.model_loaded or not self.current_case_loaded:
            return False

        if self.has_ai_prediction and self.ai_mask is not None:
            return True  # 已经预测过了

        print("🤖 运行AI分割...")
        image_tensor = torch.from_numpy(self.image).unsqueeze(0).float()
        self.ai_mask = self.predict_case(image_tensor)

        if self.ai_mask is not None:
            self.has_ai_prediction = True
            print("✅ AI预测完成")
            return True
        else:
            print("❌ AI预测失败")
            return False

    def run_ai_prediction(self):
        """显式运行AI预测 - 用户明确知道这是在执行耗时操作"""
        if not self.model_loaded:
            print("⚠️  请先加载模型")
            return False

        if not self.current_case_loaded:
            print("⚠️  请先加载病例")
            return False

        if self.has_ai_prediction:
            print("ℹ️  已经运行过AI预测")
            return True

        print("🤖 运行AI分割...")
        try:
            image_tensor = torch.from_numpy(self.image).unsqueeze(0).float()
            self.ai_mask = self.predict_case(image_tensor)

            if self.ai_mask is not None:
                self.has_ai_prediction = True
                print("✅ AI预测完成")
                return True
            else:
                print("❌ AI预测失败")
                return False
        except Exception as e:
            print(f"❌ AI预测异常: {e}")
            return False

    def get_ai_prediction(self, slice_idx=None):
        """只获取AI预测的切片，绝对不触发预测"""
        if slice_idx is None:
            slice_idx = self.current_slice

        # 简单的数据获取，没有副作用
        if not self.has_ai_prediction or self.ai_mask is None:
            return None

        if 0 <= slice_idx < self.ai_mask.shape[0]:
            return self.ai_mask[slice_idx]
        else:
            print(f"⚠️  切片索引超出范围: {slice_idx}")
            return None

    def clear_ai_prediction(self):
        """清除当前的AI预测"""
        self.ai_mask = None
        print("🧹 已清除AI预测结果")
