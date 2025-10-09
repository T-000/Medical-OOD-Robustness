import torch
import matplotlib.pyplot as plt
import numpy as np
from medical_viewer import MedicalViewer


class BasicViewer:
    def __init__(self):
        self.viewer = MedicalViewer()
        self.fig = None
        self.current_case_loaded = None  # 跟踪当前加载的病例

        # 加载模型
        if not self.viewer.load_model():
            print("⚠️  模型加载失败，将继续使用基础功能")

    def show_slice(self, case_idx=0, slice_idx=None, run_ai=False):
        """显示切片，包括AI预测
        Args:
            run_ai: 是否运行AI预测，默认False避免意外触发
        """
        # 只有在切换病例时才重新加载
        if self.current_case_loaded != case_idx:
            case_info = self.viewer.load_case(case_idx)
            self.current_case_loaded = case_idx
            print(f"📊 加载病例: {case_info['name']}")
        else:
            case_info = self.viewer.get_case_info()

        # 获取切片数据
        ct_slice, mask_slice = self.viewer.get_slice(slice_idx)

        # AI预测逻辑 - 修复版本
        ai_slice = None
        if self.viewer.model_loaded:
            if run_ai:
                # 显式运行AI预测
                if self.viewer.run_ai_prediction():
                    ai_slice = self.viewer.get_ai_prediction(slice_idx)
                else:
                    print("⚠️  AI预测运行失败")
            else:
                # 只获取已有的AI预测结果
                ai_slice = self.viewer.get_ai_prediction(slice_idx)
                if ai_slice is None:
                    print("💡 提示: 使用 show_slice(run_ai=True) 运行AI预测")

        # 创建可视化
        self._create_display(ct_slice, mask_slice, ai_slice, case_info)

    def _create_display(self, ct_slice, mask_slice, ai_slice, case_info):
        """创建显示界面"""
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # CT扫描
        axes[0, 0].imshow(ct_slice, cmap='gray', aspect='auto')
        axes[0, 0].set_title(f'CT Scan - Slice {self.viewer.current_slice}')
        axes[0, 0].axis('off')

        # 医生标注 (带颜色)
        axes[0, 1].imshow(ct_slice, cmap='gray', aspect='auto')
        if mask_slice.max() > 0:
            colored_mask = self._colorize_mask(mask_slice)
            axes[0, 1].imshow(colored_mask, alpha=0.7, aspect='auto')
        axes[0, 1].set_title('Expert Annotations')
        axes[0, 1].axis('off')

        # AI预测
        if ai_slice is not None:
            axes[1, 0].imshow(ct_slice, cmap='gray', aspect='auto')
            colored_ai = self._colorize_mask(ai_slice)
            axes[1, 0].imshow(colored_ai, alpha=0.7, aspect='auto')
            axes[1, 0].set_title('AI Prediction')
            axes[1, 0].axis('off')

            # 对比显示
            axes[1, 1].imshow(ct_slice, cmap='gray', aspect='auto')
            axes[1, 1].imshow(colored_ai, alpha=0.5, aspect='auto')
            axes[1, 1].set_title('AI Overlay')
            axes[1, 1].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'AI Prediction\nNot Available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def _colorize_mask(self, mask):
        """将分割掩码转换为彩色"""
        # 0:背景, 1:肾脏, 2:肿瘤, 3:囊肿
        colors = np.array([[0, 0, 0, 0],  # 透明
                           [0, 1, 0, 0.7],  # 绿色-肾脏
                           [1, 0, 0, 0.7],  # 红色-肿瘤
                           [0, 0, 1, 0.7]])  # 蓝色-囊肿
        return colors[mask]

    def list_cases(self):
        """列出所有可用病例（不重复加载）"""
        total_cases = len(self.viewer.dataset)
        print(f"📁 Available cases: {total_cases}")

        # 只显示信息，不实际加载数据
        for i in range(min(5, total_cases)):
            print(f"  {i}: case_{i:05d}")

    def compare_annotations(self, case_idx=0, slice_idx=64):
        """对比医生标注和AI预测"""
        # 加载病例
        if self.current_case_loaded != case_idx:
            self.viewer.load_case(case_idx)
            self.current_case_loaded = case_idx

        # 运行AI预测
        print("🤖 运行AI预测并对比...")
        self.viewer.run_ai_prediction()

        # 获取数据
        ct_slice, doctor_mask = self.viewer.get_slice(slice_idx)
        ai_mask = self.viewer.get_ai_prediction(slice_idx)

        # 打印统计信息
        doctor_stats = dict(zip(*np.unique(doctor_mask, return_counts=True)))
        ai_stats = dict(zip(*np.unique(ai_mask, return_counts=True))) if ai_mask is not None else {}

        print(f"👨‍⚕️ 医生标注 - 类别分布: {doctor_stats}")
        print(f"🤖 AI预测 - 类别分布: {ai_stats}")

        # 可视化对比
        self._create_comparison_display(ct_slice, doctor_mask, ai_mask)

    def _create_comparison_display(self, ct_slice, doctor_mask, ai_mask):
        """创建详细对比显示"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：单独显示
        axes[0, 0].imshow(ct_slice, cmap='gray')
        axes[0, 0].set_title('CT Scan')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(doctor_mask, cmap='tab10')
        axes[0, 1].set_title('Doctor Annotation')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(ai_mask if ai_mask is not None else np.zeros_like(ct_slice),
                          cmap='tab10')
        axes[0, 2].set_title('AI Prediction')
        axes[0, 2].axis('off')

        # 第二行：叠加显示和差异
        axes[1, 0].imshow(ct_slice, cmap='gray')
        axes[1, 0].imshow(doctor_mask, cmap='tab10', alpha=0.5)
        axes[1, 0].set_title('Doctor Overlay')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(ct_slice, cmap='gray')
        if ai_mask is not None:
            axes[1, 1].imshow(ai_mask, cmap='tab10', alpha=0.5)
        axes[1, 1].set_title('AI Overlay')
        axes[1, 1].axis('off')

        # 差异图
        if ai_mask is not None:
            difference = doctor_mask - ai_mask
            axes[1, 2].imshow(difference, cmap='coolwarm', vmin=-2, vmax=2)
            axes[1, 2].set_title('Difference\n(Red=AI多, Blue=医生多)')
        else:
            axes[1, 2].text(0.5, 0.5, 'No AI Prediction',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    viewer = BasicViewer()
    viewer.compare_annotations(case_idx=0, slice_idx=64)
    viewer.list_cases()
    viewer.show_slice(case_idx=0, slice_idx=64)