# test_basic.py
from basic_viewer import BasicViewer


def test_basic_functionality():
    print("🧪 测试基础功能...")

    # 1. 创建查看器
    viewer = BasicViewer()

    # 2. 列出可用病例
    viewer.list_cases()

    # 3. 测试显示切片（不运行AI）
    print("\n📊 测试基础显示...")
    viewer.show_slice(case_idx=0, slice_idx=64)  # 不运行AI

    # 4. 测试不同切片
    print("\n📊 测试不同切片...")
    viewer.show_slice(case_idx=0, slice_idx=80)


if __name__ == "__main__":
    test_basic_functionality()