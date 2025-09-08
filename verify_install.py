# verify_install.py (最终无误版)
import torch
import numpy as np

# 修正：将 BVp 改为 BVP (all caps)
from pyVHR.BVP.methods import cpu_OMIT

print("--- 开始使用正确的 cpu_OMIT 函数验证 pyVHR 环境 ---")

# 1. 检查 PyTorch 和 CUDA
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA (GPU) 支持: { '已启用' if use_cuda else '未启用' } ({device})")

# 2. 测试核心函数的调用
try:
    print("\n>>> 核心测试：尝试调用 cpu_OMIT 函数...")
    
    # 创建虚拟RGB信号
    dummy_rgb_signal = np.random.rand(1, 3, 100).astype(np.float32)
    print(f"创建虚拟RGB信号，形状为: {dummy_rgb_signal.shape}")
    
    # 直接调用 cpu_OMIT 函数
    bvp_signal = cpu_OMIT(dummy_rgb_signal)
    print("cpu_OMIT 函数调用成功！")
    
    # 检查返回的BVP信号
    print(f"函数返回BVP信号，形状为: {bvp_signal.shape}")
    assert bvp_signal.shape == (1, 100)

    print("\n✅✅✅ 恭喜！我们成功了！pyVHR 核心功能运行正常！ ✅✅✅")
    print("环境就绪，可以开始您的研究了！")

except Exception as e:
    import traceback
    print(f"\n❌ 测试失败！遇到了一个错误：")
    traceback.print_exc()

print("\n--- 调试马拉松结束 ---")
