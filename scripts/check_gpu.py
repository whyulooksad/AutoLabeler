#!/usr/bin/env python3
"""
GPU环境检查脚本
在部署前运行此脚本检查GPU和PyTorch环境
"""

import torch
import sys

def check_gpu_environment():
    print("=== GPU环境检查 ===")
    
    # 检查CUDA是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu.name}")
            print(f"  显存: {gpu.total_memory / 1024**3:.1f} GB")
            print(f"  计算能力: {gpu.major}.{gpu.minor}")
        
        # 测试GPU内存分配
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print("✓ GPU内存分配测试成功")
            print(f"当前GPU: {torch.cuda.current_device()}")
            print(f"已用显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU内存分配测试失败: {e}")
    else:
        print("⚠️  CUDA不可用，将使用CPU模式")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")

if __name__ == "__main__":
    check_gpu_environment()

