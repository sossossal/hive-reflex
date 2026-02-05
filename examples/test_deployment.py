#!/usr/bin/env python3
"""
模型部署工具链测试示例
演示完整的 PyTorch → Flash 固件流程

@file test_deployment.py
"""

import torch
import torch.nn as nn
import sys
import os

# 添加工具路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from model_to_flash import model_to_flash


# ============================================================================
# 创建测试模型
# ============================================================================

class TinyGestureNet(nn.Module):
    """简单的手势识别 MLP"""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def create_test_model():
    """创建并保存测试模型"""
    print("=" * 60)
    print("创建测试模型...")
    print("=" * 60)
    
    model = TinyGestureNet()
    
    # 随机初始化权重
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.randn_like(param) * 0.1
    
    # 保存为 PyTorch 格式
    model_path = 'test_gesture_model.pth'
    torch.save(model, model_path)
    print(f"✅ 模型已保存: {model_path}")
    
    # 导出为 ONNX
    onnx_path = 'test_gesture_model.onnx'
    dummy_input = torch.randn(1, 8)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ ONNX 已导出: {onnx_path}")
    
    # 统计模型参数
    num_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * 4 for p in model.parameters())  # FP32
    
    print(f"\n模型统计:")
    print(f"  参数量: {num_params}")
    print(f"  FP32 大小: {model_size} 字节 ({model_size/1024:.1f} KB)")
    print(f"  预计 INT8 大小: {num_params} 字节 ({num_params/1024:.1f} KB)")
    
    return model_path, onnx_path


# ============================================================================
# 测试部署流程
# ============================================================================

def test_pytorch_deployment():
    """测试 PyTorch → Flash 流程"""
    print("\n" + "=" * 60)
    print("测试 1: PyTorch 模型部署")
    print("=" * 60)
    
    model_path, _ = create_test_model()
    output_path = 'test_firmware_pytorch.bin'
    
    metadata = {
        'model_name': 'TinyGestureNet',
        'framework': 'PyTorch',
        'input_shape': [8],
        'output_shape': [2],
    }
    
    try:
        model_to_flash(model_path, output_path, auto_slice=False, metadata=metadata)
        print(f"\n✅ 测试通过! 固件已生成: {output_path}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_onnx_deployment():
    """测试 ONNX → Flash 流程"""
    print("\n" + "=" * 60)
    print("测试 2: ONNX 模型部署")
    print("=" * 60)
    
    _, onnx_path = create_test_model()
    output_path = 'test_firmware_onnx.bin'
    
    metadata = {
        'model_name': 'TinyGestureNet',
        'framework': 'ONNX',
        'quantization': 'INT8',
    }
    
    try:
        model_to_flash(onnx_path, output_path, auto_slice=False, metadata=metadata)
        print(f"\n✅ 测试通过! 固件已生成: {output_path}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_large_model_slicing():
    """测试大模型自动切片"""
    print("\n" + "=" * 60)
    print("测试 3: 大模型自动切片")
    print("=" * 60)
    
    # 创建一个较大的模型（接近 512 KB）
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 512)  # ~1 MB
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = LargeModel()
    large_model_path = 'test_large_model.pth'
    torch.save(model, large_model_path)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"大模型参数量: {num_params} (~{num_params/1024:.0f} KB INT8)")
    
    output_path = 'test_firmware_sliced.bin'
    
    try:
        model_to_flash(large_model_path, output_path, auto_slice=True, metadata={'model_name': 'LargeNet'})
        print(f"\n✅ 测试通过! 切片固件已生成: {output_path}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def inspect_firmware(firmware_path: str):
    """检查固件文件内容"""
    print("\n" + "=" * 60)
    print(f"检查固件: {firmware_path}")
    print("=" * 60)
    
    import struct
    
    with open(firmware_path, 'rb') as f:
        # 读取头部
        magic = f.read(4)
        version = struct.unpack('<H', f.read(2))[0]
        num_slices = struct.unpack('<H', f.read(2))[0]
        total_size = struct.unpack('<I', f.read(4))[0]
        metadata_len = struct.unpack('<H', f.read(2))[0]
        reserved = struct.unpack('<H', f.read(2))[0]
        
        # 读取元数据
        metadata_json = f.read(metadata_len).decode('utf-8')
        
        print(f"固件头部:")
        print(f"  Magic: {magic} ({'✅ 正确' if magic == b'HRF2' else '❌ 错误'})")
        print(f"  Version: {version >> 8}.{(version >> 4) & 0xF}.0")
        print(f"  切片数: {num_slices}")
        print(f"  总大小: {total_size} 字节 ({total_size/1024:.1f} KB)")
        print(f"\n元数据:")
        print(f"  {metadata_json}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  Hive-Reflex 模型部署工具链 - 集成测试              ║")
    print("╚" + "=" * 58 + "╝")
    
    # 测试 1: PyTorch
    test_pytorch_deployment()
    
    # 测试 2: ONNX
    test_onnx_deployment()
    
    # 测试 3: 大模型切片
    test_large_model_slicing()
    
    # 检查生成的固件
    print("\n")
    inspect_firmware('test_firmware_pytorch.bin')
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - test_gesture_model.pth")
    print("  - test_gesture_model.onnx")
    print("  - test_large_model.pth")
    print("  - test_firmware_pytorch.bin")
    print("  - test_firmware_onnx.bin")
    print("  - test_firmware_sliced.bin")
    print("\n下一步: 将 .bin 文件烧录到 Flash 并在 ZCU102 上测试")


if __name__ == '__main__':
    main()
