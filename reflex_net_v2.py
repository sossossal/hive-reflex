"""
ReflexNet 升级版 - 支持 MLIR 编译器导出
整合量化感知训练 (QAT) 和 CIM 优化
"""

import torch
import torch.nn as nn
import argparse

class ReflexNetV2(nn.Module):
    """
    ReflexNet 2.0 - 优化用于 Digital CIM 架构
    
    改进:
    - 量化友好的层设计
    - 适配 CIM 矩阵维度
    - 支持 MLIR 导出
    """
    def __init__(self, quantize=False):
        super(ReflexNetV2, self).__init__()
        
        # 输入维度: 12 (6 IMU + 3 Hist + 2 Current + 1 TargetDiff)
        # 优化为 CIM 友好的维度 (32的倍数)
        
        self.fc1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        
        # LSTM 单元
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        
        self.fc2 = nn.Linear(16, 1)
        self.tanh = nn.Tanh()
        
        # 量化配置
        if quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant = None
            self.dequant = None

    def forward(self, x, hidden_state=None):
        # x shape: (batch, seq_len, 12)
        
        if self.quant:
            x = self.quant(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        # LSTM 处理
        if hidden_state is None:
            x, new_hidden = self.lstm(x)
        else:
            x, new_hidden = self.lstm(x, hidden_state)
        
        # 取最后一个时间步
        last_step = x[:, -1, :]
        
        out = self.fc2(last_step)
        out = self.tanh(out)
        
        if self.dequant:
            out = self.dequant(out)
        
        return out, new_hidden


def export_to_onnx(model_path='reflex_net_v2.onnx', quantize=False):
    """导出为 ONNX 格式"""
    print(f"🔨 导出 ReflexNet V2 → ONNX")
    
    model = ReflexNetV2(quantize=quantize)
    model.eval()
    
    # 统计参数
    param_count = sum(p.numel() for p in model.parameters())
    param_size_kb = param_count * 4 / 1024  # FP32
    
    print(f"  参数量: {param_count} ({param_size_kb:.2f} KB)")
    
    # 准备示例输入
    dummy_input = torch.randn(1, 5, 12)  # Batch=1, Seq=5, Feat=12
    h0 = torch.zeros(1, 1, 16)
    c0 = torch.zeros(1, 1, 16)
    
    # 量化 (如果需要)
    if quantize:
        print("  应用动态量化 (INT8)...")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        print(f"  量化后模型大小约为 {param_size_kb / 4:.2f} KB")
    
    # 导出
    print(f"  导出文件: {model_path}")
    torch.onnx.export(
        model,
        (dummy_input, (h0, c0)),
        model_path,
        input_names=['input', 'h_in', 'c_in'],
        output_names=['output', 'h_out', 'c_out'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch', 1: 'seq_len'}}
    )
    
    print("  ✓ ONNX 导出完成")
    return model_path


def export_to_mlir(model, output_path='reflex_net_v2.mlir'):
    """
    导出为 MLIR 格式 (需要 torch-mlir)
    这是未来的目标，当前使用 ONNX 作为中间格式
    """
    try:
        import torch_mlir
        
        print(f"🔨 导出 ReflexNet V2 → MLIR")
        
        # 创建示例输入
        example_input = torch.randn(1, 5, 12)
        
        # 编译为 MLIR
        mlir_module = torch_mlir.compile(
            model,
            example_input,
            output_type="torch"
        )
        
        # 保存 MLIR IR
        with open(output_path, 'w') as f:
            f.write(str(mlir_module))
        
        print(f"  ✓ MLIR 导出完成: {output_path}")
        
    except ImportError:
        print("  ⚠️  torch-mlir 未安装，使用 ONNX 作为替代")
        return export_to_onnx()


def compile_for_cim(onnx_path):
    """
    使用 MLIR 编译器编译为 CIM 目标代码
    这会调用 mlir_compiler/compile.py
    """
    import subprocess
    import os
    
    print("\n🚀 使用 MLIR 编译器编译...")
    
    compiler_script = os.path.join(
        os.path.dirname(__file__),
        'mlir_compiler',
        'compile.py'
    )
    
    cmd = [
        'python3',
        compiler_script,
        '--model', onnx_path,
        '--output-c', 'build/reflex_inference.c',
        '--output-weights', 'build/reflex_weights.bin'
    ]
    
    subprocess.run(cmd, check=True)
    print("  ✓ CIM 编译完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReflexNet V2 导出工具')
    parser.add_argument('--quantize', action='store_true', help='导出量化模型')
    parser.add_argument('--mlir', action='store_true', help='导出 MLIR 格式')
    parser.add_argument('--compile-cim', action='store_true', help='编译为 CIM 代码')
    
    args = parser.parse_args()
    
    # 导出模型
    if args.mlir:
        model = ReflexNetV2(quantize=args.quantize)
        export_to_mlir(model)
    else:
        onnx_path = export_to_onnx(quantize=args.quantize)
        
        # 如果需要，编译为 CIM 代码
        if args.compile_cim:
            compile_for_cim(onnx_path)
    
    print("\n✅ 完成!")
    if args.compile_cim:
        print("\n下一步:")
        print("  make APP_SRCS='examples/example_reflex_node.c build/reflex_inference.c'")
