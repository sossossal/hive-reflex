import torch
import torch.nn as nn

class ReflexNet(nn.Module):
    def __init__(self):
        super(ReflexNet, self).__init__()
        
        # 极简模型设计，适配 IMC-22 (512KB SRAM)
        # Input Size: 12 (6 IMU + 3 Hist + 2 Current + 1 TargetDiff)
        
        self.fc1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        
        # LSTM 单元用于处理惯性 (Inertia) 和延迟
        # 隐藏层状态 h_n, c_n 需要在 RISC-V 内存中持久化
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        
        self.fc2 = nn.Linear(16, 1) # 输出单一的力矩修正值
        self.tanh = nn.Tanh()       # 输出范围归一化到 [-1, 1]

    def forward(self, x, hidden_state):
        # x shape: (batch, seq_len, 12)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        # LSTM 处理
        x, new_hidden = self.lstm(x, hidden_state)
        
        # 取最后一个时间步
        last_step = x[:, -1, :] 
        
        out = self.fc2(last_step)
        out = self.tanh(out) # * Max_Torque
        
        return out, new_hidden

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())

def export_to_onnx(use_quantization=False):
    """导出模型为 ONNX 格式，可选量化"""
    model = ReflexNet()
    model.eval()
    
    # 模型统计
    param_count = count_parameters(model)
    param_size_kb = param_count * 4 / 1024  # float32 大小
    
    print(f"ReflexNet 参数量: {param_count} ({param_size_kb:.2f} KB)")
    
    dummy_input = torch.randn(1, 5, 12) # Batch=1, Seq=5, Feat=12
    h0 = torch.zeros(1, 1, 16)
    c0 = torch.zeros(1, 1, 16)
    
    if use_quantization:
        print("应用动态量化 (INT8)...")
        model_int8 = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        model = model_int8
        output_file = "reflex_net_int8.onnx"
    else:
        output_file = "reflex_net.onnx"
    
    print(f"导出 ONNX 模型: {output_file}")
    
    # 注意：导出 LSTM 到 ONNX 需要注意 Opset 版本
    torch.onnx.export(model, (dummy_input, (h0, c0)), output_file,
                      input_names=['input', 'h_in', 'c_in'],
                      output_names=['output', 'h_out', 'c_out'],
                      opset_version=11,
                      dynamic_axes={'input': {0: 'batch', 1: 'seq_len'}})
    
    print(f"✓ 导出完成")
    if use_quantization:
        print(f"  量化后模型大小约为原来的 1/4")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true', help='导出 INT8 量化模型')
    args = parser.parse_args()
    
    export_to_onnx(use_quantization=args.quantize)
    
    if args.quantize:
        print("\n提示: 使用 --quantize 导出的模型需要在训练时进行量化感知训练 (QAT)")
        print("      以获得最佳精度。详见 train_reflex_net.py")
