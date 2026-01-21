#!/usr/bin/env python3
"""
TinyML 自适应模型训练流程
收集传感器数据、训练模型、量化导出

使用方法:
    python train_adaptive_model.py --collect      # 收集数据
    python train_adaptive_model.py --train        # 训练模型
    python train_adaptive_model.py --export       # 导出模型
    python train_adaptive_model.py --all          # 完整流程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import struct

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    'data_dir': './training_data',
    'model_dir': './models',
    'feature_dim': 8,
    'hidden_dim': 16,
    'output_dim': 2,  # (pid_weight, compliance)
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'target_model_size_kb': 10,
}


# ============================================================================
# 数据收集
# ============================================================================

def generate_synthetic_data(num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成训练数据
    
    输入特征:
        0: torque (力矩)
        1: velocity (速度)
        2: position_error (位置误差)
        3: velocity_error (速度误差)  
        4: external_force (外力)
        5: error_magnitude (误差幅度)
        6: history_0 (历史特征 1)
        7: history_1 (历史特征 2)
        
    输出:
        0: pid_weight (PID 权重 0-1)
        1: compliance (合规度 0-1)
    """
    print(f"生成 {num_samples} 个合成训练样本...")
    
    # 生成输入特征
    torque = np.random.randn(num_samples) * 5
    velocity = np.random.randn(num_samples) * 2
    position_error = np.random.randn(num_samples) * 0.3
    velocity_error = np.random.randn(num_samples) * 0.5
    external_force = np.abs(np.random.randn(num_samples)) * 10
    error_magnitude = np.abs(position_error) + np.abs(velocity_error * 0.1)
    history_0 = np.roll(torque, 1)
    history_1 = np.roll(torque, 2)
    
    X = np.stack([
        torque, velocity, position_error, velocity_error,
        external_force, error_magnitude, history_0, history_1
    ], axis=1).astype(np.float32)
    
    # 生成目标值（基于规则的标签）
    load_level = np.abs(torque) / 10.0
    
    # PID 权重：高负载时偏向 PID
    pid_weight = np.clip(0.5 + load_level * 0.4 + error_magnitude * 0.1, 0.1, 0.9)
    
    # 合规度：低负载高合规度
    compliance = np.clip(0.5 - load_level * 0.3, 0.1, 0.9)
    
    y = np.stack([pid_weight, compliance], axis=1).astype(np.float32)
    
    print(f"  特征范围: {X.min():.2f} ~ {X.max():.2f}")
    print(f"  PID 权重范围: {y[:,0].min():.2f} ~ {y[:,0].max():.2f}")
    print(f"  合规度范围: {y[:,1].min():.2f} ~ {y[:,1].max():.2f}")
    
    return X, y


def collect_real_data(duration_s: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    从实际硬件收集数据（占位符）
    
    实际实现需要连接到硬件传感器
    """
    print(f"从硬件收集 {duration_s} 秒的数据...")
    print("  警告: 使用合成数据替代（无硬件连接）")
    
    # 模拟 60 秒数据 @ 100Hz
    num_samples = duration_s * 100
    return generate_synthetic_data(num_samples)


def save_dataset(X: np.ndarray, y: np.ndarray, name: str):
    """保存数据集"""
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    
    filepath = os.path.join(CONFIG['data_dir'], f'{name}.npz')
    np.savez(filepath, X=X, y=y)
    print(f"数据集保存: {filepath} ({len(X)} 样本)")


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载数据集"""
    filepath = os.path.join(CONFIG['data_dir'], f'{name}.npz')
    data = np.load(filepath)
    return data['X'], data['y']


# ============================================================================
# 模型定义
# ============================================================================

class AdaptiveMLP(nn.Module):
    """轻量级 MLP 自适应控制器"""
    
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # 输出 0-1
        )
    
    def forward(self, x):
        return self.layers(x)


class QuantizedAdaptiveMLP(nn.Module):
    """量化感知 MLP"""
    
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # 量化参数
        self.register_buffer('scale1', torch.ones(1))
        self.register_buffer('scale2', torch.ones(1))
        self.register_buffer('scale3', torch.ones(1))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
    def quantize_weights(self):
        """量化权重到 int8"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                max_val = param.abs().max()
                scale = max_val / 127
                param.data = torch.round(param.data / scale) * scale


# ============================================================================
# 训练
# ============================================================================

def train_model(X: np.ndarray, y: np.ndarray, 
                epochs: int = 100,
                quantize_aware: bool = True) -> nn.Module:
    """训练模型"""
    
    print(f"\n{'=' * 50}")
    print("开始训练")
    print(f"{'=' * 50}")
    
    # 数据划分
    n = len(X)
    train_idx = int(n * 0.8)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:], y[train_idx:]
    
    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    
    val_tensor = (
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    
    # 创建模型
    if quantize_aware:
        model = QuantizedAdaptiveMLP(
            CONFIG['feature_dim'],
            CONFIG['hidden_dim'],
            CONFIG['output_dim']
        )
    else:
        model = AdaptiveMLP(
            CONFIG['feature_dim'],
            CONFIG['hidden_dim'],
            CONFIG['output_dim']
        )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor[0])
            val_loss = criterion(val_pred, val_tensor[1]).item()
        
        scheduler.step(val_loss)
        
        # 量化感知训练：定期量化权重
        if quantize_aware and epoch % 10 == 0:
            model.quantize_weights()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 最终量化
    if quantize_aware:
        model.quantize_weights()
    
    print(f"\n最佳验证损失: {best_val_loss:.4f}")
    
    return model


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict:
    """评估模型"""
    
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    
    mae = np.abs(pred - y).mean(axis=0)
    mse = ((pred - y) ** 2).mean(axis=0)
    
    results = {
        'mae_pid': mae[0],
        'mae_compliance': mae[1],
        'mse_pid': mse[0],
        'mse_compliance': mse[1],
    }
    
    print("\n模型评估:")
    print(f"  PID 权重 - MAE: {mae[0]:.4f}, MSE: {mse[0]:.4f}")
    print(f"  合规度 - MAE: {mae[1]:.4f}, MSE: {mse[1]:.4f}")
    
    return results


# ============================================================================
# 导出
# ============================================================================

def export_to_c_header(model: nn.Module, filepath: str):
    """导出模型为 C 头文件"""
    
    print(f"\n导出模型到 C 头文件: {filepath}")
    
    layers = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 量化到 int8
            max_val = param.abs().max().item()
            scale = max_val / 127 if max_val > 0 else 1.0
            quantized = np.round(param.detach().numpy() / scale).astype(np.int8)
            layers.append({
                'name': name.replace('.', '_'),
                'weights': quantized,
                'scale': scale
            })
        elif 'bias' in name:
            bias = (param.detach().numpy() * 128).astype(np.int32)
            layers[-1]['bias'] = bias
    
    with open(filepath, 'w') as f:
        f.write("/* Auto-generated TinyML model weights */\n")
        f.write(f"/* Generated: {datetime.now().isoformat()} */\n\n")
        f.write("#ifndef ADAPTIVE_MODEL_WEIGHTS_H\n")
        f.write("#define ADAPTIVE_MODEL_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        total_size = 0
        
        for i, layer in enumerate(layers):
            weights = layer['weights']
            f.write(f"/* Layer {i+1}: {layer['name']} */\n")
            f.write(f"static const float LAYER{i+1}_SCALE = {layer['scale']:.6f}f;\n")
            
            # 权重
            flat = weights.flatten()
            f.write(f"static const int8_t LAYER{i+1}_WEIGHTS[{len(flat)}] = {{\n    ")
            for j, w in enumerate(flat):
                f.write(f"{w}")
                if j < len(flat) - 1:
                    f.write(", ")
                if (j + 1) % 16 == 0:
                    f.write("\n    ")
            f.write("\n};\n")
            total_size += len(flat)
            
            # 偏置
            if 'bias' in layer:
                bias = layer['bias']
                f.write(f"static const int32_t LAYER{i+1}_BIAS[{len(bias)}] = {{")
                for j, b in enumerate(bias):
                    f.write(f"{b}")
                    if j < len(bias) - 1:
                        f.write(", ")
                f.write("};\n\n")
                total_size += len(bias) * 4
        
        f.write(f"/* Total model size: {total_size} bytes */\n\n")
        f.write("#endif /* ADAPTIVE_MODEL_WEIGHTS_H */\n")
    
    print(f"  模型大小: {total_size} 字节 ({total_size/1024:.1f} KB)")
    
    return total_size


def export_to_binary(model: nn.Module, filepath: str):
    """导出模型为二进制文件"""
    
    print(f"\n导出模型到二进制: {filepath}")
    
    with open(filepath, 'wb') as f:
        # 魔数
        f.write(b'TML1')
        
        # 版本
        f.write(struct.pack('<H', 0x0201))  # 2.1
        
        # 层数
        layer_count = sum(1 for n, _ in model.named_parameters() if 'weight' in n)
        f.write(struct.pack('<B', layer_count))
        
        # 写入各层
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.detach().numpy()
                max_val = np.abs(weights).max()
                scale = max_val / 127 if max_val > 0 else 1.0
                quantized = np.round(weights / scale).astype(np.int8)
                
                # 形状
                f.write(struct.pack('<HH', *weights.shape))
                # Scale
                f.write(struct.pack('<f', scale))
                # 权重数据
                f.write(quantized.tobytes())
    
    file_size = os.path.getsize(filepath)
    print(f"  文件大小: {file_size} 字节 ({file_size/1024:.1f} KB)")
    
    return file_size


def save_pytorch_model(model: nn.Module, filepath: str):
    """保存 PyTorch 模型"""
    torch.save(model.state_dict(), filepath)
    print(f"PyTorch 模型保存: {filepath}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TinyML 自适应模型训练')
    parser.add_argument('--collect', action='store_true', help='收集训练数据')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--export', action='store_true', help='导出模型')
    parser.add_argument('--all', action='store_true', help='执行完整流程')
    parser.add_argument('--samples', type=int, default=10000, help='样本数量')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    
    args = parser.parse_args()
    
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    if args.all:
        args.collect = args.train = args.export = True
    
    if not (args.collect or args.train or args.export):
        parser.print_help()
        return
    
    print("=" * 60)
    print("Hive-Reflex TinyML 自适应模型训练")
    print("=" * 60)
    
    # 收集数据
    if args.collect:
        print("\n[1] 数据收集")
        X, y = generate_synthetic_data(args.samples)
        save_dataset(X, y, 'adaptive_dataset')
    
    # 训练
    if args.train:
        print("\n[2] 模型训练")
        X, y = load_dataset('adaptive_dataset')
        model = train_model(X, y, epochs=args.epochs, quantize_aware=True)
        save_pytorch_model(model, os.path.join(CONFIG['model_dir'], 'adaptive_model.pt'))
        evaluate_model(model, X, y)
    
    # 导出
    if args.export:
        print("\n[3] 模型导出")
        
        # 加载模型
        model = QuantizedAdaptiveMLP(
            CONFIG['feature_dim'],
            CONFIG['hidden_dim'],
            CONFIG['output_dim']
        )
        model.load_state_dict(torch.load(
            os.path.join(CONFIG['model_dir'], 'adaptive_model.pt')
        ))
        
        # 导出 C 头文件
        export_to_c_header(
            model, 
            os.path.join(CONFIG['model_dir'], 'adaptive_model_weights.h')
        )
        
        # 导出二进制
        export_to_binary(
            model,
            os.path.join(CONFIG['model_dir'], 'adaptive_model.bin')
        )
    
    print("\n" + "=" * 60)
    print("训练流程完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
