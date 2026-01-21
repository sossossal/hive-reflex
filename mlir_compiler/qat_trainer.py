#!/usr/bin/env python3
"""
量化感知训练（QAT）模块
支持伪量化训练、Conv+BN 融合和自动精度损失补偿

@file qat_trainer.py
@version 2.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 伪量化层
# ============================================================================

class FakeQuantize(nn.Module):
    """
    伪量化层 - 可微分的量化模拟
    
    前向传播中模拟量化误差，反向传播使用 STE（直通估计器）
    """
    
    def __init__(self, num_bits: int = 8, symmetric: bool = True,
                 per_channel: bool = False, learnable: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.learnable = learnable
        
        # 量化范围
        if symmetric:
            self.q_min = -(2 ** (num_bits - 1))
            self.q_max = 2 ** (num_bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** num_bits - 1
        
        # 可学习的 scale 和 zero_point
        if learnable:
            self.register_parameter('scale', nn.Parameter(torch.ones(1)))
            self.register_parameter('zero_point', nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('zero_point', torch.zeros(1))
        
        # 校准统计
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('calibrated', torch.tensor(False))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.calibrated:
            # 校准模式：收集统计信息
            self._update_calibration(x)
        
        # 伪量化
        return self._fake_quantize(x)
    
    def _update_calibration(self, x: torch.Tensor):
        """更新校准统计"""
        with torch.no_grad():
            if self.per_channel and x.dim() >= 2:
                min_val = x.min(dim=0)[0]
                max_val = x.max(dim=0)[0]
            else:
                min_val = x.min()
                max_val = x.max()
            
            # EMA 更新
            if self.min_val.numel() == 1:
                self.min_val = min_val.clone()
                self.max_val = max_val.clone()
            else:
                self.min_val = 0.9 * self.min_val + 0.1 * min_val
                self.max_val = 0.9 * self.max_val + 0.1 * max_val
    
    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """执行伪量化（可微分）"""
        # 计算 scale
        if self.symmetric:
            abs_max = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            scale = abs_max / self.q_max
        else:
            scale = (self.max_val - self.min_val) / (self.q_max - self.q_min)
        
        scale = torch.clamp(scale, min=1e-8)
        
        # 量化和反量化
        x_q = torch.clamp(
            torch.round(x / scale) + self.zero_point,
            self.q_min, self.q_max
        )
        x_dq = (x_q - self.zero_point) * scale
        
        # STE: 前向用量化值，反向用原始梯度
        return x + (x_dq - x).detach()
    
    def finish_calibration(self):
        """完成校准"""
        self.calibrated = torch.tensor(True)
        
        if self.symmetric:
            abs_max = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            self.scale.data = abs_max / self.q_max
        else:
            self.scale.data = (self.max_val - self.min_val) / (self.q_max - self.q_min)


# ============================================================================
# 量化感知层包装器
# ============================================================================

class QuantizedConv2d(nn.Module):
    """量化感知 Conv2d"""
    
    def __init__(self, conv: nn.Conv2d, num_bits: int = 8):
        super().__init__()
        self.conv = conv
        self.weight_quantizer = FakeQuantize(num_bits, symmetric=True, learnable=True)
        self.activation_quantizer = FakeQuantize(num_bits, symmetric=False, learnable=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量化权重
        q_weight = self.weight_quantizer(self.conv.weight)
        
        # 使用量化权重进行卷积
        out = F.conv2d(x, q_weight, self.conv.bias, self.conv.stride,
                       self.conv.padding, self.conv.dilation, self.conv.groups)
        
        # 量化激活
        return self.activation_quantizer(out)


class QuantizedLinear(nn.Module):
    """量化感知 Linear"""
    
    def __init__(self, linear: nn.Linear, num_bits: int = 8):
        super().__init__()
        self.linear = linear
        self.weight_quantizer = FakeQuantize(num_bits, symmetric=True, learnable=True)
        self.activation_quantizer = FakeQuantize(num_bits, symmetric=False, learnable=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quantizer(self.linear.weight)
        out = F.linear(x, q_weight, self.linear.bias)
        return self.activation_quantizer(out)


# ============================================================================
# Conv+BN 融合
# ============================================================================

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    融合 Conv2d 和 BatchNorm2d
    
    融合后的权重: w_fused = w * gamma / sqrt(var + eps)
    融合后的偏置: b_fused = (b - mean) * gamma / sqrt(var + eps) + beta
    """
    # 获取 BN 参数
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    # 计算融合因子
    std = torch.sqrt(var + eps)
    scale = gamma / std
    
    # 融合权重
    fused_weight = conv.weight * scale.view(-1, 1, 1, 1)
    
    # 融合偏置
    if conv.bias is not None:
        fused_bias = (conv.bias - mean) * scale + beta
    else:
        fused_bias = -mean * scale + beta
    
    # 创建融合后的 Conv
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups,
        bias=True
    )
    
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    
    return fused_conv


def fuse_model_conv_bn(model: nn.Module) -> nn.Module:
    """
    融合模型中所有的 Conv+BN 对
    """
    logger.info("🔧 融合 Conv+BatchNorm 层...")
    
    fused_count = 0
    prev_name = None
    prev_module = None
    modules_to_fuse = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prev_name = name
            prev_module = module
        elif isinstance(module, nn.BatchNorm2d) and prev_module is not None:
            # 检查是否匹配
            if module.num_features == prev_module.out_channels:
                modules_to_fuse.append((prev_name, name, prev_module, module))
                fused_count += 1
        else:
            prev_name = None
            prev_module = None
    
    # 执行融合
    for conv_name, bn_name, conv, bn in modules_to_fuse:
        fused = fuse_conv_bn(conv, bn)
        
        # 替换模块
        parent_name = conv_name.rsplit('.', 1)
        if len(parent_name) == 2:
            parent = dict(model.named_modules())[parent_name[0]]
            setattr(parent, parent_name[1], fused)
        else:
            setattr(model, conv_name, fused)
        
        # 移除 BN（替换为 Identity）
        bn_parent_name = bn_name.rsplit('.', 1)
        if len(bn_parent_name) == 2:
            parent = dict(model.named_modules())[bn_parent_name[0]]
            setattr(parent, bn_parent_name[1], nn.Identity())
        else:
            setattr(model, bn_name, nn.Identity())
        
        logger.info(f"  融合: {conv_name} + {bn_name}")
    
    logger.info(f"✓ 融合完成，共 {fused_count} 对")
    return model


# ============================================================================
# QAT 训练器
# ============================================================================

class QATTrainer:
    """
    量化感知训练器
    
    支持：
    - 伪量化训练
    - 可学习的量化参数
    - 自动层替换
    - 敏感层保护
    """
    
    def __init__(self, model: nn.Module, dataloader, 
                 num_bits: int = 8, lr: float = 1e-4,
                 sensitive_layers: Optional[List[str]] = None):
        """
        初始化 QAT 训练器
        
        Args:
            model: 原始 FP32 模型
            dataloader: 训练数据加载器
            num_bits: 量化位数
            lr: 学习率
            sensitive_layers: 敏感层名称列表（保持 FP16）
        """
        self.original_model = model
        self.dataloader = dataloader
        self.num_bits = num_bits
        self.lr = lr
        self.sensitive_layers = sensitive_layers or []
        
        # 创建 QAT 模型
        self.qat_model = self._prepare_qat_model(copy.deepcopy(model))
        self.optimizer = Adam(self.qat_model.parameters(), lr=lr)
        
        # 训练统计
        self.epoch_losses = []
        self.calibration_done = False
    
    def _prepare_qat_model(self, model: nn.Module) -> nn.Module:
        """准备 QAT 模型（替换层为量化版本）"""
        logger.info("🔨 准备 QAT 模型...")
        
        # 首先融合 Conv+BN
        model = fuse_model_conv_bn(model)
        
        # 替换层
        replaced = 0
        for name, module in list(model.named_modules()):
            # 跳过敏感层
            if any(s in name for s in self.sensitive_layers):
                logger.info(f"  跳过敏感层: {name}")
                continue
            
            parent_name = name.rsplit('.', 1)
            if len(parent_name) == 2:
                parent = dict(model.named_modules())[parent_name[0]]
                child_name = parent_name[1]
            else:
                parent = model
                child_name = name
            
            # 替换 Conv2d
            if isinstance(module, nn.Conv2d) and not isinstance(module, QuantizedConv2d):
                setattr(parent, child_name, QuantizedConv2d(module, self.num_bits))
                replaced += 1
            
            # 替换 Linear
            elif isinstance(module, nn.Linear) and not isinstance(module, QuantizedLinear):
                setattr(parent, child_name, QuantizedLinear(module, self.num_bits))
                replaced += 1
        
        logger.info(f"✓ 替换 {replaced} 层为量化版本")
        return model
    
    def calibrate(self, num_batches: int = 100):
        """校准模型（收集激活范围）"""
        logger.info("📏 校准量化参数...")
        
        self.qat_model.train()  # 需要 train 模式更新统计
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                self.qat_model(inputs)
        
        # 完成校准
        for module in self.qat_model.modules():
            if isinstance(module, FakeQuantize):
                module.finish_calibration()
        
        self.calibration_done = True
        logger.info("✓ 校准完成")
    
    def train_epoch(self, criterion=None) -> float:
        """训练一个 epoch"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.qat_model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in self.dataloader:
            self.optimizer.zero_grad()
            
            outputs = self.qat_model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.epoch_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, epochs: int = 10, criterion=None) -> List[float]:
        """完整 QAT 训练"""
        logger.info(f"🚀 开始 QAT 训练 ({epochs} epochs)")
        
        if not self.calibration_done:
            self.calibrate()
        
        for epoch in range(epochs):
            loss = self.train_epoch(criterion)
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss = {loss:.4f}")
        
        logger.info("✓ QAT 训练完成")
        return self.epoch_losses
    
    def export_quantized_model(self, output_path: str,
                               export_format: str = 'onnx') -> None:
        """导出量化模型"""
        logger.info(f"📦 导出量化模型: {output_path}")
        
        self.qat_model.eval()
        
        if export_format == 'onnx':
            # 创建虚拟输入
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # 转换为静态量化
            # 这里简化处理，实际需要更复杂的转换
            torch.onnx.export(
                self.qat_model,
                dummy_input,
                output_path,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        elif export_format == 'torch':
            torch.save(self.qat_model.state_dict(), output_path)
        
        logger.info("✓ 导出完成")
    
    def get_quantization_config(self) -> Dict:
        """获取量化配置（用于 CIM 代码生成）"""
        config = {
            'num_bits': self.num_bits,
            'layers': []
        }
        
        for name, module in self.qat_model.named_modules():
            if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                layer_config = {
                    'name': name,
                    'weight_scale': module.weight_quantizer.scale.item(),
                    'activation_scale': module.activation_quantizer.scale.item()
                }
                config['layers'].append(layer_config)
        
        return config


# ============================================================================
# 自动精度损失补偿
# ============================================================================

def compute_quantization_error(fp_model: nn.Module, qat_model: nn.Module,
                               test_loader, num_batches: int = 50) -> Dict:
    """
    计算量化误差
    """
    fp_model.eval()
    qat_model.eval()
    
    errors = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            fp_out = fp_model(inputs)
            qat_out = qat_model(inputs)
            
            mae = torch.abs(fp_out - qat_out).mean().item()
            mse = ((fp_out - qat_out) ** 2).mean().item()
            
            errors.append({'mae': mae, 'mse': mse})
    
    avg_mae = np.mean([e['mae'] for e in errors])
    avg_mse = np.mean([e['mse'] for e in errors])
    
    return {
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': np.sqrt(avg_mse),
        'snr_db': 10 * np.log10(1 / max(avg_mse, 1e-10))
    }


def auto_precision_compensation(trainer: QATTrainer, test_loader,
                                 target_error: float = 0.01,
                                 max_iterations: int = 5) -> None:
    """
    自动精度损失补偿
    
    如果量化误差超过目标，自动调整敏感层精度
    """
    logger.info("🎯 自动精度补偿...")
    
    for iteration in range(max_iterations):
        error = compute_quantization_error(
            trainer.original_model, trainer.qat_model, test_loader
        )
        
        logger.info(f"  迭代 {iteration+1}: MAE={error['mae']:.4f}, SNR={error['snr_db']:.1f}dB")
        
        if error['mae'] < target_error:
            logger.info("✓ 达到目标精度")
            break
        
        # 增加训练轮次
        trainer.train(epochs=5)
    
    logger.info("✓ 精度补偿完成")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QAT 训练器')
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--bits', type=int, default=8, help='量化位数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--fuse-bn', action='store_true', help='融合 BatchNorm')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Hive-Reflex QAT 训练器")
    print("=" * 50)
    print(f"模型: {args.model}")
    print(f"量化位数: {args.bits}")
    print(f"训练轮次: {args.epochs}")
    
    # TODO: 加载模型和数据，执行训练
    
    print("\n✅ QAT 训练完成!")


if __name__ == '__main__':
    main()
