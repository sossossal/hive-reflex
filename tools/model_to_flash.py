#!/usr/bin/env python3
"""
Hive-Reflex 模型自动化部署工具链
功能：切片 → 量化 → 打包成 Flash .bin 文件

使用方法:
    # 从 PyTorch 模型部署
    python model_to_flash.py --input model.pth --output firmware.bin
    
    # 从 ONNX 模型部署
    python model_to_flash.py --input model.onnx --output firmware.bin
    
    # 自动量化和切片大模型
    python model_to_flash.py --input large_model.onnx --output firmware.bin --auto-slice

@file model_to_flash.py
@version 2.1.0
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import numpy_helper
import struct
import os
import argparse
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 配置常量
# ============================================================================

FLASH_PAGE_SIZE = 4096  # Flash 页大小（字节）
CIM_SRAM_SIZE = 512 * 1024  # 512 KB
MAX_LAYER_SIZE = 256 * 1024  # 单层最大尺寸

FIRMWARE_MAGIC = b'HRF2'  # Hive-Reflex Firmware v2
FIRMWARE_VERSION = 0x0210  #  2.1.0


# ============================================================================
# 模型加载
# ============================================================================

def load_pytorch_model(model_path: str) -> nn.Module:
    """加载 PyTorch 模型"""
    logger.info(f"📦 加载 PyTorch 模型: {model_path}")
    
    if model_path.endswith('.pth') or model_path.endswith('.pt'):
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict):  # state_dict
            raise ValueError("请提供完整模型，而非 state_dict")
        return model
    else:
        raise ValueError(f"不支持的 PyTorch 文件格式: {model_path}")


def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """加载 ONNX 模型"""
    logger.info(f"📦 加载 ONNX 模型: {model_path}")
    
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    return model


def extract_onnx_weights(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """从 ONNX 模型提取权重"""
    weights = {}
    
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    
    logger.info(f"  提取 {len(weights)} 个权重张量")
    
    return weights


# ============================================================================
# 量化
# ============================================================================

def quantize_weights_int8(weights: np.ndarray, symmetric: bool = True) -> Tuple[np.ndarray, float]:
    """
    量化权重到 INT8
    
    Returns:
        quantized: INT8 量化后的权重
        scale: 量化比例因子
    """
    if symmetric:
        # 对称量化 [-127, 127]
        abs_max = np.abs(weights).max()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    else:
        # 非对称量化 [0, 255]
        w_min = weights.min()
        w_max = weights.max()
        scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
        zero_point = int(-w_min / scale)
        quantized = np.clip(np.round(weights / scale + zero_point), 0, 255).astype(np.uint8)
        # 转回有符号表示
        quantized = (quantized.astype(np.int16) - zero_point).astype(np.int8)
    
    return quantized, scale


def quantize_model(weights_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    量化整个模型的权重
    
    Returns:
        quantized_dict: 字典，包含量化权重和 scale
    """
    logger.info(f"🔢 量化模型权重到 INT8...")
    
    quantized_dict = {}
    total_original_size = 0
    total_quantized_size = 0
    
    for name, weights in weights_dict.items():
        original_size = weights.nbytes
        
        quantized, scale = quantize_weights_int8(weights, symmetric=True)
        quantized_size = quantized.nbytes
        
        quantized_dict[name] = {
            'weights': quantized,
            'scale': scale,
            'shape': weights.shape,
            'dtype': 'int8'
        }
        
        total_original_size += original_size
        total_quantized_size += quantized_size
    
    compression_ratio = total_original_size / total_quantized_size if total_quantized_size > 0 else 1
    
    logger.info(f"  原始大小: {total_original_size/1024:.1f} KB")
    logger.info(f"  量化后: {total_quantized_size/1024:.1f} KB")
    logger.info(f"  压缩比: {compression_ratio:.2f}x")
    
    return quantized_dict


# ============================================================================
# 模型切片
# ============================================================================

def slice_model_layers(quantized_dict: Dict[str, Dict], max_layer_size: int = MAX_LAYER_SIZE) -> List[Dict]:
    """
    将模型切片为适合 CIM SRAM 的层组
    
    Args:
        quantized_dict: 量化后的权重字典
        max_layer_size: 单个切片的最大尺寸（字节）
    
    Returns:
        slices: 切片列表，每个切片包含多个层
    """
    logger.info(f"✂️  切片模型 (最大切片: {max_layer_size/1024:.0f} KB)...")
    
    slices = []
    current_slice = {'layers': {}, 'size': 0}
    
    for name, layer_data in quantized_dict.items():
        layer_size = layer_data['weights'].nbytes + 4 + 4  # weights + scale + shape
        
        # 如果单层超过最大尺寸，需要进一步切分权重
        if layer_size > max_layer_size:
            logger.warning(f"  层 {name} ({layer_size/1024:.1f} KB) 超过最大尺寸，将进行权重切分")
            # 这里实现权重级别的切分（简化版）
            # 实际需要考虑计算图的依赖关系
            sub_slices = _split_large_layer(name, layer_data, max_layer_size)
            for sub_slice in sub_slices:
                slices.append(sub_slice)
            continue
        
        # 检查是否需要创建新切片
        if current_slice['size'] + layer_size > max_layer_size:
            if current_slice['layers']:  # 当前切片非空
                slices.append(current_slice)
            current_slice = {'layers': {}, 'size': 0}
        
        # 添加层到当前切片
        current_slice['layers'][name] = layer_data
        current_slice['size'] += layer_size
    
    # 添加最后一个切片
    if current_slice['layers']:
        slices.append(current_slice)
    
    logger.info(f"  生成 {len(slices)} 个切片")
    for i, s in enumerate(slices):
        logger.info(f"    切片 {i+1}: {len(s['layers'])} 层, {s['size']/1024:.1f} KB")
    
    return slices


def _split_large_layer(name: str, layer_data: Dict, max_size: int) -> List[Dict]:
    """切分单个超大层"""
    weights = layer_data['weights']
    shape = weights.shape
    
    # 简化策略：沿第一维切分
    if len(shape) >= 2:
        dim0_size = shape[0]
        bytes_per_row = weights[0].nbytes
        max_rows = max_size // bytes_per_row
        
        sub_slices = []
        for i in range(0, dim0_size, max_rows):
            end = min(i + max_rows, dim0_size)
            sub_weights = weights[i:end]
            
            sub_slice = {
                'layers': {
                    f"{name}_part{i//max_rows}": {
                        'weights': sub_weights,
                        'scale': layer_data['scale'],
                        'shape': sub_weights.shape,
                        'dtype': 'int8',
                        'is_partial': True,
                        'partial_index': (i, end)
                    }
                },
                'size': sub_weights.nbytes
            }
            sub_slices.append(sub_slice)
        
        logger.info(f"    → 切分为 {len(sub_slices)} 个子层")
        return sub_slices
    else:
        # 无法切分，返回原层
        return [{'layers': {name: layer_data}, 'size': weights.nbytes}]


# ============================================================================
# Flash 固件打包
# ============================================================================

def generate_flash_firmware(slices: List[Dict], output_path: str, metadata: Optional[Dict] = None):
    """
    生成 Flash 友好的 .bin 固件文件
    
    固件格式:
    [Header]
        Magic: 4 bytes ('HRF2')
        Version: 2 bytes (0x0210 = 2.1.0)
        Num Slices: 2 bytes
        Total Size: 4 bytes
        Metadata Length: 2 bytes
        Reserved: 2 bytes
    [Metadata JSON]
        可变长度元数据
    [Slice 0]
        Slice Header: 8 bytes
        Layer Count: 2 bytes
        Reserved: 2 bytes
        Layers...
    [Slice 1]
        ...
    [Padding to Page Boundary]
    """
    logger.info(f"📦 打包 Flash 固件: {output_path}")
    
    with open(output_path, 'wb') as f:
        # ========== 固件头部 ==========
        f.write(FIRMWARE_MAGIC)  # Magic
        f.write(struct.pack('<H', FIRMWARE_VERSION))  # Version
        f.write(struct.pack('<H', len(slices)))  # Num Slices
        
        # 计算总大小（稍后回填）
        total_size_offset = f.tell()
        f.write(struct.pack('<I', 0))  # Total Size (placeholder)
        
        # 元数据
        if metadata is None:
            metadata = {
                'model_name': 'untitled',
                'timestamp': str(np.datetime64('now')),
                'num_slices': len(slices)
            }
        metadata_json = json.dumps(metadata, indent=None).encode('utf-8')
        f.write(struct.pack('<H', len(metadata_json)))  # Metadata Length
        f.write(struct.pack('<H', 0))  # Reserved
        f.write(metadata_json)  # Metadata
        
        # ========== 切片数据 ==========
        for slice_idx, slice_data in enumerate(slices):
            slice_start = f.tell()
            
            # 切片头部
            f.write(struct.pack('<I', slice_data['size']))  # Slice Size
            f.write(struct.pack('<H', len(slice_data['layers'])))  # Layer Count
            f.write(struct.pack('<H', 0))  # Reserved
            
            # 各层数据
            for layer_name, layer_info in slice_data['layers'].items():
                # 层头部
                layer_name_bytes = layer_name.encode('utf-8')
                f.write(struct.pack('<H', len(layer_name_bytes)))
                f.write(layer_name_bytes)
                
                # 形状
                shape = layer_info['shape']
                f.write(struct.pack('<B', len(shape)))  # Num Dims
                for dim in shape:
                    f.write(struct.pack('<I', dim))
                
                # Scale
                f.write(struct.pack('<f', layer_info['scale']))
                
                # 权重数据
                weights = layer_info['weights']
                f.write(struct.pack('<I', weights.nbytes))
                f.write(weights.tobytes())
        
        # ========== 对齐到 Flash 页边界 ==========
        current_pos = f.tell()
        padding_size = (FLASH_PAGE_SIZE - (current_pos % FLASH_PAGE_SIZE)) % FLASH_PAGE_SIZE
        if padding_size > 0:
            f.write(b'\xFF' * padding_size)  # Flash 擦除默认 0xFF
        
        # 回填总大小
        total_size = f.tell()
        f.seek(total_size_offset)
        f.write(struct.pack('<I', total_size))
    
    file_size = os.path.getsize(output_path)
    logger.info(f"  ✅ 固件大小: {file_size} 字节 ({file_size/1024:.1f} KB)")
    logger.info(f"  Flash 页数: {file_size // FLASH_PAGE_SIZE} 页 (+{file_size % FLASH_PAGE_SIZE} 字节)")
    
    return file_size


# ============================================================================
# 完整流程
# ============================================================================

def model_to_flash(input_path: str, output_path: str, auto_slice: bool = True, metadata: Optional[Dict] = None):
    """
    模型 → Flash 固件的完整流程
    
    Args:
        input_path: 输入模型路径 (.pth, .onnx)
        output_path: 输出 .bin 固件路径
        auto_slice: 是否自动切片大模型
        metadata: 可选的元数据字典
    """
    logger.info("=" * 60)
    logger.info("Hive-Reflex 模型自动化部署工具链")
    logger.info("=" * 60)
    
    # Step 1: 加载模型
    if input_path.endswith('.onnx'):
        onnx_model = load_onnx_model(input_path)
        weights_dict = extract_onnx_weights(onnx_model)
    elif input_path.endswith(('.pth', '.pt')):
        pytorch_model = load_pytorch_model(input_path)
        weights_dict = {name: param.detach().cpu().numpy() 
                        for name, param in pytorch_model.named_parameters()}
    else:
        raise ValueError(f"不支持的模型格式: {input_path}")
    
    # Step 2: 量化
    quantized_dict = quantize_model(weights_dict)
    
    # Step 3: 切片（如果需要）
    if auto_slice:
        slices = slice_model_layers(quantized_dict, max_layer_size=MAX_LAYER_SIZE)
    else:
        # 单切片（整个模型）
        total_size = sum(d['weights'].nbytes for d in quantized_dict.values())
        if total_size > CIM_SRAM_SIZE:
            logger.warning(f"⚠️  模型大小 ({total_size/1024:.0f} KB) 超过 SRAM ({CIM_SRAM_SIZE/1024:.0f} KB)！")
            logger.warning("    建议使用 --auto-slice 选项")
        
        slices = [{'layers': quantized_dict, 'size': total_size}]
    
    # Step 4: 生成固件
    if metadata is None:
        metadata = {
            'model_name': Path(input_path).stem,
            'input_format': Path(input_path).suffix[1:],  # 去掉 '.'
            'cim_sram_size': CIM_SRAM_SIZE,
            'quantization': 'int8',
        }
    
    firmware_size = generate_flash_firmware(slices, output_path, metadata)
    
    logger.info("=" * 60)
    logger.info("✅ 部署完成!")
    logger.info(f"    输入: {input_path}")
    logger.info(f"    输出: {output_path} ({firmware_size/1024:.1f} KB)")
    logger.info(f"    切片: {len(slices)} 个")
    logger.info("=" * 60)
    
    return output_path


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hive-Reflex 模型自动化部署工具链',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 部署 ONNX 模型
  python model_to_flash.py --input model.onnx --output firmware.bin
  
  # 部署大模型（自动切片）
  python model_to_flash.py --input large_model.onnx --output firmware.bin --auto-slice
  
  # 添加元数据
  python model_to_flash.py --input model.pth --output firmware.bin --name "MyModel" --version 1.0
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='输入模型路径 (.pth, .onnx)')
    parser.add_argument('--output', '-o', required=True, help='输出固件路径 (.bin)')
    parser.add_argument('--auto-slice', action='store_true', help='自动切片大模型')
    parser.add_argument('--max-slice', type=int, default=MAX_LAYER_SIZE, help='最大切片大小（字节）')
    parser.add_argument('--name', help='模型名称（元数据）')
    parser.add_argument('--version', help='模型版本（元数据）')
    
    args = parser.parse_args()
    
    # 构建元数据
    metadata = {}
    if args.name:
        metadata['model_name'] = args.name
    if args.version:
        metadata['model_version'] = args.version
    
    # 执行部署
    try:
        model_to_flash(
            args.input,
            args.output,
            auto_slice=args.auto_slice,
            metadata=metadata if metadata else None
        )
    except Exception as e:
        logger.error(f"❌ 部署失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
