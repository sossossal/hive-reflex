"""
量化引擎 - INT8/INT4 量化
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class QuantParams:
    """量化参数"""
    scale: float
    zero_point: int
    dtype: str  # 'int8' or 'int4'


class Quantizer:
    """量化器"""
    
    def __init__(self):
        self.quant_params = {}
    
    def quantize_int8(self, weights: np.ndarray) -> tuple:
        """量化为 INT8"""
        # 计算 scale 和 zero_point
        w_min = weights.min()
        w_max = weights.max()
        
        scale = (w_max - w_min) / 255.0
        zero_point = int(-w_min / scale)
        
        # 量化
        w_quant = np.clip(np.round(weights / scale + zero_point), 0, 255).astype(np.uint8)
        
        params = QuantParams(scale=scale, zero_point=zero_point, dtype='int8')
        
        return w_quant, params
    
    def quantize_int4(self, weights: np.ndarray) -> tuple:
        """量化为 INT4"""
        # 计算 scale 和 zero_point
        w_min = weights.min()
        w_max = weights.max()
        
        scale = (w_max - w_min) / 15.0
        zero_point = int(-w_min / scale)
        
        # 量化到 0-15
        w_quant = np.clip(np.round(weights / scale + zero_point), 0, 15).astype(np.uint8)
        
        # 打包：两个 INT4 放到一个字节
        w_packed = self._pack_int4(w_quant)
        
        params = QuantParams(scale=scale, zero_point=zero_point, dtype='int4')
        
        return w_packed, params
    
    def _pack_int4(self, data: np.ndarray) -> np.ndarray:
        """打包 INT4 数据（两个值一个字节）"""
        flat = data.flatten()
        
        # 确保长度为偶数
        if len(flat) % 2 != 0:
            flat = np.append(flat, 0)
        
        # 打包
        packed = np.zeros(len(flat) // 2, dtype=np.uint8)
        for i in range(0, len(flat), 2):
            packed[i // 2] = (flat[i] << 4) | flat[i + 1]
        
        return packed
    
    def quantize_model(self, model: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """量化整个模型"""
        print(f"  使用 {strategy.upper()} 量化策略")
        
        total_original = 0
        total_quantized = 0
        
        for layer in model['layers']:
            if layer['weights'] is None:
                continue
            
            weights = layer['weights']
            bias = layer['bias']
            
            if strategy == 'int8':
                w_quant, w_params = self.quantize_int8(weights)
                if bias is not None:
                    b_quant, b_params = self.quantize_int8(bias)
                else:
                    b_quant, b_params = None, None
                
            elif strategy == 'int4':
                w_quant, w_params = self.quantize_int4(weights)
                if bias is not None:
                    b_quant, b_params = self.quantize_int4(bias)
                else:
                    b_quant, b_params = None, None
                
            elif strategy == 'mixed':
                # 前几层用 INT8，后面用 INT4
                layer_idx = model['layers'].index(layer)
                if layer_idx < len(model['layers']) // 2:
                    w_quant, w_params = self.quantize_int8(weights)
                    if bias is not None:
                        b_quant, b_params = self.quantize_int8(bias)
                    else:
                        b_quant, b_params = None, None
                else:
                    w_quant, w_params = self.quantize_int4(weights)
                    if bias is not None:
                        b_quant, b_params = self.quantize_int4(bias)
                    else:
                        b_quant, b_params = None, None
            else:
                # 不量化
                w_quant = weights
                b_quant = bias
                w_params = None
                b_params = None
            
            # 更新层
            layer['weights_quantized'] = w_quant
            layer['bias_quantized'] = b_quant
            layer['quant_params_weights'] = w_params
            layer['quant_params_bias'] = b_params
            
            # 统计
            total_original += layer['size']
            if w_quant is not None:
                total_quantized += w_quant.nbytes
            if b_quant is not None:
                total_quantized += b_quant.nbytes
        
        model['quantized_size'] = total_quantized
        
        return model


if __name__ == '__main__':
    # 测试
    quantizer = Quantizer()
    
    # 测试 INT8
    weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
    w_int8, params = quantizer.quantize_int8(weights)
    print(f"INT8: {weights.nbytes} -> {w_int8.nbytes} ({4/1:.1f}x)")
    
    # 测试 INT4
    w_int4, params = quantizer.quantize_int4(weights)
    print(f"INT4: {weights.nbytes} -> {w_int4.nbytes} ({8/1:.1f}x)")
