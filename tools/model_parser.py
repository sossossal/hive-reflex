"""
模型解析器 - 支持 ONNX 模型解析
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class LayerInfo:
    """层信息"""
    name: str
    type: str
    input_shape: tuple
    output_shape: tuple
    weights: np.ndarray = None
    bias: np.ndarray = None
    
    @property
    def num_params(self):
        """参数数量"""
        count = 0
        if self.weights is not None:
            count += self.weights.size
        if self.bias is not None:
            count += self.bias.size
        return count
    
    @property
    def size_bytes(self):
        """权重大小（字节）"""
        size = 0
        if self.weights is not None:
            size += self.weights.nbytes
        if self.bias is not None:
            size += self.bias.nbytes
        return size


class ModelParser:
    """ONNX 模型解析器"""
    
    def __init__(self):
        self.model = None
        self.layers = []
        
    def load_onnx(self, model_path: str) -> Dict[str, Any]:
        """加载 ONNX 模型"""
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            print("警告: onnx 未安装，使用模拟数据")
            return self._create_dummy_model(model_path)
        
        print(f"  加载 ONNX 模型: {model_path}")
        self.model = onnx.load(model_path)
        
        # 提取层信息
        self.layers = self._extract_layers()
        
        total_params = sum(layer.num_params for layer in self.layers)
        total_size = sum(layer.size_bytes for layer in self.layers)
        
        return {
            'layers': [self._layer_to_dict(l) for l in self.layers],
            'total_params': total_params,
            'original_size': total_size,
        }
    
    def _extract_layers(self) -> List[LayerInfo]:
        """从 ONNX 模型提取层"""
        layers = []
        
        # 遍历计算图中的节点
        for node in self.model.graph.node:
            if node.op_type in ['Conv', 'Gemm', 'MatMul']:
                layer = LayerInfo(
                    name=node.name or f"{node.op_type}_{len(layers)}",
                    type=node.op_type,
                    input_shape=(1, 3, 224, 224),  # 简化
                    output_shape=(1, 64, 112, 112),
                )
                
                # 提取权重
                for init in self.model.graph.initializer:
                    if init.name in node.input:
                        from onnx import numpy_helper
                        weights = numpy_helper.to_array(init)
                        if layer.weights is None:
                            layer.weights = weights
                        else:
                            layer.bias = weights
                
                layers.append(layer)
        
        return layers
    
    def _create_dummy_model(self, model_path: str) -> Dict[str, Any]:
        """创建模拟模型（用于测试）"""
        print("  警告: 使用模拟数据")
        
        # 模拟 8 层 CNN
        layer_sizes = [
            (3, 64, 3, 3),    # Conv1: 3x64x3x3
            (64, 128, 3, 3),  # Conv2
            (128, 128, 3, 3), # Conv3
            (128, 256, 3, 3), # Conv4
            (256, 256, 3, 3), # Conv5
            (256, 512, 3, 3), # Conv6
            (512, 1024),      # FC1
            (1024, 10),       # FC2
        ]
        
        total_params = 0
        total_size = 0
        layers = []
        
        for i, shape in enumerate(layer_sizes):
            if len(shape) == 4:  # Conv
                weights = np.random.randn(*shape).astype(np.float32)
                bias = np.random.randn(shape[1]).astype(np.float32)
            else:  # FC
                weights = np.random.randn(*shape).astype(np.float32)
                bias = np.random.randn(shape[1]).astype(np.float32)
            
            layer_info = LayerInfo(
                name=f'layer_{i}',
                type='Conv' if len(shape) == 4 else 'Gemm',
                input_shape=shape[:2] if len(shape) == 2 else shape,
                output_shape=shape[1:],
                weights=weights,
                bias=bias
            )
            
            layers.append(layer_info)
            total_params += layer_info.num_params
            total_size += layer_info.size_bytes
        
        return {
            'layers': [self._layer_to_dict(l) for l in layers],
            'total_params': total_params,
            'original_size': total_size,
        }
    
    def _layer_to_dict(self, layer: LayerInfo) -> Dict[str, Any]:
        """转换层信息为字典"""
        return {
            'name': layer.name,
            'type': layer.type,
            'size': layer.size_bytes,
            'params': layer.num_params,
            'weights': layer.weights,
            'bias': layer.bias,
        }


if __name__ == '__main__':
    # 测试
    parser = ModelParser()
    model = parser.load_onnx("dummy.onnx")
    print(f"模型层数: {len(model['layers'])}")
    print(f"总参数量: {model['total_params']:,}")
    print(f"模型大小: {model['original_size'] / 1024 / 1024:.2f} MB")
