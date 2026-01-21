#!/usr/bin/env python3
"""
CIM 代码生成器 - 针对 Digital CIM 架构优化的代码生成
"""

import onnx
from onnx import numpy_helper
import numpy as np
from typing import List, Dict

class CIMCodeGenerator:
    """CIM 目标代码生成器"""
    
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        self.layer_count = 0
        self.code_lines = []
        
    def generate(self, output_c: str, output_weights: str, output_config: str):
        """
        生成 CIM 优化的 C 代码
        
        Args:
            output_c: C 代码输出路径
            output_weights: 权重二进制输出路径
            output_config: 配置JSON输出路径
        """
        print("🔨 CIM 代码生成器")
        print("=" * 50)
        
        # 分析模型结构
        layers = self._analyze_graph()
        print(f"✓ 分析模型: {len(layers)} 层")
        
        # 生成代码
        self._generate_header()
        self._generate_inference_function(layers)
        self._generate_footer()
        
        # 写入文件
        with open(output_c, 'w') as f:
            f.write('\n'.join(self.code_lines))
        
        print(f"✓ 生成 C 代码: {output_c}")
        
        # 导出权重
        weights_data = self._export_weights()
        with open(output_weights, 'wb') as f:
            f.write(weights_data)
        
        print(f"✓ 导出权重: {output_weights} ({len(weights_data)} bytes)")
        
        # 生成配置
        config = self._generate_config(layers)
        import json
        with open(output_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ 生成配置: {output_config}")
        
        return output_c
    
    def _analyze_graph(self) -> List[Dict]:
        """分析计算图，提取层信息"""
        layers = []
        
        for node in self.graph.node:
            layer = {
                'name': node.name or f"layer_{len(layers)}",
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
            }
            
            # 提取属性
            if node.op_type == 'Gemm' or node.op_type == 'MatMul':
                layer['type'] = 'fc'
                # 从 initializer 获取形状
                for init in self.graph.initializer:
                    if init.name in node.input:
                        weights = numpy_helper.to_array(init)
                        layer['shape'] = weights.shape
            
            elif node.op_type == 'LSTM':
                layer['type'] = 'lstm'
                # 获取 LSTM 参数
                for attr in node.attribute:
                    if attr.name == 'hidden_size':
                        layer['hidden_size'] = attr.i
            
            elif node.op_type == 'Relu' or node.op_type.endswith('Relu'):
                layer['type'] = 'activation'
                layer['activation'] = 'relu'
            
            elif node.op_type == 'Tanh':
                layer['type'] = 'activation'
                layer['activation'] = 'tanh'
            
            layers.append(layer)
        
        return layers
    
    def _generate_header(self):
        """生成代码头部"""
        self.code_lines.extend([
            "/**",
            " * 自动生成的 CIM 推理代码",
            " * 由 MLIR 编译器生成",
            " */",
            "",
            '#include "imc22_cim.h"',
            '#include "model_loader.h"',
            '#include <string.h>',
            "",
            "// 权重数据 (在 FLASH 中)",
            "extern const uint8_t model_weights[];",
            "extern const uint32_t model_weights_size;",
            "",
        ])
    
    def _generate_inference_function(self, layers: List[Dict]):
        """生成推理函数"""
        self.code_lines.extend([
            "/**",
            " * @brief 模型推理函数",
            " * @param input 输入数据",
            " * @param output 输出数据",
            " * @param context 推理上下文",
            " * @return 0 成功, -1 失败",
            " */",
            "int model_inference_optimized(const float *input, float *output, void *context) {",
            "    InferenceContext_t *ctx = (InferenceContext_t*)context;",
            "    ",
            "    // 临时缓冲区",
            "    float *temp1 = ctx->temp_buffer;",
            f"    float *temp2 = temp1 + {self._estimate_buffer_size(layers)};",
            "    ",
        ])
        
        # 为每一层生成代码
        input_var = "input"
        
        for i, layer in enumerate(layers):
            output_var = "output" if i == len(layers) - 1 else f"temp{(i % 2) + 1}"
            
            if layer['type'] == 'fc':
                self._generate_fc_layer(layer, input_var, output_var, i)
            elif layer['type'] == 'lstm':
                self._generate_lstm_layer(layer, input_var, output_var, i)
            elif layer['type'] == 'activation':
                self._generate_activation(layer, input_var, output_var)
            
            input_var = output_var
        
        self.code_lines.extend([
            "    ",
            "    return 0;",
            "}",
            "",
        ])
    
    def _generate_fc_layer(self, layer: Dict, input_var: str, output_var: str, idx: int):
        """生成全连接层代码"""
        self.code_lines.extend([
            f"    // Layer {idx}: 全连接 ({layer.get('shape', 'unknown')})",
            f"    {{",
            f"        const float *weights = (const float*)(model_weights + weight_offset_{idx});",
            f"        const float *bias = weights + {layer.get('shape', [0,0])[0] * layer.get('shape', [0,0])[1]};",
            f"        ",
            f"        // 使用 CIM 加速",
            f"        CIM_FullyConnected(",
            f"            {input_var}, {output_var},",
            f"            weights, bias,",
            f"            {layer.get('shape', [0,0])[1]}, {layer.get('shape', [0,0])[0]},",
            f"            {1 if 'Relu' in layer.get('op_type', '') else 0}  // 激活函数",
            f"        );",
            f"    }}",
            "    ",
        ])
    
    def _generate_lstm_layer(self, layer: Dict, input_var: str, output_var: str, idx: int):
        """生成 LSTM 层代码"""
        hidden_size = layer.get('hidden_size', 16)
        
        self.code_lines.extend([
            f"    // Layer {idx}: LSTM (hidden={hidden_size})",
            f"    {{",
            f"        const float *weights = (const float*)(model_weights + weight_offset_{idx});",
            f"        ",
            f"        // 使用 CIM LSTM 加速器",
            f"        CIM_LSTM(",
            f"            {input_var},",
            f"            ctx->lstm_h,",
            f"            ctx->lstm_c,",
            f"            ctx->lstm_h,  // 更新隐藏状态",
            f"            ctx->lstm_c,  // 更新细胞状态",
            f"            (void*)weights",
            f"        );",
            f"        ",
            f"        // 复制输出",
            f"        memcpy({output_var}, ctx->lstm_h, {hidden_size} * sizeof(float));",
            f"    }}",
            "    ",
        ])
    
    def _generate_activation(self, layer: Dict, input_var: str, output_var: str):
        """生成激活函数代码"""
        act_type = layer.get('activation', 'relu')
        
        if input_var != output_var:
            self.code_lines.append(f"    memcpy({output_var}, {input_var}, layer_size * sizeof(float));")
        
        if act_type == 'relu':
            self.code_lines.append(f"    CIM_ReLU({output_var}, layer_size);")
        elif act_type == 'tanh':
            self.code_lines.append(f"    CIM_Tanh({output_var}, layer_size);")
        
        self.code_lines.append("    ")
    
    def _generate_footer(self):
        """生成代码尾部"""
        self.code_lines.extend([
            "// 权重偏移量 (自动计算)",
            "const uint32_t weight_offset_0 = 0;",
            "// ... (其他层的偏移)",
            "",
        ])
    
    def _export_weights(self) -> bytes:
        """导出权重为二进制"""
        weights_list = []
        
        for init in self.graph.initializer:
            tensor = numpy_helper.to_array(init)
            
            # 转换为 INT8 (简化版)
            if tensor.dtype == np.float32:
                scale = max(abs(tensor.min()), abs(tensor.max())) / 127.0
                tensor_int8 = np.clip(tensor / scale, -128, 127).astype(np.int8)
                weights_list.append(tensor_int8.tobytes())
            else:
                weights_list.append(tensor.tobytes())
        
        return b''.join(weights_list)
    
    def _generate_config(self, layers: List[Dict]) -> Dict:
        """生成模型配置"""
        return {
            'model_name': 'optimized_model',
            'num_layers': len(layers),
            'layers': [
                {
                    'name': layer['name'],
                    'type': layer['type'],
                    'shape': str(layer.get('shape', 'unknown'))
                }
                for layer in layers
            ]
        }
    
    def _estimate_buffer_size(self, layers: List[Dict]) -> int:
        """估计缓冲区大小"""
        max_size = 0
        
        for layer in layers:
            if 'shape' in layer:
                size = max(layer['shape'])
                max_size = max(max_size, size)
        
        return max_size or 256


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIM 代码生成器')
    parser.add_argument('--model', required=True, help='ONNX 模型路径')
    parser.add_argument('--output-c', default='generated_inference.c', help='C 代码输出')
    parser.add_argument('--output-weights', default='generated_weights.bin', help='权重输出')
    parser.add_argument('--output-config', default='model_config.json', help='配置输出')
    
    args = parser.parse_args()
    
    model = onnx.load(args.model)
    generator = CIMCodeGenerator(model)
    generator.generate(args.output_c, args.output_weights, args.output_config)
    
    print("\n✅ 代码生成完成!")


if __name__ == '__main__':
    main()
