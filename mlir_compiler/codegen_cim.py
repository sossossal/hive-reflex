#!/usr/bin/env python3
"""
CIM 代码生成器 - 针对 Digital CIM 架构优化的代码生成
"""

import sys
import os
import time

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    # Fallback for environments without ONNX (Demo Mode)
    print("Warning: ONNX not found. Using Mock.")
    from unittest.mock import MagicMock
    onnx = MagicMock()
    numpy_helper = MagicMock()
    
    # Mock load to return a dummy model structure
    def mock_load(path):
        m = MagicMock()
        # Create a dummy graph with 1 layer
        node = MagicMock()
        node.op_type = 'Gemm' # FC
        node.name = 'fc1'
        node.attribute = []
        node.input = ['in']
        node.output = ['out']
        m.graph.node = [node]
        m.graph.initializer = [] # Ensure initializer is mocked too
        return m
    onnx.load = mock_load

import numpy as np
from typing import List, Dict

from transformer_ops import TransformerMapper
from memory_planner import MemoryPlanner
from rtl_pruner import RTLPruner

def parse_onnx_model(onnx_model, transformer_mapper, graph_initializer) -> List[Dict]:
    """
    Parse ONNX graph into linear layer list
    """
    layers = []
    # If mock
    if isinstance(onnx_model, MagicMock):
        # This branch is for the mocked onnx.load, which returns a MagicMock model
        # The mock_load function already sets up a dummy graph.node.
        # We need to process it similarly to a real ONNX model, but with mocked data.
        # For simplicity, let's return a predefined mock layer list here.
        # A more robust mock might iterate through the mocked graph.node.
        return [{'name': 'fc1', 'type': 'fc', 'shape': (10, 10), 'inputs': ['in'], 'outputs': ['out']}]
        
    for node in onnx_model.graph.node:
        # Check if it's a Transformer op first
        transformer_layer = transformer_mapper.analyze_node(node)
        if transformer_layer:
            layers.append(transformer_layer)
            continue

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
            for init in graph_initializer:
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


class CIMCodeGenerator:
    """CIM 目标代码生成器"""
    
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        self.layer_count = 0
        self.code_lines = []
        self.transformer_mapper = TransformerMapper()
        
    def generate(self, output_c: str, output_weights: str, output_config: str, overrides_path: str = None):
        """
        生成 CIM 优化的 C 代码
        
        Args:
            output_c: C 代码输出路径
            output_weights: 权重二进制输出路径
            output_config: 配置JSON输出路径
            overrides_path: 覆盖配置文件路径 (JSON)
        """
        print("🔨 CIM 代码生成器")
        print("=" * 50)
        
        # 加载覆盖配置
        overrides = {}
        if overrides_path:
            import json
            import os
            if os.path.exists(overrides_path):
                try:
                    with open(overrides_path, 'r') as f:
                        overrides = json.load(f)
                    print(f"✓ 加载覆盖配置: {overrides_path}")
                except Exception as e:
                    print(f"⚠️ 无法加载覆盖配置: {e}")

        # 分析模型结构
        self.raw_layers = self._analyze_graph()
        
        # 2. RUN AUTO-PARTITIONING (New v2.0 Feature)
        from partitioner import GraphPartitioner
        partitioner = GraphPartitioner(self.raw_layers)
        self.partitions = partitioner.partition()
        partitioner.print_summary() 
        
        # Flatten partitions back to layers (with explicit 'target' set)
        # Detailed CodeGen for Hybrid Firmware is Phase 8.2
        # For now, we update layer['target'] based on partition results so users can see the decision.
        self.layers = []
        for partition in self.partitions:
            target = partition['target']
            for layer in partition['layers']:
                layer['target'] = target
                self.layers.append(layer)
        
        # 应用覆盖
        for i, layer in enumerate(self.layers):
            idx_str = str(i)
            if idx_str in overrides:
                target = overrides[idx_str]
                layer['target'] = target
                print(f"  -> Layer {i} ({layer['name']}) 覆盖为: {target}")
        
        print(f"✓ 分析模型: {len(self.layers)} 层")
        
        # 生成代码
        self._generate_header()
        self._generate_inference_function(self.layers)
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
        # Important: Store config for subsequent steps (e.g. firmware generation)
        self.config = self._generate_config(self.layers)
        
        import json
        with open(output_config, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✓ 生成配置: {output_config}")
        
        return output_c
    
    def _analyze_graph(self) -> List[Dict]:
        """分析计算图，提取层信息"""
        print(f"[Debug] Analyzing Graph... Node Count: {len(self.graph.node)}")
        layers = []
        
        for node in self.graph.node:
            print(f"[Debug] Node: {node.name} Type: {node.op_type}")
            # Check if it's a Transformer op first
            transformer_layer = self.transformer_mapper.analyze_node(node)
            if transformer_layer:
                layers.append(transformer_layer)
                continue

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
        # 1. Run Memory Planning
        planner = MemoryPlanner(layers)
        allocations, total_size = planner.plan()
        
        self.code_lines.extend([
            "/**",
            " * @brief 模型推理函数 (Optimized Memory Layout)",
            f" * Required Scratchpad Size: {total_size} bytes",
            " */",
            "int model_inference_optimized(const float *input, float *output, void *context) {",
            "    InferenceContext_t *ctx = (InferenceContext_t*)context;",
            "    uint8_t *heap = (uint8_t*)ctx->temp_buffer;",
            "    "
        ])
        
        # 2. Generate Code using allocated offsets
        # Helper to get variable name (pointer to heap offset)
        def get_tensor_ptr(tensor_name, var_name, is_const=False):
            if tensor_name in allocations:
                offset = allocations[tensor_name]
                prefix = "const " if is_const else ""
                return f"{prefix}float *{var_name} = ({prefix}float*)(heap + {offset});"
            else:
                # Fallback for external inputs/outputs or unallocated
                # Ideally inputs/outputs are separate arguments
                return f"// {var_name} maps to external {tensor_name}"

        # Initialize pointers for each intermediate tensor is too verbose in C
        # Instead, we cast on the fly or define macros.
        # Let's use macros for readability at the top of function?
        # Or just pointer arithmetic inline.
        
        # Mapping: Layer Output Name -> Allocation Offset
        
        previous_output = "input" 
        
        for i, layer in enumerate(layers):
            # Input Name
            input_name = layer['inputs'][0]
            # Output Name
            output_name = layer['outputs'][0]
            
            # Determine C source variable for Input
            if i == 0:
                c_input = "input"
            else:
                # Find the offset for this input tensor
                # Note: 'previous_output' logic assumes linear chain, 
                # but graph nodes explicitly state inputs.
                # Use the allocation map!
                if input_name in allocations:
                    c_input = f"((const float*)(heap + {allocations[input_name]}))"
                else:
                    c_input = "input" # Fallback
            
            # Determine C source variable for Output
            if i == len(layers) - 1:
                c_output = "output"
            else:
                if output_name in allocations:
                    c_output = f"((float*)(heap + {allocations[output_name]}))"
                else:
                    c_output = "output" # Should not happen mid-layer
            
            # Residual Handling (Add Layer)
            if layer['op_type'] == 'Add' or layer['type'] == 'ResidualLayerNorm':
                 # Needs second input
                 if len(layer['inputs']) > 1:
                     res_name = layer['inputs'][1]
                     if res_name in allocations:
                         c_residual = f"((const float*)(heap + {allocations[res_name]}))"
                     else:
                         c_residual = "input" # fallback
            
            # Generate Layer Code with new pointers
            target = layer.get('target', 'cim')
            
            if target == 'cpu':
                # Generate CPU Fallback Call
                # We assume a generic RISC-V function: RISCV_Compute(op_type, inputs..., output)
                self.code_lines.extend([
                    f"    // Layer {i} [CPU Fallback]",
                    f"    RISCV_Compute_Fallback(",
                    f"        \"{layer['op_type']}\",",
                    f"        {c_input}, {c_output}, layer_size",
                    f"    );",
                    ""
                ])
                # Skip CIM generation
                continue

            # CIM Generation
            if layer['type'] == 'fc':
                self._generate_fc_layer(layer, c_input, c_output, i)
            elif layer['type'] == 'lstm':
                self._generate_lstm_layer(layer, c_input, c_output, i)
            elif layer['type'] == 'activation':
                if layer.get('activation') == 'gelu':
                     code = self.transformer_mapper.generate_c_code(layer, c_input, c_output)
                     self.code_lines.extend(code)
                     self.code_lines.append("")
                else:
                    self._generate_activation(layer, c_input, c_output)
            elif layer['type'] in ['softmax', 'layernorm']:
                 code = self.transformer_mapper.generate_c_code(layer, c_input, c_output)
                 self.code_lines.extend(code)
                 self.code_lines.append("")
            
            # Special case for ResidualLayerNorm (Fused)
            elif layer.get('op_type') == 'ResidualLayerNorm':
                 # This is a custom fused op we created in optimizer.py
                 # Need to implement the C call here manually since it's not in standard codegen
                 # c_input is input1, c_residual is input2
                 self.code_lines.append(f"    // ResidualLayerNorm (Fused)")
                 # Assuming we have a CIM kernel or just naive C for now
                 self.code_lines.append(f"    CIM_ResidualLayerNorm({c_input}, {c_residual}, {c_output}, layer_size);")
                 self.code_lines.append("")

        self.code_lines.extend([
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


    def generate_firmware(self, output_firmware: str, driver_src: str = None):
        """
        生成完整的 SoC 固件 (main.c)
        集成用户驱动和自动推理循环
        """
        code = [
            "/**",
            " * Hive-Reflex SoC Main Firmware",
            " * Auto-generated by Silicon Compiler",
            " */",
            "",
            '#include "imc22_cim.h"',
            '#include "device_interface.h"',
            '#include <stdio.h>',
            '#include <stddef.h>',
            "",
            "// External Inference Function (Generated by codegen_cim.py)",
            "extern int model_inference_optimized(const float *input, float *output, void *context);",
            "",
            "// Inference Context (SRAM Buffer)",
            "static uint8_t sram_heap[1024 * 64]; // 64KB Heap",
            "static InferenceContext_t ctx = { .temp_buffer = (float*)sram_heap };",
            "",
            "// IO Buffers",
            "static float input_tensor[256];",
            "static float output_tensor[256];",
            "",
            "void main(void) {",
            "    printf(\"[SoC] Booting Hive-Reflex Core...\\n\");",
            "",
            "    // 1. Initialize User Device",
            "    if (HEx_Device_Init() != 0) {",
            "        printf(\"[SoC] Device Init Failed!\\n\");",
            "        while(1);",
            "    }",
            "    printf(\"[SoC] Device Ready.\\n\");",
            "",
            "    while(1) {",
            "        // 2. Read Sensors",
            "        int samples = HEx_Device_Read(input_tensor, sizeof(input_tensor));",
            "        if (samples < 0) continue;",
            "",
            "        // 3. Run Inference",
            "        // Note: In real SoC, we might sleep here until sensor interrupt",
            "        int ret = model_inference_optimized(input_tensor, output_tensor, &ctx);",
            "",
            "        // 4. Actuate",
            "        if (ret == 0) {",
            "            HEx_Device_Act(output_tensor, sizeof(output_tensor));",
            "        }",
            "    }",
            "}",
        ]
        
        with open(output_firmware, 'w') as f:
            f.write('\n'.join(code))
        print(f"✓ 生成 SoC 固件: {output_firmware}")

    def generate_python_firmware(self, output_path, driver_path=None):
        """
        Generate a Python 'Digital Twin' of the firmware for Host Emulation.
        This allows testing the SoC logic (Sensor -> Inference -> Actuation) on PC.
        """
        code = []
        code.append("#!/usr/bin/env python3")
        code.append('"""')
        code.append("Digital Twin Firmware for Hive-Reflex SoC")
        code.append("Generated by codegen_cim.py")
        code.append('"""')
        code.append("import sys")
        code.append("import os")
        code.append("import numpy as np")
        code.append("import time")
        
        # Add path for simulator
        code.append("# Ensure simulator is importable")
        code.append("sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulator'))")
        code.append("sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # fallback")
        code.append("try:")
        code.append("    from simulator.sim_reflex import SimReflex")
        code.append("except ImportError:")
        code.append("    from sim_reflex import SimReflex")
        
        # Constants
        code.append(f"\n# Memory Constraints ({self.config.get('SRAM_SIZE', 64)}KB)")
        code.append("SRAM_SIZE = 64 * 1024")
        code.append("sim = SimReflex(sram_size_kb=64)")
        
        # Calibration Function (Accuracy Recovery)
        code.append("\ndef calibrate_system():")
        code.append("    print('[SoC-FW] Starting On-Chip Calibration...')")
        code.append("    # Simulate hardware check")
        code.append("    # In real HW: run known pattern through CIM, check ADC offset")
        code.append("    time.sleep(0.1)") 
        code.append("    # Apply compensation")
        code.append("    sim.registers['CIM_CALIB'] = 0xAA") # Dummy register
        code.append("    print('[SoC-FW] Calibration Complete. Offset adjusted.')")

        # Inference Function
        code.append("\ndef run_inference(input_data):")
        code.append("    print('[SoC-FW] Starting Inference...')")
        code.append("    print(f'  > Input Shape: {input_data.shape}')")
        code.append("    start_t = time.time()")
        
        # Generate Layer Calls (Simulated)
        curr_in = 0
        curr_out = 16384 # 0x4000
        
        # Load Input
        code.append("    # Write Input to SRAM")
        code.append("    sim.mem_write(0, input_data.tobytes())")
        
        for layer in self.layers:
            if layer.get('type') == 'fc':
                 # "(Out, In)" string -> Tuple
                 shape_raw = layer.get('shape', (10, 10))
                 if isinstance(shape_raw, str):
                     shape = eval(shape_raw)
                 else:
                     shape = shape_raw
                     
                 out_dim, in_dim = shape
                 code.append(f"    # Layer: {layer['name']} ({shape})")
                 code.append(f"    sim.cim_fully_connected({curr_in}, {curr_out}, 0, {in_dim}, {out_dim}, relu=1)")
                 
                 # Swap
                 curr_in, curr_out = curr_out, curr_in

        code.append("    end_t = time.time()")
        code.append("    print(f'[SoC-FW] Inference Done. Time: {(end_t-start_t)*1000:.2f}ms')")
        
        # Robust Shape Handling
        if self.layers:
            last_shape_raw = self.layers[-1].get('shape', (10, 10))
            if isinstance(last_shape_raw, str):
                last_shape = eval(last_shape_raw)
            else:
                last_shape = last_shape_raw
            out_size = last_shape[0] * 4
        else:
            code.append("    # Warning: No layers detected")
            last_shape = (10, 10)
            out_size = 40
            
        code.append(f"    # Read Output (Assuming last layer size {last_shape})")
        code.append(f"    output_bytes = sim.mem_read({curr_in}, {out_size})")
        code.append("    output = np.frombuffer(output_bytes, dtype=np.float32)")
        code.append("    return output")

        # Main Loop (Mocking the C main)
        code.append("\ndef main():")
        code.append("    print('--- Hive-Reflex Digital Twin Booting ---')")
        code.append("    sim.stats['cycles'] = 0")
        code.append("    ")
        code.append("    # 0. System Calibration")
        code.append("    calibrate_system()")
        code.append("    ")
        code.append("    # Mock Sensor Loop")
        code.append("    for i in range(3):")
        code.append("        print(f'\\n[SoC-FW] Iteration {i}')")
        code.append("        # 1. Read Sensor (Mock)")
        code.append("        print('  > Reading Sensors...')")
        
        # Assume input size based on first layer
        if self.layers:
            # Safe access
            shape_raw = self.layers[0].get('shape', (10, 10))
            if isinstance(shape_raw, str):
                first_in_dim = eval(shape_raw)[1]
            else:
                 first_in_dim = shape_raw[1]
        else:
            first_in_dim = 16
            
        code.append(f"        input_sample = np.random.randn({first_in_dim}).astype(np.float32)")
        code.append("        ")
        code.append("        # 2. Run Inference")
        code.append("        result = run_inference(input_sample)")
        code.append("        ")
        code.append("        # 3. Actuate (Mock)")
        code.append("        pred = np.argmax(result)")
        code.append("        print(f'  > Actuation: Class {pred} (Conf: {result[pred]:.2f})')")
        code.append("    ")
        code.append("    print('\\n[SoC-FW] System Halted.')")
        code.append("    print(f'Total Simulated Cycles: {int(sim.stats[\"cycles\"])}')")

        code.append("\nif __name__ == '__main__':")
        code.append("    main()")

        with open(output_path, 'w') as f:
            f.write('\n'.join(code))
        print(f"✓ Generated Digital Twin: {output_path}")

    def generate_rtl_config(self, output_path, io_config_path=None):
        """
        Generate RTL configuration file (soc_config.vh)
        """
        if self.peak_memory == 0:
             self._perform_memory_planning()
             
        pruner = RTLPruner(self.layers, self.peak_memory)
        pruner.analyze()
        if io_config_path:
            pruner.parse_io_config(io_config_path)
        pruner.generate_config(output_path)

def main():
    """主函数"""
    import argparse
    import onnx
    
    parser = argparse.ArgumentParser(description='CIM 代码生成器')
    parser.add_argument('--model', required=True, help='ONNX 模型路径')
    parser.add_argument('--output-c', default='generated_inference.c', help='C 代码输出')
    parser.add_argument('--output-weights', default='generated_weights.bin', help='权重输出')
    parser.add_argument('--output-config', default='model_config.json', help='配置输出')
    parser.add_argument('--overrides', help='覆盖配置文件路径', default=None)
    parser.add_argument('--output-firmware', help='输出完整 SoC 固件路径 (main.c)', default=None)
    parser.add_argument('--output-python', help='输出 Python Digital Twin (firmware.py)', default=None)
    parser.add_argument('--driver', help='用户驱动源文件路径 (用于验证)', default=None)
    parser.add_argument('--prune-rtl', help='RTL裁剪配置输出路径 (soc_config.vh)', default=None)
    parser.add_argument('--io-config', help='IO 配置 JSON 路径 (e.g. devices.json)', default=None)
    
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        model = onnx.load(args.model)
        # Check if transformer needed
        # transformer_mapper = TransformerMapper()
        # graph_init = model.graph.initializer
        # layers = parse_onnx_model(model, transformer_mapper, graph_init)
        # generator = CIMCodeGenerator(layers)
        # (Simplification: reusing existing structure logic)
        # We need to adapt because previous refactor changed CIMCodeGenerator init?
        # Let's check init signature
        pass
    else:
        print(f"⚠️ Model file {args.model} not found. Using MOCK model for Demo.")
        # Force Mock
        from unittest.mock import MagicMock
        model = MagicMock() 
        # Populate Dummy Graph for code generation to succeed
        node = MagicMock()
        node.name = "fc1"
        node.op_type = "Gemm"
        node.input = ["in"]
        node.output = ["out"]
        # Make attribute empty
        node.attribute = []
        
        # Setup graph
        model.graph.node = [node]
        model.graph.initializer = [] # No weights, but handled gracefully?
        
    generator = CIMCodeGenerator(model)
    generator.generate(args.output_c, args.output_weights, args.output_config, args.overrides)
    
    # Optional: Generate full firmware
    if args.output_firmware:
        generator.generate_firmware(args.output_firmware, args.driver)
        
    # Optional: Generate Python Digital Twin
    if args.output_python:
        generator.generate_python_firmware(args.output_python)
        
    # Optional: Generate RTL Config
    if args.prune_rtl:
        generator.generate_rtl_config(args.prune_rtl, args.io_config)
    
    print("\n✅ 代码生成完成!")


if __name__ == '__main__':
    main()
