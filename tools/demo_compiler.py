#!/usr/bin/env python3
"""
模型编译工具链演示 - 展示完整工作流程
"""

import sys
sys.path.append('.')

from model_parser import ModelParser
from quantizer import Quantizer

print("="*70)
print(" Hive-Reflex 模型编译工具链演示")
print("="*70)

# Step 1: 解析模型（使用模拟数据）
print("\n[1/3] 解析模型...")
parser = ModelParser()
model = parser.load_onnx("dummy.onnx")  # 会自动使用模拟数据
print(f"  ✓ 模型包含 {len(model['layers'])} 层")
print(f"  ✓ 总参数量: {model['total_params']:,}")
print(f"  ✓ 原始大小: {model['original_size'] / 1024 / 1024:.2f} MB")

# Step 2: 量化
print("\n[2/3] 量化 (INT8)...")
quantizer = Quantizer()
model = quantizer.quantize_model(model, 'int8')
print(f"  ✓ 量化后大小: {model['quantized_size'] / 1024 / 1024:.2f} MB")
reduction = (1 - model['quantized_size']/model['original_size'])*100
print(f"  ✓ 大小减少: {reduction:.1f}%")

# Step 3: 生成输出
print("\n[3/3] 生成输出文件...")

# 生成 C 头文件
header_content = f"""#ifndef MODEL_GENERATED_H
#define MODEL_GENERATED_H

// 模型元数据
#define MODEL_NUM_LAYERS {len(model['layers'])}
#define MODEL_TOTAL_SIZE {model['quantized_size']}

// 层配置
typedef struct {{
    uint32_t flash_addr;
    uint32_t size;
}} LayerConfig_t;

// Flash 地址映射
"""

addr = 0x08000000
for i in range(len(model['layers'])):
    size = model['layers'][i]['size'] // 4  # INT8
    header_content += f"#define LAYER_{i}_ADDR 0x{addr:08X}\n"
    header_content += f"#define LAYER_{i}_SIZE {size}\n"
    addr += (size + 255) & ~255

header_content += "\n#endif // MODEL_GENERATED_H\n"

with open('test_model.h', 'w') as f:
    f.write(header_content)

print(f"  ✓ 生成: test_model.h")

# 生成 Flash 映射 JSON
import json
flash_map = {
    'model_name': 'demo_model',
    'num_layers': len(model['layers']),
    'quantization': 'int8',
    'original_size_mb': model['original_size'] / 1024 / 1024,
    'quantized_size_mb': model['quantized_size'] / 1024 / 1024,
    'compression_ratio': model['original_size'] / model['quantized_size']
}

with open('test_model_layout.json', 'w') as f:
    json.dump(flash_map, f, indent=2)

print(f"  ✓ 生成: test_model_layout.json")

# 总结
print("\n" + "="*70)
print(" 编译完成!")
print("="*70)
print(f"原始大小:   {model['original_size'] / 1024 / 1024:.2f} MB")
print(f"量化后:     {model['quantized_size'] / 1024 / 1024:.2f} MB (INT8)")
print(f"总减少:     {reduction:.1f}%")
print("="*70)
print(f"\n生成文件:")
print(f"  - test_model.h           (C 头文件)")
print(f"  - test_model_layout.json (Flash 映射)")
print("="*70)
