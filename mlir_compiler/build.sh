#!/bin/bash
# 完整的 MLIR 编译流程 (更新版)
# 包含优化 Pass

set -e

echo "🚀 Hive-Reflex 2.0 MLIR 编译器 (优化版)"
echo "========================================"

# 配置
MODEL_PATH=${1:-"../reflex_net_v2.onnx"}
OUTPUT_DIR="build"
TARGET="imc22"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# =====================================================================
# 步骤 1: ONNX 模型加载
# =====================================================================
echo ""
echo "步骤 1/6: 加载 ONNX 模型"
if [ ! -f "$MODEL_PATH" ]; then
    echo "  错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi
echo "  ✓ 模型: $MODEL_PATH"

# =====================================================================
# 步骤 2: 图优化 (算子融合、常量折叠)
# =====================================================================
echo ""
echo "步骤 2/6: 图优化"
python3 optimizer.py \
    --input $MODEL_PATH \
    --output $OUTPUT_DIR/model_optimized.onnx

if [ $? -ne 0 ]; then
    echo "  ✗ 优化失败"
    exit 1
fi

# =====================================================================
# 步骤 3: 量化优化
# =====================================================================
echo ""
echo "步骤 3/6: 量化优化"
python3 quantization.py \
    --model $OUTPUT_DIR/model_optimized.onnx \
    --output $OUTPUT_DIR/model_quantized.onnx \
    --dtype int8

if [ $? -ne 0 ]; then
    echo "  ⚠️  量化跳过 (可选)"
    cp $OUTPUT_DIR/model_optimized.onnx $OUTPUT_DIR/model_quantized.onnx
fi

# =====================================================================
# 步骤 4: CIM 代码生成
# =====================================================================
echo ""
echo "步骤 4/6: CIM 代码生成"
python3 codegen_cim.py \
    --model $OUTPUT_DIR/model_quantized.onnx \
    --output-c $OUTPUT_DIR/inference_optimized.c \
    --output-weights $OUTPUT_DIR/weights_optimized.bin \
    --output-config $OUTPUT_DIR/model_config.json

if [ $? -ne 0 ]; then
    echo "  ✗ 代码生成失败"
    exit 1
fi

# =====================================================================
# 步骤 5: 生成权重头文件
# =====================================================================
echo ""
echo "步骤 5/6: 生成权重头文件"
xxd -i $OUTPUT_DIR/weights_optimized.bin > $OUTPUT_DIR/weights_optimized.h

if [ $? -ne 0 ]; then
    echo "  ✗ 权重转换失败"
    exit 1
fi

echo "  ✓ 权重头文件: $OUTPUT_DIR/weights_optimized.h"

# =====================================================================
# 步骤 6: 性能报告
# =====================================================================
echo ""
echo "步骤 6/6: 生成性能报告"

# 计算文件大小
ORIGINAL_SIZE=$(wc -c < $MODEL_PATH)
OPTIMIZED_SIZE=$(wc -c < $OUTPUT_DIR/model_optimized.onnx)
WEIGHTS_SIZE=$(wc -c < $OUTPUT_DIR/weights_optimized.bin)

REDUCTION=$(echo "scale=2; (1 - $OPTIMIZED_SIZE / $ORIGINAL_SIZE) * 100" | bc)

echo ""
echo "性能报告:"
echo "  原始模型: $ORIGINAL_SIZE bytes"
echo "  优化模型: $OPTIMIZED_SIZE bytes"
echo "  权重大小: $WEIGHTS_SIZE bytes"
echo "  模型缩减: ${REDUCTION}%"

# 读取配置
if command -v jq &> /dev/null; then
    NUM_LAYERS=$(jq '.num_layers' $OUTPUT_DIR/model_config.json)
    echo "  层数: $NUM_LAYERS"
fi

# =====================================================================
# 完成
# =====================================================================
echo ""
echo "✅ MLIR 编译完成!"
echo ""
echo "生成的文件:"
echo "  📦 优化模型: $OUTPUT_DIR/model_optimized.onnx"
echo "  📦 量化模型: $OUTPUT_DIR/model_quantized.onnx"
echo "  📦 C 代码:   $OUTPUT_DIR/inference_optimized.c"
echo "  📦 权重:     $OUTPUT_DIR/weights_optimized.bin"
echo "  📦 配置:     $OUTPUT_DIR/model_config.json"
echo ""
echo "下一步:"
echo "  make APP_SRCS='$OUTPUT_DIR/inference_optimized.c'"
