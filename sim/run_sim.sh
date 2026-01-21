#!/bin/bash
# Verilator 仿真脚本 - MAC 单元测试

set -e

echo "=========================================="
echo "MAC Unit Verilator 仿真"
echo "=========================================="
echo ""

# 创建构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# RTL 文件
RTL_FILES="../rtl/mac_unit.v"
TB_FILE="mac_unit_tb.v"

echo "步骤 1/3: 编译 RTL"
echo "------------------------------------------"

# 使用 Verilator 编译
verilator --cc --exe --build -j 4 \
    --trace \
    --Mdir $BUILD_DIR \
    -Wall \
    $RTL_FILES $TB_FILE \
    -CFLAGS "-std=c++11" \
    -o mac_unit_sim

if [ $? -eq 0 ]; then
    echo "✓ 编译成功"
else
    echo "✗ 编译失败"
    exit 1
fi

echo ""
echo "步骤 2/3: 运行仿真"
echo "------------------------------------------"

# 运行仿真
./$BUILD_DIR/mac_unit_sim

if [ $? -eq 0 ]; then
    echo "✓ 仿真完成"
else
    echo "✗ 仿真失败"
    exit 1
fi

echo ""
echo "步骤 3/3: 查看波形"
echo "------------------------------------------"

if [ -f "mac_unit_tb.vcd" ]; then
    echo "✓ 波形文件生成: mac_unit_tb.vcd"
    echo ""
    echo "使用 GTKWave 查看:"
    echo "  gtkwave mac_unit_tb.vcd"
else
    echo "⚠ 未生成波形文件"
fi

echo ""
echo "=========================================="
echo "仿真完成!"
echo "=========================================="
