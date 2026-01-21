#!/bin/bash
# Hive-Reflex 2.0 FPGA 环境自动搭建脚本
# 适用于 Ubuntu 20.04/22.04

set -e

echo "╔════════════════════════════════════════════╗"
echo "║  Hive-Reflex 2.0 FPGA 环境搭建             ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ 检测到 Linux 系统"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ 检测到 macOS 系统"
else
    echo "✗ 不支持的操作系统: $OSTYPE"
    exit 1
fi

# 创建工作目录
WORK_DIR="$HOME/fpga_dev"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo ""
echo "工作目录: $WORK_DIR"
echo ""

# ========================================================================
# 步骤 1: 安装基础工具
# ========================================================================
echo "步骤 1/7: 安装基础开发工具"
echo "-----------------------------------"

sudo apt-get update

# 基础工具
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libusb-1.0-0-dev \
    libftdi1-dev

echo "✓ 基础工具安装完成"

# ========================================================================
# 步骤 2: 安装 RISC-V 工具链
# ========================================================================
echo ""
echo "步骤 2/7: 安装 RISC-V 工具链"
echo "-----------------------------------"

if ! command -v riscv32-unknown-elf-gcc &> /dev/null; then
    echo "下载 RISC-V 工具链..."
    
    # 使用预编译版本
    RISCV_TOOLCHAIN_URL="https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v12.2.0-3/xpack-riscv-none-elf-gcc-12.2.0-3-linux-x64.tar.gz"
    
    wget $RISCV_TOOLCHAIN_URL -O riscv-toolchain.tar.gz
    tar -xzf riscv-toolchain.tar.gz
    
    # 移动到系统目录
    sudo mv xpack-riscv-none-elf-gcc-* /opt/riscv
    
    # 添加到 PATH
    echo 'export PATH="/opt/riscv/bin:$PATH"' >> ~/.bashrc
    export PATH="/opt/riscv/bin:$PATH"
    
    rm riscv-toolchain.tar.gz
    
    echo "✓ RISC-V 工具链安装完成"
else
    echo "✓ RISC-V 工具链已安装"
fi

# 验证
riscv32-unknown-elf-gcc --version

# ========================================================================
# 步骤 3: 安装 Verilator (开源仿真器)
# ========================================================================
echo ""
echo "步骤 3/7: 安装 Verilator"
echo "-----------------------------------"

if ! command -v verilator &> /dev/null; then
    echo "编译安装 Verilator..."
    
    git clone https://github.com/verilator/verilator
    cd verilator
    git checkout v5.020
    
    autoconf
    ./configure
    make -j$(nproc)
    sudo make install
    
    cd ..
    
    echo "✓ Verilator 安装完成"
else
    echo "✓ Verilator 已安装"
fi

verilator --version

# ========================================================================
# 步骤 4: 安装 GTKWave (波形查看器)
# ========================================================================
echo ""
echo "步骤 4/7: 安装 GTKWave"
echo "-----------------------------------"

sudo apt-get install -y gtkwave

echo "✓ GTKWave 安装完成"

# ========================================================================
# 步骤 5: 安装 Python 验证工具
# ========================================================================
echo ""
echo "步骤 5/7: 安装 Python 验证工具"
echo "-----------------------------------"

pip3 install --user \
    cocotb \
    pytest \
    numpy \
    matplotlib \
    pyserial

echo "✓ Python 工具安装完成"

# ========================================================================
# 步骤 6: 安装 OpenOCD (JTAG 调试)
# ========================================================================
echo ""
echo "步骤 6/7: 安装 OpenOCD"
echo "-----------------------------------"

if ! command -v openocd &> /dev/null; then
    echo "编译安装 OpenOCD..."
    
    git clone https://github.com/openocd-org/openocd.git
    cd openocd
    
    ./bootstrap
    ./configure --enable-ftdi --enable-jlink
    make -j$(nproc)
    sudo make install
    
    cd ..
    
    echo "✓ OpenOCD 安装完成"
else
    echo "✓ OpenOCD 已安装"
fi

openocd --version

# ========================================================================
# 步骤 7: 克隆 Rocket Chip (RISC-V 软核)
# ========================================================================
echo ""
echo "步骤 7/7: 克隆 Rocket Chip"
echo "-----------------------------------"

if [ ! -d "rocket-chip" ]; then
    echo "克隆 Rocket Chip..."
    git clone https://github.com/chipsalliance/rocket-chip.git
    
    cd rocket-chip
    git submodule update --init --recursive
    cd ..
    
    echo "✓ Rocket Chip 克隆完成"
else
    echo "✓ Rocket Chip 已存在"
fi

# ========================================================================
# 完成
# ========================================================================
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  ✅ FPGA 开发环境搭建完成!                 ║"
echo "╚════════════════════════════════════════════╝"
echo ""

echo "已安装的工具:"
echo "  ✓ RISC-V GCC $(riscv32-unknown-elf-gcc --version | head -n1)"
echo "  ✓ Verilator $(verilator --version | head -n1)"
echo "  ✓ GTKWave $(gtkwave --version 2>&1 | head -n1)"
echo "  ✓ OpenOCD $(openocd --version 2>&1 | head -n1)"
echo "  ✓ Python $(python3 --version)"
echo ""

echo "工作目录: $WORK_DIR"
echo ""

echo "下一步:"
echo "  1. 安装 Xilinx Vivado (需要手动下载)"
echo "     下载地址: https://www.xilinx.com/support/download.html"
echo ""
echo "  2. 或安装 Intel Quartus Prime (免费版)"
echo "     下载地址: https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/download.html"
echo ""
echo "  3. 重启终端使 PATH 生效"
echo "     source ~/.bashrc"
echo ""
echo "  4. 开始 FPGA 开发!"
echo ""

# 创建环境检查脚本
cat > check_env.sh << 'EOF'
#!/bin/bash
echo "检查 FPGA 开发环境..."
echo ""

# 检查各工具
tools=(
    "riscv32-unknown-elf-gcc"
    "verilator"
    "gtkwave"
    "openocd"
    "python3"
)

all_ok=true

for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "✓ $tool"
    else
        echo "✗ $tool (未安装)"
        all_ok=false
    fi
done

echo ""

if [ "$all_ok" = true ]; then
    echo "✅ 所有工具已就绪!"
else
    echo "⚠️  部分工具缺失"
fi
EOF

chmod +x check_env.sh

echo "环境检查脚本已创建: ./check_env.sh"
