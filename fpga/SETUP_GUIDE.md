# Hive-Reflex 2.0 FPGA 环境搭建指南

## 📋 概述

本指南将帮助您搭建完整的 FPGA 验证开发环境。

## 🖥️ 系统要求

### 硬件要求
- **CPU**: 4核心以上
- **内存**: 16GB+（推荐 32GB）
- **硬盘**: 100GB+ 可用空间
- **操作系统**: 
  - Ubuntu 20.04/22.04 LTS（推荐）
  - Windows 10/11 Pro
  - macOS（部分支持）

### FPGA 开发板
- **推荐**: Xilinx ZCU102
- **经济**: Intel DE10-Nano
- **学习**: Lattice iCE40

---

## 🚀 快速开始

### Linux (Ubuntu)

```bash
# 1. 下载脚本
cd d:/新建文件夹/hive-reflex/fpga

# 2. 给予执行权限
chmod +x setup_fpga_env.sh

# 3. 运行安装脚本
./setup_fpga_env.sh

# 4. 检查环境
./check_env.sh
```

### Windows

```powershell
# 以管理员身份运行 PowerShell

# 1. 进入目录
cd d:\新建文件夹\hive-reflex\fpga

# 2. 运行安装脚本
.\setup_fpga_env.ps1

# 3. 检查环境
.\check_env.ps1
```

---

## 📦 安装的工具

### 核心工具
| 工具 | 版本 | 用途 |
|------|------|------|
| **RISC-V GCC** | 12.2.0 | 交叉编译工具链 |
| **Verilator** | 5.020 | RTL 仿真器 |
| **GTKWave** | latest | 波形查看器 |
| **OpenOCD** | 0.12.0 | JTAG 调试 |
| **Python** | 3.8+ | 验证脚本 |

### Python 包
- `cocotb` - 硬件验证框架
- `pytest` - 测试框架
- `numpy` - 数值计算
- `matplotlib` - 数据可视化
- `pyserial` - 串口通信

### FPGA 工具 (手动安装)
- **Xilinx Vivado** 2023.2 (推荐)
- **Intel Quartus Prime** Lite (免费)

---

## 🔧 Vivado 安装

### 下载

访问 [Xilinx 下载页面](https://www.xilinx.com/support/download.html)

### Linux 安装

```bash
# 1. 下载 Vivado Web Installer
# 选择: Vivado ML Standard Edition

# 2. 运行安装程序
chmod +x Xilinx_Unified_*.bin
sudo ./Xilinx_Unified_*.bin

# 3. 选择组件
#    ✓ Vivado
#    ✓ Vitis (可选)

# 4. 安装路径
#    /opt/Xilinx

# 5. 添加到环境变量
echo 'source /opt/Xilinx/Vivado/2023.2/settings64.sh' >> ~/.bashrc
source ~/.bashrc
```

### Windows 安装

1. 运行 `Xilinx_Unified_*.exe`
2. 选择 Vivado ML Standard
3. 安装到 `C:\Xilinx`
4. 重启计算机

---

## 🧪 验证环境

### 运行检查脚本

```bash
# Linux
./check_env.sh

# Windows
.\check_env.ps1
```

### 预期输出

```
检查 FPGA 开发环境...

✓ riscv32-unknown-elf-gcc
✓ verilator
✓ gtkwave
✓ openocd
✓ python3

✅ 所有工具已就绪!
```

### 测试 RISC-V 工具链

```bash
# 编译测试程序
riscv32-unknown-elf-gcc --version

# 应该显示:
# riscv32-unknown-elf-gcc (xPack GNU RISC-V Embedded GCC...) 12.2.0
```

### 测试 Vivado

```bash
vivado -version

# 应该显示:
# Vivado v2023.2 (64-bit)
```

---

## 🎯 创建第一个项目

### 1. 创建 Vivado 项目

```bash
cd d:/新建文件夹/hive-reflex/fpga/vivado

vivado -mode batch -source create_project.tcl
```

### 2. 综合和实现

```bash
vivado -mode batch -source build.tcl
```

### 3. 查看结果

```bash
# 比特流
ls -lh ./output/*.bit

# 资源利用率报告
cat ./reports/utilization_impl.txt

# 时序报告
cat ./reports/timing_impl.txt
```

---

## 📊 项目结构

```
fpga/
├── setup_fpga_env.sh         # Linux 安装脚本
├── setup_fpga_env.ps1         # Windows 安装脚本
├── check_env.sh               # 环境检查(Linux)
├── check_env.ps1              # 环境检查(Windows)
│
├── vivado/                    # Vivado 项目
│   ├── create_project.tcl     # 创建项目
│   ├── build.tcl              # 构建脚本
│   └── vivado_project/        # 项目目录(自动生成)
│
├── constraints/               # 约束文件
│   └── zcu102.xdc            # ZCU102 引脚约束
│
└── sim/                       # 仿真文件
    └── testbench.v
```

---

## 🐛 故障排除

### 问题 1: RISC-V 工具链无法找到

**解决方案:**
```bash
# Linux
export PATH="/opt/riscv/bin:$PATH"
source ~/.bashrc

# Windows
# 添加 C:\riscv\bin 到系统 PATH
```

### 问题 2: Vivado 许可证问题

**解决方案:**
1. 注册 Xilinx 账号
2. 生成免费许可证 (Webpack)
3. 安装许可证文件

### 问题 3: USB 权限问题 (Linux)

**解决方案:**
```bash
# 添加用户到 dialout 组
sudo usermod -a -G dialout $USER

# 重新登录
```

### 问题 4: OpenOCD 无法连接

**解决方案:**
```bash
# 检查 USB 设备
lsusb

# 检查权限
sudo chmod 666 /dev/bus/usb/*/*
```

---

## 📚 下一步

环境搭建完成后，您可以：

1. **开始 RTL 开发** - 参考 [hardware_validation_plan.md](file:///C:/Users/%E8%8D%A3%E8%80%80/.gemini/antigravity/brain/fcf659df-124f-41ad-9fe7-b48e2742b793/hardware_validation_plan.md)
2. **运行仿真** - 参考 Week 1 任务
3. **综合和实现** - 使用提供的 TCL 脚本
4. **烧录测试** - 连接 FPGA 开发板

---

## 🔗 参考链接

- [Xilinx Vivado 文档](https://www.xilinx.com/support/documentation.html)
- [Verilator 手册](https://verilator.org/guide/latest/)
- [Cocotb 文档](https://docs.cocotb.org/)
- [Rocket Chip Wiki](https://github.com/chipsalliance/rocket-chip/wiki)

---

**版本**: 1.0  
**更新**: 2026-01-19
