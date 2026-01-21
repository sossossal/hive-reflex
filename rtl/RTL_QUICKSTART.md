# Hive-Reflex 2.0 RTL 快速开始指南

## 🚀 开始 FPGA 验证

### 1. 准备开发板

**推荐配置：**
- FPGA: Xilinx ZCU102 或 Intel DE10-Nano
- 调试器: JTAG
- 串口: USB-UART

### 2. 安装工具

```bash
# Xilinx Vivado
wget https://www.xilinx.com/support/download.html

# 或 Intel Quartus
wget https://www.intel.com/programmable/downloads

# RISC-V 工具链
sudo apt-get install gcc-riscv64-unknown-elf
```

### 3. 编译 RTL

```tcl
# Vivado 脚本
create_project hive_reflex ./build -part xczu9eg-ffvb1156-2-e

# 添加源文件
add_files rtl/cim_mac_array.v
# ... 其他文件

# 综合
synth_design -top hive_reflex_top

# 实现
opt_design
place_design
route_design

# 生成比特流
write_bitstream hive_reflex.bit
```

### 4. 下载到 FPGA

```bash
# 使用 Vivado Hardware Manager
# 或命令行
vivado -mode batch -source program_fpga.tcl
```

### 5. 测试验证

```c
// 编译测试固件
make APP_SRCS=tests/test_cim.c

// 通过 JTAG 加载
openocd -f imc22.cfg
```

## 📊 预期结果

- 系统频率: 100 MHz ✓
- 推理延迟: <25 μs
- 资源利用: <70%

---

详细计划见 [hardware_validation_plan.md](file:///C:/Users/%E8%8D%A3%E8%80%80/.gemini/antigravity/brain/fcf659df-124f-41ad-9fe7-b48e2742b793/hardware_validation_plan.md)
