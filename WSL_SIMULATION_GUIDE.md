# IMC-22 芯片 WSL 仿真验证指南

**环境**: Windows Subsystem for Linux (WSL2)  
**完整仿真**: QEMU + RISC-V 工具链  
**预计时间**: 2-3 小时设置 + 10-15 天验证

---

## 🚀 快速开始

### 步骤 1: 安装 WSL2（如未安装）

```powershell
# 在 PowerShell（管理员）中运行
wsl --install -d Ubuntu
# 或
wsl --install -d Ubuntu-22.04

# 重启计算机后，设置 Ubuntu 用户名和密码
```

### 步骤 2: 验证 WSL 安装

```powershell
# 检查 WSL 版本
wsl --list --verbose

# 应该显示:
#   NAME            STATE           VERSION
# * Ubuntu          Running         2
```

### 步骤 3: 进入 WSL

```powershell
wsl
# 现在您在 Ubuntu Linux 环境中
```

---

## 📦 安装工具链（在 WSL 中）

### 1. 更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装 RISC-V 工具链

```bash
# 安装预编译的 RISC-V GCC
sudo apt install -y gcc-riscv64-unknown-elf

# 或安装通用版本
sudo apt install -y gcc-riscv64-linux-gnu

# 验证安装
riscv64-unknown-elf-gcc --version
```

**如果上述安装失败**，手动安装：

```bash
# 下载预编译工具链
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.11.20/riscv32-elf-ubuntu-22.04-gcc-nightly-2023.11.20-nightly.tar.gz

# 解压
tar -xzf riscv32-elf-ubuntu-22.04-gcc-nightly-2023.11.20-nightly.tar.gz

# 移动到 /opt
sudo mv riscv /opt/

# 添加到 PATH
echo 'export PATH=/opt/riscv/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
riscv32-unknown-elf-gcc --version
```

### 3. 安装 QEMU

```bash
sudo apt install -y qemu-system-riscv32 qemu-system-riscv64

# 验证安装
qemu-system-riscv32 --version
```

### 4. 安装其他必需工具

```bash
sudo apt install -y make git python3 python3-pip
pip3 install numpy
```

---

## 🔧 配置项目

### 1. 访问 Windows 文件

在 WSL 中，Windows 文件位于 `/mnt/` 下：

```bash
# 进入项目目录
cd /mnt/d/新建文件夹/hive-reflex

# 列出文件
ls -la
```

### 2. 修改 Makefile（适配 WSL）

创建 `Makefile.wsl`:

```makefile
# 工具链配置（使用 WSL 路径）
CROSS_COMPILE = riscv32-unknown-elf-
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump
SIZE = $(CROSS_COMPILE)size

# QEMU 配置
QEMU = qemu-system-riscv32
QEMU_MACHINE = virt
QEMU_FLAGS = -nographic -machine $(QEMU_MACHINE)

# 项目配置
TARGET = hive_node
SDK_DIR = imc22_sdk
BUILD_DIR = build

# 编译选项
ARCH_FLAGS = -march=rv32imac -mabi=ilp32
OPT_FLAGS = -O2 -g
WARN_FLAGS = -Wall -Wextra

CFLAGS = $(ARCH_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) \
         -I$(SDK_DIR) \
         -ffunction-sections -fdata-sections \
         -ffreestanding

LDFLAGS = $(ARCH_FLAGS) \
          -T $(SDK_DIR)/linker.ld \
          -nostartfiles \
          -Wl,--gc-sections \
          -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

# 源文件
SDK_SRCS = $(SDK_DIR)/startup.c \
           $(SDK_DIR)/imc22_can.c \
           $(SDK_DIR)/imc22_npu.c

APP_SRCS ?= hive_node_ctrl.c

SRCS = $(SDK_SRCS) $(APP_SRCS)
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)

# 默认目标
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).bin
	@echo "Build complete!"
	@$(SIZE) $(BUILD_DIR)/$(TARGET).elf

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/$(SDK_DIR)

# 编译规则
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo "CC $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# 链接规则
$(BUILD_DIR)/$(TARGET).elf: $(OBJS)
	@echo "LD $@"
	@$(CC) $(LDFLAGS) $(OBJS) -o $@

# 生成二进制文件
$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	@echo "OBJCOPY $@"
	@$(OBJCOPY) -O binary $< $@

# 仿真运行
.PHONY: sim
sim: $(BUILD_DIR)/$(TARGET).elf
	@echo "Starting QEMU simulation..."
	$(QEMU) $(QEMU_FLAGS) -kernel $(BUILD_DIR)/$(TARGET).elf

# 清理
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean sim
```

---

## ✅ 验证环境

### 测试编译

```bash
cd /mnt/d/新建文件夹/hive-reflex

# 使用 WSL Makefile
make -f Makefile.wsl clean
make -f Makefile.wsl

# 应该看到:
# CC startup.c
# CC imc22_can.c
# CC imc22_npu.c
# CC hive_node_ctrl.c
# LD build/hive_node.elf
# OBJCOPY build/hive_node.bin
# Build complete!
```

### 测试仿真

```bash
make -f Makefile.wsl sim

# QEMU 应该启动并运行代码
# 按 Ctrl+A 然后 X 退出 QEMU
```

---

## 🧪 运行测试套件

### 1. CAN 驱动测试

```bash
# 编译测试
make -f Makefile.wsl APP_SRCS=tests/test_can.c TARGET=test_can

# 运行仿真
make -f Makefile.wsl sim TARGET=test_can

# 预期输出:
# [PASS] CAN_Init should return 0
# [PASS] CAN_Send should return 0
# ...
# ✓ All tests PASSED!
```

### 2. 使用 Python 测试框架

```bash
# 运行所有测试
python3 tools/run_sim_tests.py --test-dir build

# 输出:
# Running: test_can.bin
# [PASS] ...
# ✓ test_can         PASS
# 
# Total: 3 | Passed: 3 | Failed: 0
```

---

## 📊 仿真验证流程

### 阶段 1: 基础验证（1天）

```bash
# 1. Hello World
make -f Makefile.wsl APP_SRCS=examples/example_hello.c TARGET=hello
make -f Makefile.wsl sim TARGET=hello

# 2. CAN 测试
make -f Makefile.wsl APP_SRCS=tests/test_can.c TARGET=test_can
make -f Makefile.wsl sim TARGET=test_can

# 3. NPU 测试
make -f Makefile.wsl APP_SRCS=tests/test_npu.c TARGET=test_npu
make -f Makefile.wsl sim TARGET=test_npu
```

### 阶段 2: 集成测试（2-3天）

```bash
# 完整控制循环
make -f Makefile.wsl
make -f Makefile.wsl sim

# 观察输出，验证:
# - CAN 初始化
# - NPU 加载
# - 控制循环运行
```

### 阶段 3: 性能测试（1-2天）

```bash
# 运行性能基准
make -f Makefile.wsl APP_SRCS=tests/benchmark.c TARGET=benchmark

# 带性能分析
qemu-system-riscv32 -nographic -machine virt \
    -kernel build/benchmark.elf \
    -d cpu,exec -D profile.log

# 分析性能
python3 tools/analyze_performance.py profile.log
```

---

## 🔍 调试技巧

### GDB 调试

```bash
# 终端 1: 启动 QEMU GDB 服务器
qemu-system-riscv32 -nographic -machine virt \
    -kernel build/hive_node.elf -s -S

# 终端 2: 连接 GDB
riscv32-unknown-elf-gdb build/hive_node.elf
(gdb) target remote localhost:1234
(gdb) break main
(gdb) continue
```

### 查看反汇编

```bash
riscv32-unknown-elf-objdump -d build/hive_node.elf > disasm.txt
less disasm.txt
```

---

## 📝 常见问题

### Q: 工具链找不到？
**A**: 检查 PATH 设置：
```bash
echo $PATH
which riscv32-unknown-elf-gcc
```

### Q: WSL 文件权限问题？
**A**: 在 WSL 中创建工作副本：
```bash
cp -r /mnt/d/新建文件夹/hive-reflex ~/hive-reflex
cd ~/hive-reflex
```

### Q: QEMU 仿真无输出？
**A**: 检查串口配置，可能需要添加 `-serial stdio`

---

## 🚀 下一步

### 立即执行（今天）

```bash
# 1. 安装 WSL（如未安装）
wsl --install -d Ubuntu

# 2. 在 WSL 中安装工具
sudo apt update
sudo apt install -y qemu-system-riscv32 make gcc-riscv64-unknown-elf

# 3. 进入项目并测试
cd /mnt/d/新建文件夹/hive-reflex
make -f Makefile.wsl
```

### 本周计划

- Day 1: 环境搭建和基础测试
- Day 2-3: CAN 和 NPU 驱动验证
- Day 4-5: 完整控制循环测试
- Day 6-7: 性能测试和报告

---

**创建日期**: 2026-01-17  
**预计完成**: 2026-01-24（7天基础验证）
