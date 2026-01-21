# Windows 环境 IMC-22 仿真指南

**环境**: Windows 10/11 + QEMU + RISC-V 工具链  
**无需 WSL**: 直接在 Windows 上运行  
**设置时间**: 约 30 分钟

---

## 🚀 快速开始

### 步骤 1: 下载 RISC-V GCC 工具链

#### 选项 A: xPack RISC-V GCC（推荐）

```powershell
# 下载地址
# https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases

# 下载最新版本（例如 13.2.0-2）
# riscv-none-elf-gcc-13.2.0-2-win32-x64.zip

# 解压到
C:\Tools\riscv-gcc\
```

#### 选项 B: SiFive 预编译工具链

```powershell
# 下载地址
# https://www.sifive.com/software

# 解压后添加到 PATH
```

### 步骤 2: 添加到环境变量

```powershell
# 添加工具链到 PATH
$env:PATH += ";C:\Tools\riscv-gcc\bin"

# 永久添加（系统设置 → 环境变量 → Path）
# 或在 PowerShell Profile 中添加
```

### 步骤 3: 验证安装

```powershell
# 验证 GCC
riscv-none-elf-gcc --version

# 验证 QEMU（已安装）
& "C:\Program Files\qemu\qemu-system-riscv32.exe" --version
```

---

## 🔧 项目配置

### 创建 Windows Makefile

文件：`Makefile.windows`

```makefile
# Windows 环境配置
TOOLCHAIN_PATH = C:/Tools/riscv-gcc/bin
QEMU_PATH = C:/Program Files/qemu

# 工具链
CC = $(TOOLCHAIN_PATH)/riscv-none-elf-gcc.exe
AS = $(TOOLCHAIN_PATH)/riscv-none-elf-as.exe
LD = $(TOOLCHAIN_PATH)/riscv-none-elf-ld.exe
OBJCOPY = $(TOOLCHAIN_PATH)/riscv-none-elf-objcopy.exe
OBJDUMP = $(TOOLCHAIN_PATH)/riscv-none-elf-objdump.exe
SIZE = $(TOOLCHAIN_PATH)/riscv-none-elf-size.exe

# QEMU
QEMU = "$(QEMU_PATH)/qemu-system-riscv32.exe"
QEMU_FLAGS = -nographic -machine virt

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
OBJS = $(SRCS:.c=.o)

# Windows 路径转换
BUILD_OBJS = $(foreach obj,$(OBJS),$(BUILD_DIR)/$(obj))

# 默认目标
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).bin
	@echo Build complete!
	@$(SIZE) $(BUILD_DIR)/$(TARGET).elf

# 创建目录
$(BUILD_DIR):
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	@if not exist $(BUILD_DIR)\$(SDK_DIR) mkdir $(BUILD_DIR)\$(SDK_DIR)

# 编译规则
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo Compiling $<
	@$(CC) $(CFLAGS) -c $< -o $@

# 链接
$(BUILD_DIR)/$(TARGET).elf: $(BUILD_OBJS)
	@echo Linking $@
	@$(CC) $(LDFLAGS) $(BUILD_OBJS) -o $@

# 生成 BIN
$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	@echo Creating binary $@
	@$(OBJCOPY) -O binary $< $@

# 运行仿真
sim: $(BUILD_DIR)/$(TARGET).elf
	@echo Starting QEMU simulation...
	@$(QEMU) $(QEMU_FLAGS) -kernel $(BUILD_DIR)/$(TARGET).elf

# 清理
clean:
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)

.PHONY: all sim clean
```

---

## 📝 快速构建脚本

创建：`build.ps1`

```powershell
# IMC-22 Windows 构建脚本

param(
    [string]$Target = "all",
    [string]$Toolchain = "C:\Tools\riscv-gcc\bin"
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "IMC-22 Windows Build Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 设置环境
$env:PATH = "$Toolchain;$env:PATH"

# 验证工具链
Write-Host "`nVerifying toolchain..." -ForegroundColor Yellow
& "$Toolchain\riscv-none-elf-gcc.exe" --version

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: RISC-V toolchain not found!" -ForegroundColor Red
    Write-Host "Please install from: https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases" -ForegroundColor Red
    exit 1
}

# 执行 Make
Write-Host "`nBuilding project..." -ForegroundColor Yellow

switch ($Target) {
    "all" {
        make -f Makefile.windows all
    }
    "sim" {
        make -f Makefile.windows sim
    }
    "clean" {
        make -f Makefile.windows clean
    }
    default {
        make -f Makefile.windows $Target
    }
}

Write-Host "`nBuild completed!" -ForegroundColor Green
```

---

## ⚡ 一键安装脚本

创建：`setup-windows.ps1`

```powershell
# IMC-22 Windows 环境一键设置脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IMC-22 Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. 检查 QEMU
Write-Host "`n[1/3] Checking QEMU..." -ForegroundColor Yellow
$qemu = "C:\Program Files\qemu\qemu-system-riscv32.exe"
if (Test-Path $qemu) {
    Write-Host "✓ QEMU found" -ForegroundColor Green
} else {
    Write-Host "✗ QEMU not found" -ForegroundColor Red
    Write-Host "  Please install from: https://www.qemu.org/download/#windows" -ForegroundColor Yellow
}

# 2. 下载 RISC-V 工具链
Write-Host "`n[2/3] Checking RISC-V toolchain..." -ForegroundColor Yellow
$toolchainPath = "C:\Tools\riscv-gcc"

if (Test-Path "$toolchainPath\bin\riscv-none-elf-gcc.exe") {
    Write-Host "✓ Toolchain found" -ForegroundColor Green
} else {
    Write-Host "Toolchain not found. Downloading..." -ForegroundColor Yellow
    
    # 创建目录
    New-Item -ItemType Directory -Force -Path $toolchainPath | Out-Null
    
    # 下载链接
    $downloadUrl = "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v13.2.0-2/xpack-riscv-none-elf-gcc-13.2.0-2-win32-x64.zip"
    $zipFile = "$env:TEMP\riscv-gcc.zip"
    
    Write-Host "Downloading from GitHub..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes..." -ForegroundColor Yellow
    
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
        Write-Host "Download complete. Extracting..." -ForegroundColor Yellow
        
        Expand-Archive -Path $zipFile -DestinationPath $toolchainPath -Force
        Remove-Item $zipFile
        
        Write-Host "✓ Toolchain installed" -ForegroundColor Green
    } catch {
        Write-Host "✗ Download failed. Please download manually:" -ForegroundColor Red
        Write-Host "  $downloadUrl" -ForegroundColor Yellow
        Write-Host "  Extract to: $toolchainPath" -ForegroundColor Yellow
    }
}

# 3. 添加到 PATH
Write-Host "`n[3/3] Configuring environment..." -ForegroundColor Yellow
$env:PATH += ";$toolchainPath\bin"
Write-Host "✓ PATH updated (current session)" -ForegroundColor Green

Write-Host "`nTo make PATH permanent, add to System Environment Variables:" -ForegroundColor Yellow
Write-Host "  $toolchainPath\bin" -ForegroundColor Cyan

# 4. 验证
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nRISC-V GCC:" -ForegroundColor Yellow
& "$toolchainPath\bin\riscv-none-elf-gcc.exe" --version

Write-Host "`nQEMU:" -ForegroundColor Yellow
& $qemu --version

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Run: .\build.ps1" -ForegroundColor Cyan
Write-Host "  2. Run: .\build.ps1 sim" -ForegroundColor Cyan
```

---

## 🎯 使用指南

### 一键设置

```powershell
# 在项目目录运行
cd d:\新建文件夹\hive-reflex
.\setup-windows.ps1
```

### 编译项目

```powershell
# 方式 1: 使用脚本
.\build.ps1

# 方式 2: 使用 Make
make -f Makefile.windows

# 清理
make -f Makefile.windows clean
```

### 运行仿真

```powershell
# 方式 1: 使用脚本
.\build.ps1 sim

# 方式 2: 使用 Make
make -f Makefile.windows sim

# 方式 3: 直接运行 QEMU
& "C:\Program Files\qemu\qemu-system-riscv32.exe" -nographic -machine virt -kernel build\hive_node.elf
```

---

## 🔧 常见问题

### Q: Make 命令找不到？
**A**: 安装 Make for Windows
```powershell
# 使用 Chocolatey
choco install make

# 或使用 MinGW
# https://sourceforge.net/projects/mingw-w64/
```

### Q: PowerShell 脚本无法运行？
**A**: 允许脚本执行
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Q: 工具链下载失败？
**A**: 手动下载
1. 访问: https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases
2. 下载 `xpack-riscv-none-elf-gcc-*-win32-x64.zip`
3. 解压到 `C:\Tools\riscv-gcc\`

---

## 📋 检查清单

- [ ] QEMU 已安装（`C:\Program Files\qemu\`）
- [ ] RISC-V GCC 已安装（`C:\Tools\riscv-gcc\`）
- [ ] Make 已安装
- [ ] 环境变量已配置
- [ ] 测试编译成功
- [ ] 测试仿真运行

---

**创建日期**: 2026-01-17  
**适用系统**: Windows 10/11
