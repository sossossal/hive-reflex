# IMC-22 RISC-V 核心集成指南

## 🎯 系统架构

```
┌──────────────────────────────────────────┐
│         RISC-V Core (RV32IMAC)           │
│              @ 100 MHz                    │
├──────────────────────────────────────────┤
│              AHB Bus Matrix              │
│          (Arbitration & Routing)         │
├─────────┬──────────┬───────────┬─────────┤
│   CPU   │   DMA    │    CIM    │  Flash  │
│ Master  │  Master  │  Master   │  Slave  │
└─────────┴──────────┴───────────┴─────────┘
```

## 📋 内存映射

| 区域 | 起始地址 | 大小 | 说明 |
|------|---------|------|------|
| **FLASH** | 0x08000000 | 2MB | 代码和常量（XIP）|
| **SRAM** | 0x20000000 | 512KB | 数据和栈 |
| **CIM** | 0x50000000 | 512KB | 神经网络权重 |
| **外设** | 0x40000000 | - | 寄存器映射 |

### 外设地址分配

| 外设 | 地址 | 偏移 |
|-----|------|------|
| RBB | 0x40010000 | +0x10000 |
| FLASH_CTRL | 0x40020000 | +0x20000 |
| GPIO | 0x40030000 | +0x30000 |
| UART | 0x40040000 | +0x40000 |
| SPI | 0x40050000 | +0x50000 |
| PWM | 0x40070000 | +0x70000 |
| ADC | 0x40080000 | +0x80000 |
| CAN | 0x40090000 | +0x90000 |
| DMA | 0x400A0000 | +0xA0000 |
| TIMER | 0x400B0000 | +0xB0000 |

## 🚀 编译和烧录

### 1. 安装工具链

```bash
# Ubuntu/Debian
sudo apt-get install gcc-riscv32-unknown-elf openocd

# macOS
brew install riscv-gnu-toolchain openocd
```

### 2. 编译固件

```bash
# 编译
make APP_SRCS=examples/example_reflex_inference.c

# 查看大小
make size

# 生成反汇编
make disasm
```

### 3. 烧录到芯片

#### 方法 1: 使用 OpenOCD

```bash
# 启动 OpenOCD
openocd -f imc22.cfg

# 在另一个终端连接
telnet localhost 4444

# 烧录固件
> flash write_image erase build/hive_reflex.bin 0x08000000
> verify_image build/hive_reflex.bin 0x08000000
> reset run
```

#### 方法 2: 使用 Makefile

```bash
make flash
```

## 🐛 调试

### GDB 调试

```bash
# 终端 1: 启动 OpenOCD
openocd -f imc22.cfg

# 终端 2: 启动 GDB
riscv32-unknown-elf-gdb build/hive_reflex.elf

# 在 GDB 中
(gdb) target remote :3333
(gdb) load
(gdb) break main
(gdb) continue
(gdb) info registers
(gdb) backtrace
```

### UART 调试日志

```c
// 在代码中添加调试输出
printf("调试信息: value = %d\n", value);

// 连接 UART（115200 波特率）
screen /dev/ttyUSB0 115200
```

## 🔧 系统配置

### 修改系统时钟

编辑 `imc22_sdk/imc22.h`:

```c
#define IMC22_SYSCLK_HZ     100000000   // 改为目标频率
```

### 配置堆栈大小

编辑 `imc22_sdk/linker.ld`:

```ld
STACK_SIZE = 32K;  // 增加栈大小
HEAP_SIZE  = 128K; // 增加堆大小
```

### 中断优先级

编辑 `imc22_sdk/startup.c` 修改中断处理函数：

```c
void CAN_IRQHandler(void) {
    // 处理 CAN 中断
}
```

## 📊 总线仲裁

**优先级（高到低）：**
1. CIM（最高）- 神经网络推理需要高带宽
2. DMA - 批量数据传输
3. CPU - 正常指令执行
4. 外设 - 较低优先级

**配置方法：**
编辑 `system_imc22.c` 中的 `Bus_Config()`

## 🛠️ DMA 使用

### 配置 DMA 传输

```c
#include "imc22.h"

// 源和目标缓冲区
uint8_t src[1024];
uint8_t dst[1024];

// 启动 DMA 传输
DMA_Transfer(0, src, dst, 1024);

// 等待完成
DMA_Wait(0, 100);  // 100ms 超时
```

### DMA 中断模式

```c
// 使能 DMA 中断
DMA->CH[0].CTRL |= DMA_CTRL_IRQ_EN;

// 实现中断处理函数
void DMA_IRQHandler(void) {
    // 处理 DMA 完成
}
```

## 🧪 验证测试

### 最小测试程序

```c
#include "imc22.h"

int main(void) {
    System_Init();
    
    printf("IMC-22 启动成功!\n");
    SystemInfo_Print();
    
    // LED 闪烁测试
    while(1) {
        GPIO->TOGGLE = (1 << 0);  // 翻转 LED
        Delay_ms(500);
    }
    
    return 0;
}
```

### 内存测试

```c
// 测试 SRAM
uint32_t *sram = (uint32_t*)SRAM_BASE;
sram[0] = 0x12345678;
assert(sram[0] == 0x12345678);

// 测试 CIM SRAM
uint32_t *cim = (uint32_t*)CIM_BASE;
cim[0] = 0xABCDEF00;
assert(cim[0] == 0xABCDEF00);
```

### FLASH 测试

```c
#include "imc22_flash.h"

FLASH_Init(true);  // 使能 XIP

// 读取 FLASH
uint8_t buffer[256];
FLASH_Read(0, buffer, 256);

// 写入 FLASH
FLASH_EraseSector(0x1000);
FLASH_Write(0x1000, data, 256);
```

## ⚠️ 常见问题

### Q1: 编译失败 "undefined reference to _start"

**A:** 确保链接脚本正确配置，检查 `linker.ld`

### Q2: 烧录失败

**A:** 检查调试器连接，确认 OpenOCD 配置正确

### Q3: 程序运行后立即崩溃

**A:** 
1. 检查栈大小是否足够
2. 验证中断向量表
3. 确认时钟配置正确

### Q4: UART 无输出

**A:** 
1. 检查波特率配置
2. 确认 GPIO 复用功能
3. 验证时钟使能

## 📚 参考文档

- [imc22.h](file:///d:/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/hive-reflex/imc22_sdk/imc22.h) - 硬件定义
- [linker.ld](file:///d:/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/hive-reflex/imc22_sdk/linker.ld) - 链接脚本
- [startup.c](file:///d:/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/hive-reflex/imc22_sdk/startup.c) - 启动代码
- [riscv_custom.h](file:///d:/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/hive-reflex/imc22_sdk/riscv_custom.h) - 自定义指令

---

**版本**: 2.0  
**更新**: 2026-01-19
