# IMC-22 SDK 编程指南

## 📚 目录

1. [快速开始](#快速开始)
2. [SDK 架构](#sdk-架构)
3. [外设驱动](#外设驱动)
4. [NPU 使用指南](#npu-使用指南)
5. [构建和烧录](#构建和烧录)
6. [调试技巧](#调试技巧)

---

## 快速开始

### 工具链安装

```bash
# 安装 RISC-V 工具链
# Ubuntu/Debian:
sudo apt-get install gcc-riscv32-unknown-elf

# macOS:
brew install riscv-gnu-toolchain
```

### 编译示例程序

```bash
cd hive-reflex

# 编译 Hello World
make APP_SRCS=examples/example_hello.c

# 编译完整节点控制
make APP_SRCS=examples/example_reflex_node.c

# 查看二进制大小
ls -lh build/*.bin
```

### 烧录到硬件

```bash
# 使用 OpenOCD (需要 J-Link 或 ST-Link)
make flash
```

---

## SDK 架构

```
imc22_sdk/
├── imc22.h          # 主头文件 (寄存器定义、内存映射)
├── imc22_can.h/c    # CAN-FD 驱动
├── imc22_npu.h/c    # 神经加速器驱动
├── imc22_spi.h      # SPI 驱动
├── imc22_pwm.h      # PWM 驱动
├── imc22_adc.h      # ADC 驱动
├── startup.c        # 启动代码 (向量表、复位处理)
└── linker.ld        # 链接脚本 (内存布局)
```

### 内存映射

| 区域 | 起始地址 | 大小 | 说明 |
|------|---------|------|------|
| Flash | 0x08000000 | 2 MB | 代码和常量 |
| SRAM | 0x20000000 | 512 KB | 数据和栈 |
| 外设 | 0x40000000 | - | 寄存器映射 |
| NPU | 0x50000000 | 128 KB | 神经加速器 SRAM |

---

## 外设驱动

### CAN-FD 通信

```c
#include "imc22.h"

int main(void) {
    // 1. 初始化 CAN (1 Mbps)
    CAN_Config_t cfg = {
        .baudrate = 1000000,
        .fd_mode = true,
        .loopback = false
    };
    CAN_Init(&cfg);
    
    // 2. 设置接收过滤器
    CAN_SetFilter(0, 0x200, 0x700); // 接收 0x200-0x2FF
    
    // 3. 发送消息
    CAN_Message_t msg = {
        .id = 0x201,
        .dlc = 8,
        .data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
    };
    CAN_Send(&msg);
    
    // 4. 接收消息 (非阻塞)
    if (CAN_Receive(&msg) == 0) {
        // 处理消息
    }
}

// 中断模式接收
void CAN_RxCallback(CAN_Message_t *msg) {
    // 在中断中处理接收数据
}
```

### PWM 电机控制

```c
// 初始化 PWM (20 kHz)
PWM_Init(20000);

// 设置占空比 (0-100%)
PWM_SetDuty(0, 50.0f);  // 通道 0, 50%
PWM_SetDuty(1, 75.0f);  // 通道 1, 75%
```

### ADC 电流采样

```c
ADC_Init();

// 读取原始值 (12-bit)
uint16_t raw = ADC_Read(0);

// 读取电压值
float voltage = ADC_ReadVoltage(0, 3.3f); // 参考电压 3.3V
```

---

## NPU 使用指南

### 1. 准备模型权重

```python
# 使用 reflex_net.py 导出 ONNX
python reflex_net.py --quantize

# 转换为 C 数组 (使用工具或 xxd)
xxd -i reflex_net_int8.onnx > reflex_weights.c
```

### 2. 加载模型

```c
// 权重数据 (在 Flash 中)
extern const uint8_t reflex_net_weights[];
extern const uint32_t reflex_net_size;

NPU_Model_t model;
model.weight_size = reflex_net_size;
model.dtype = NPU_DTYPE_INT8;
model.has_lstm = true;

NPU_Init();
NPU_LoadModel(&model, reflex_net_weights);
```

### 3. 执行推理

```c
// 准备输入 (12 维)
float input[12] = {
    0.1, 0.2, 0.3,  // Gyro
    0.0, 0.0, 9.8,  // Accel
    0.0, 0.0, 0.0,  // Gyro Prev
    1.2,            // Current
    0.5,            // Error Angle
    0.0             // 保留
};

float output[1] = {0};

// 创建上下文
float lstm_h[16] = {0};
float lstm_c[16] = {0};

NPU_Context_t ctx = {
    .model = &model,
    .lstm_h = lstm_h,
    .lstm_c = lstm_c,
    .lstm_size = 16
};

// 推理
NPU_Inference(&ctx, input, output);

// output[0] 即为反射力矩修正值 [-1, 1]
```

### 4. 性能测试

```c
uint32_t start = GetCycleCount();
NPU_Inference(&ctx, input, output);
uint32_t end = GetCycleCount();

uint32_t cycles = end - start;
uint32_t us = cycles / (IMC22_SYSCLK_HZ / 1000000);
printf("推理耗时: %lu us\n", us);
```

---

## 构建和烧录

### Makefile 目标

```bash
# 编译所有
make

# 仅编译
make build/hive_node.elf

# 生成反汇编
make disasm

# 清理
make clean

# 烧录
make flash
```

### 自定义编译选项

修改 `Makefile`:

```makefile
# 优化级别 (-O0, -O1, -O2, -O3, -Os)
OPT_FLAGS = -O2 -g

# 应用源文件
APP_SRCS = my_app.c another_file.c
```

---

## 调试技巧

### 1. UART 日志输出

```c
void uart_puts(const char *str) {
    while (*str) {
        while (!(UART->STATUS & UART_STATUS_TXE));
        UART->DATA = *str++;
    }
}

uart_puts("Debug: value = ");
// 使用 snprintf 格式化输出
```

### 2. LED 调试

```c
#define LED_ERROR   (1 << 0)
#define LED_OK      (1 << 1)

// 错误指示
GPIO->SET = LED_ERROR;

// 正常运行指示
GPIO->TOGGLE = LED_OK;
```

### 3. 性能分析

```c
#define PROFILE_START() uint32_t _t = GetCycleCount()
#define PROFILE_END(name) \
    printf("%s: %lu cycles\n", name, GetCycleCount() - _t)

PROFILE_START();
NPU_Inference(&ctx, input, output);
PROFILE_END("NPU Inference");
```

### 4. GDB 调试

```bash
# 启动 OpenOCD (终端 1)
openocd -f interface/jlink.cfg -f target/riscv.cfg

# 启动 GDB (终端 2)
riscv32-unknown-elf-gdb build/hive_node.elf

# 在 GDB 中连接
(gdb) target remote :3333
(gdb) load
(gdb) break main
(gdb) continue
```

---

## 常见问题

### Q: 如何更改 CAN ID?

A: 修改 `example_reflex_node.c` 中的 `MY_NODE_ID`:

```c
#define MY_NODE_ID 2  // 改为节点 2
```

### Q: 如何调整控制频率?

A: 修改 `CONTROL_FREQ_HZ`:

```c
#define CONTROL_FREQ_HZ 500  // 改为 500Hz
```

### Q: NPU 推理超时怎么办?

A: 检查模型是否太大，或增加超时时间:

```c
NPU_WaitDone(500);  // 增加到 500us
```

---

## 许可证

MIT License - 仅供学习和研究使用

---

**更新日期**: 2026-01-16  
**SDK 版本**: v1.0  
**支持芯片**: IMC-22 (RISC-V RV32IMAC)
