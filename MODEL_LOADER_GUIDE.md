# Hive-Reflex 2.0 模型加载器使用指南

## 📖 概述

模型加载器提供了从 FLASH 加载神经网络模型并执行推理的完整功能。

## 🚀 快速开始

### 1. 准备模型

```bash
# 导出 ONNX 模型
python reflex_net_v2.py --quantize

# 编译为 CIM 代码
python mlir_compiler/compile.py --model reflex_net_v2.onnx
```

### 2. 打包模型

```bash
# 打包权重为 FLASH 格式
python tools/flash_model.py \
    --weights build/reflex_weights.bin \
    --output build/model.flash \
    --name "ReflexNetV2" \
    --input-size 12 \
    --output-size 1 \
    --hidden-size 16 \
    --has-lstm \
    --gen-script
```

### 3. 烧录到芯片

```bash
# 方法 1: 使用完整构建脚本 (推荐)
.\build-complete.ps1 -Flash

# 方法 2: 手动烧录
openocd -f interface/jlink.cfg -f target/riscv.cfg -f build/model.ocd
```

### 4. 运行示例程序

```bash
# 编译推理示例
make APP_SRCS=examples/example_reflex_inference.c

# 烧录固件
make flash
```

## 📋 API 使用示例

### 加载模型

```c
#include "model_loader.h"

// 1. 定义模型结构
Model_t model;

// 2. 从 FLASH 加载
if (Model_LoadFromFlash(MODEL_REFLEX_V2, &model) != 0) {
    printf("加载失败\n");
    return -1;
}

// 3. 显示模型信息
Model_PrintInfo(&model);

// 4. 加载到 CIM SRAM
Model_LoadToCIM(&model, 0);  // Bank 0
```

### 执行推理

```c
// 1. 创建推理上下文
InferenceContext_t *ctx = Inference_CreateContext(&model);

// 2. 准备输入数据 (12 维)
float input[12] = {
    0.1f, 0.2f, 0.3f,  // Gyro
    0.0f, 0.0f, 9.8f,  // Accel
    0.0f, 0.0f, 0.0f,  // Gyro prev
    1.2f,              // Current
    0.5f,              // Error
    0.0f               // Reserved
};

// 3. 执行推理
float output[1];
Inference_Run(ctx, input, output);

printf("输出: %.3f\n", output[0]);

// 4. 清理
Inference_DestroyContext(ctx);
```

### 性能监控

```c
// 获取推理统计
uint32_t avg_time_us;
float fps;
Inference_GetStats(ctx, &avg_time_us, &fps);

printf("平均延迟: %lu μs\n", avg_time_us);
printf("推理速率: %.1f FPS\n", fps);

// 获取 CIM 性能
CIM_PerfStats_t stats;
CIM_GetPerfStats(&stats);
printf("GOPS: %.2f\n", stats.gops);
```

## 🔧 高级用法

### 自定义模型格式

如果你需要自定义模型格式，修改 `tools/flash_model.py`:

```python
# 自定义配置
config = {
    'input_size': 24,      # 增加输入维度
    'output_size': 4,      # 多输出
    'hidden_size': 32,     # 更大的隐藏层
    'num_layers': 5,       # 更深的网络
    'dtype': 0,            # INT8
    'has_lstm': 1,         # 包含 LSTM
    'quant_scale': 0.1,
    'quant_zero': 128
}
```

### 运行时更新模型

```c
// 从新地址加载模型
Model_Unload(&model);
Model_LoadFromFlash(MODEL_CUSTOM, &model);
Model_LoadToCIM(&model, 0);

// 重新创建推理上下文
Inference_DestroyContext(ctx);
ctx = Inference_CreateContext(&model);
```

### LSTM 状态管理

```c
// 重置 LSTM 状态 (开始新序列)
Inference_ResetState(ctx);

// 持续推理 (保持 LSTM 状态)
for (int i = 0; i < 100; i++) {
    Inference_Run(ctx, input, output);
    // LSTM 状态自动更新
}
```

## 📊 模型文件格式

### FLASH 布局

```
模型文件结构:
┌──────────────────┐ 0x0000
│ Header (128B)    │
│  - Magic         │
│  - Version       │
│  - Offsets       │
│  - CRC32         │
├──────────────────┤
│ Config (32B)     │
│  - Input size    │
│  - Output size   │
│  - Hidden size   │
│  - Quant params  │
├──────────────────┤
│ Weights (xKB)    │
│  - Layer 1       │
│  - Layer 2       │
│  - ...           │
└──────────────────┘
```

### 头部结构

```c
typedef struct {
    uint32_t magic;         // 0x43494D32 ("CIM2")
    uint16_t version;       // 0x0200 (v2.0)
    uint16_t reserved;
    uint32_t model_size;    // 总大小
    uint32_t weight_offset; // 权重偏移
    uint32_t weight_size;   // 权重大小
    uint32_t config_offset; // 配置偏移
    uint32_t config_size;   // 配置大小
    uint32_t crc32;         // CRC32 校验
    char model_name[32];    // 模型名称
    char model_hash[64];    // SHA-256 哈希
} ModelHeader_t;
```

## 🎯 性能优化

### 优化推理速度

1. **使用 INT8 量化**
   ```bash
   python reflex_net_v2.py --quantize
   ```

2. **优化 CIM Bank 分配**
   ```c
   // 预加载多个模型到不同 Bank
   Model_LoadToCIM(&model1, 0);
   Model_LoadToCIM(&model2, 1);
   ```

3. **启用 CIM 中断模式**
   ```c
   CIM_EnableIRQ(true);
   // 推理在后台运行,不阻塞主循环
   ```

### 减少功耗

```c
// 启用电源管理
Power_EnableAutoMode(100);

// 推理间隙自动进入低功耗
while(1) {
    Inference_Run(ctx, input, output);
    Delay_ms(10);  // 100Hz 推理
    Power_Update();  // 自动降低功耗
}
```

## 🐛 故障排除

### 模型加载失败

**问题**: `Model_LoadFromFlash` 返回 -1

**解决方案**:
1. 检查 FLASH 地址是否正确
2. 验证模型是否已烧录
3. 检查 CRC32 校验

### 推理输出异常

**问题**: 输出值不合理

**解决方案**:
1. 检查输入数据范围
2. 验证量化参数
3. 重置 LSTM 状态

### 性能不达预期

**问题**: 推理延迟过高

**解决方案**:
1. 确认 CIM 已正确初始化
2. 检查模型是否加载到 CIM SRAM
3. 监控 `CIM_GetPerfStats()` 输出

## 📚 参考

- [implementation_plan.md](file:///C:/Users/%E8%8D%A3%E8%80%80/.gemini/antigravity/brain/fcf659df-124f-41ad-9fe7-b48e2742b793/implementation_plan.md) - 技术方案
- [SDK_GUIDE.md](../SDK_GUIDE.md) - SDK 文档
- [example_reflex_inference.c](../examples/example_reflex_inference.c) - 完整示例

---

**版本**: 2.0  
**更新**: 2026-01-19
