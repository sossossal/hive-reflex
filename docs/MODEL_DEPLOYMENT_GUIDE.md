# Hive-Reflex 模型部署工具链使用指南

## 概览

`model_to_flash.py` 是一个**端到端的自动化部署工具**，可以将 PyTorch 或 ONNX 模型自动转换为 Hive-Reflex 芯片可用的 Flash 固件。

### 核心功能

1. ✂️ **自动切片** - 将超过 512 KB 的大模型切分为多个可加载的切片
2. 🔢 **INT8 量化** - 自动量化权重以减小模型尺寸（4x 压缩）
3. 📦 **Flash 打包** - 生成对齐 Flash 页边界的 .bin 固件文件

---

## 快速开始

### 1. 准备模型

支持的格式：
- **PyTorch**: `.pth` 或 `.pt` (完整模型，非 state_dict)
- **ONNX**: `.onnx`

```python
# PyTorch 示例：保存完整模型
import torch

model = MyModel()
torch.save(model, 'my_model.pth')  # ✅ 正确

# ❌ 错误：不要只保存 state_dict
# torch.save(model.state_dict(), 'my_model.pth')
```

### 2. 运行部署工具

```bash
# 基本用法
python tools/model_to_flash.py --input my_model.onnx --output firmware.bin

# 自动切片大模型
python tools/model_to_flash.py --input large_model.onnx --output firmware.bin --auto-slice

# 添加元数据
python tools/model_to_flash.py \
    --input model.pth \
    --output firmware.bin \
    --name "GestureRecognition" \
    --version 1.0
```

### 3. 工具输出示例

```
=============================================================
Hive-Reflex 模型自动化部署工具链
=============================================================
INFO: 📦 加载 ONNX 模型: model.onnx
INFO:   提取 12 个权重张量
INFO: 🔢 量化模型权重到 INT8...
INFO:   原始大小: 850.0 KB
INFO:   量化后: 212.5 KB
INFO:   压缩比: 4.00x
INFO: ✂️  切片模型 (最大切片: 256 KB)...
INFO:   生成 1 个切片
INFO:     切片 1: 12 层, 212.5 KB
INFO: 📦 打包 Flash 固件: firmware.bin
INFO:   ✅ 固件大小: 217088 字节 (212.0 KB)
INFO:   Flash 页数: 53 页 (+0 字节)
=============================================================
✅ 部署完成!
    输入: model.onnx
    输出: firmware.bin (212.0 KB)
    切片: 1 个
=============================================================
```

---

## 嵌入式端加载

生成的 `.bin` 固件可直接烧录到 Flash，然后使用 `flash_loader.c` 加载到 CIM SRAM。

### 示例代码

```c
#include "flash_loader.h"
#include "imc22_cim.h"

#define FLASH_MODEL_ADDR 0x10000000  // Flash 中模型起始地址
#define CIM_SRAM_BASE    0x80000000  // CIM SRAM 基地址

int main(void) {
    // 1. 验证固件
    if (flash_firmware_validate(FLASH_MODEL_ADDR) != 0) {
        printf("固件验证失败!\n");
        return -1;
    }
    
    // 2. 加载模型到 SRAM
    if (flash_load_full_model(FLASH_MODEL_ADDR, CIM_SRAM_BASE) != 0) {
        printf("模型加载失败!\n");
        return -1;
    }
    
    // 3. 初始化 CIM 引擎
    imc22_init();
    
    // 4. 运行推理
    float input[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float output[2];
    
    imc22_inference(input, output, 8, 2);
    
    printf("输出: [%.3f, %.3f]\n", output[0], output[1]);
    
    return 0;
}
```

---

## 高级功能

### 1. 多切片模型（用于超大模型）

如果模型超过 512 KB，工具会自动切片。在嵌入式端需要按需加载切片：

```c
uint16_t num_slices = flash_get_num_slices();

for (uint8_t i = 0; i < num_slices; i++) {
    // 加载切片 i
    flash_load_slice(i, CIM_SRAM_BASE);
    
    // 运行该切片的推理
    run_inference_slice(i);
}
```

### 2. 查看固件信息

使用 Python 脚本查看 `.bin` 固件的详细信息：

```python
import struct

with open('firmware.bin', 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<H', f.read(2))[0]
    num_slices = struct.unpack('<H', f.read(2))[0]
    total_size = struct.unpack('<I', f.read(4))[0]
    
    print(f"Magic: {magic}")
    print(f"Version: {version >> 8}.{(version >> 4) & 0xF}.0")
    print(f"Slices: {num_slices}")
    print(f"Size: {total_size} bytes")
```

### 3. 自定义量化参数

修改 `model_to_flash.py` 中的常量：

```python
# 调整切片大小（默认 256 KB）
MAX_LAYER_SIZE = 128 * 1024  # 128 KB 切片

# 调整 Flash 页大小（根据实际 Flash 芯片）
FLASH_PAGE_SIZE = 4096  # 或 2048, 8192 等
```

---

## 固件格式规范

### 固件布局

```
+----------------------------+
| Header (16 bytes)          |
|  - Magic: 'HRF2'           |
|  - Version: 0x0210         |
|  - Num Slices              |
|  - Total Size              |
+----------------------------+
| Metadata (JSON, 可变长)    |
+----------------------------+
| Slice 0                    |
|  - Slice Header            |
|  - Layer 0                 |
|    - Name                  |
|    - Shape                 |
|    - Scale                 |
|    - Weights (INT8[])      |
|  - Layer 1                 |
|  ...                       |
+----------------------------+
| Slice 1                    |
|  ...                       |
+----------------------------+
| Padding (对齐 Flash Page)  |
+----------------------------+
```

### 数据类型

- **权重**: INT8 (对称量化，范围 [-127, 127])
- **Scale**: FLOAT32
- **形状**: UINT32[]

---

## 完整工作流示例

### 场景：部署手势识别模型

```bash
# 1. 训练模型（使用现有工具）
python tools/train_adaptive_model.py --all

# 2. 导出为 ONNX
python -c "
import torch
import torch.onnx

model = torch.load('models/adaptive_model.pt')
dummy_input = torch.randn(1, 8)
torch.onnx.export(model, dummy_input, 'models/gesture_model.onnx')
"

# 3. 生成 Flash 固件
python tools/model_to_flash.py \
    --input models/gesture_model.onnx \
    --output fpga/firmware/gesture_model.bin \
    --name "GestureNet" \
    --version 2.1

# 4. 烧录到 Flash（使用 Vivado 或 JTAG）
vivado -mode batch -source fpga/program_flash.tcl

# 5. 运行嵌入式固件
cd fpga
make run_test
```

---

## 常见问题

### Q1: 模型太大怎么办？

**A**: 使用 `--auto-slice` 选项自动切片：

```bash
python tools/model_to_flash.py --input large_model.onnx --output fw.bin --auto-slice
```

如果仍然太大，考虑：
- 使用模型剪枝 (`tools/prune_model.py`)
- 应用稀疏化 (`mlir_compiler/sparsity_optimizer.py`)
- 使用知识蒸馏压缩模型

### Q2: 量化导致精度下降？

**A**: 使用量化感知训练 (QAT) 预先适应量化误差：

```bash
python mlir_compiler/qat_trainer.py --model model.pth --epochs 20
```

### Q3: 如何验证固件正确性？

**A**: 使用仿真器测试：

```bash
python imc22_sdk/python/sim_flash_loader.py --firmware firmware.bin --test
```

---

## 参考资料

- [模型量化原理](../docs/QUANTIZATION.md)
- [CIM SRAM 地址映射](../docs/MEMORY_MAP.md)
- [Flash 编程指南](../fpga/docs/FLASH_PROGRAMMING.md)
