# Hive-Reflex 2.1 芯片数据手册

**版本**: v2.1.0  
**日期**: 2026-02-03  
**状态**: Preliminary

---

## 1. 概览 (Overview)

Hive-Reflex 2.1 是一款基于 **Computing-in-Memory (CIM)** 架构的超低功耗边缘 AI 加速器，专为 **TinyML** 和 **机器人脊髓反射** 控制设计。当前实现基于 Xilinx Zynq UltraScale+ MPSoC (ZCU102)。

## 2. 关键规格 (Key Specifications)

| 特性 | 规格 | 说明 |
|------|------|------|
| **目标器件** | Xilinx XCZU9EG-2FFVB1156E | ZCU102 开发板 |
| **主频** | 100 MHz | 系统时钟 |
| **算力** | 51.2 GOPS (Peak) | 256 INT8 MACs @ 100MHz x 2 (Ops) |
| **功耗 (Active)** | 48.86 mW | 1.0V, 100% 负载 |
| **功耗 (DeepSleep)** | 88 μW | 0.4V, 1MHz, 99.8% 节能 |
| **接口** | AHB-Lite, UART, JTAG, GPIO | 标准嵌入式接口 |

## 3. 存储与模型容量 (Memory & Model Capacity)

Hive-Reflex 采用双级存储架构，确保高带宽 CIM 计算与灵活的系统控制。

| 存储区域 | 容量 | 用途 | 带宽 |
|----------|------|------|------|
| **CIM SRAM** | **512 KB** | 存放神经网络权重和激活值 | 高带宽 (2048-bit 内部) |
| **System SRAM** | **512 KB** | 存放固件 (.text)、堆栈 (.stack) 和 I/O 缓冲区 | 32-bit AHB |
| **Total On-Chip** | **1 MB** | 总片上可用存储 | - |

### 3.1 模型尺寸限制
- **硬限制 (Hard Limit)**: **512 KB** (CIM SRAM 容量)
- **推荐模型大小**: < **400 KB** (为输入/输出缓冲区预留空间)
- **参数数量 (INT8)**: 约 **400,000 - 500,000** 个参数
- **稀疏化扩容**: 支持 2x-4x 稀疏压缩，等效支持 **1M - 2M** 参数 (取决于稀疏度)

### 3.2 扩展潜力 (Expansion Potential)
目标器件 (XCZU9EG) 拥有约 8.5 MB 的片上 RAM (Block RAM + UltraRAM)。当前设计仅使用了 1 MB。用户可通过修改 RTL 参数 (`SRAM_ADDR_WIDTH`) 轻松扩展至 **4 MB+**。

## 4. 算力详细参数 (Compute Subsystem with Sparsity)

- **MAC 阵列**: 256 并行单元 (INT8)
- **稀疏加速**: 支持动态阈值跳过零值
- **流水线**: 3 级流水线累加树
- **量化支持**: INT8 (Weight) x INT8 (Input) -> INT32 (Accumulator)

## 5. 软件栈支持 (Software Stack)

- **编译器**: MLIR-based Compiler (支持 PyTorch/ONNX)
- **模型格式**: FlatBuffers / C Header
- **典型模型**: 
  - Adaptive MLP (0.4 KB)
  - Micro-KWS (20 KB)
  - MobileNet v1-0.25 (200 KB)

---

**注意**: 本数据手册基于 v2.1 RTL 配置生成。实际性能可能因 FPGA 综合策略而异。
