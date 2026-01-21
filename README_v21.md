# Hive-Reflex 2.1 - 超低功耗边缘 AI 加速器

> **稀疏计算 + DVFS + TinyML 自适应控制 + AI 反馈循环**  
> Computing-in-Memory (CIM) 架构，专为边缘 AI 和机器人控制设计

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![FPGA](https://img.shields.io/badge/FPGA-Xilinx%20UltraScale+-orange.svg)](https://www.xilinx.com)
[![Version](https://img.shields.io/badge/version-2.1.0-brightgreen.svg)](CHANGELOG.md)

---

## ✨ 核心特性

### 🚀 稀疏计算加速
- 动态阈值配置，削减 20-50% 无效运算
- CSR 稀疏索引生成器
- 3 级流水线累加树

### ⚡ DVFS 超低功耗
- **Active**: 48.86 mW @ 1.0V/100MHz
- **Standby**: 432 μW @ 0.6V/10MHz (99.1% 节能)
- **DeepSleep**: 88 μW @ 0.4V/1MHz (**99.8% 节能**)
- 时钟门控 + 电源门控支持

### 🧠 TinyML 自适应控制
- PID/神经反射动态混合
- 量化 MLP 推理引擎 (< 10KB Flash)
- 高负载自动检测

### 🔧 QAT 量化训练
- Conv+BN 融合优化
- INT8 精度损失 < 1%
- 自动精度补偿

### 🌐 AI 反馈循环
- 运行日志收集 (100Hz)
- 云端 Llama-3 优化接口
- OTA 固件更新机制
- 自适应参数调优

---

## 🎯 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/hive-reflex.git
cd hive-reflex

# 安装 Python 依赖
pip install numpy torch onnx pytest

# 安装 Python SDK
cd imc22_sdk/python
pip install -e .
```

### 第一个示例

```python
from imc22 import CIM, Simulator, NeuralReflex
import numpy as np

# 1. CIM 稀疏推理
sim = Simulator(mac_count=256)
input_data = np.random.randn(16).astype(np.float32)
weights = np.random.randn(16, 8).astype(np.float32)

result = sim.matmul(input_data, weights, sparse=True, threshold=2)
print(f"稀疏率: {result['sparsity']*100:.1f}%, 加速: {result['speedup']:.2f}x")

# 2. TinyML 自适应控制
reflex = NeuralReflex()
weights = reflex.compute_blend(torque=5.0, velocity=1.2)
print(f"PID: {weights['pid']:.2f}, Neural: {weights['neural']:.2f}")
```

### 运行测试

```bash
cd fpga/tests
pytest test_e2e.py -v
```

---

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **资源利用率** | LUT: 0.02%, FF: 0.12% | ZCU102 FPGA |
| **Active 功耗** | 48.86 mW | 1.0V, 100MHz |
| **DeepSleep 功耗** | 88 μW | 0.4V, 1MHz |
| **节能效果** | 99.8% | Active → DeepSleep |
| **稀疏加速** | 1.25x - 2.0x | 取决于稀疏度 |
| **TinyML 模型** | 0.4 KB | 目标 < 10KB |
| **量化精度** | < 1% 损失 | QAT 优化 |

---

## 🛠️ 项目结构

```
hive-reflex/
├── rtl/                    # Verilog RTL (11 模块, ~3000 行)
│   ├── sparse_cim_mac_array.v
│   ├── dvfs_controller.v
│   ├── power_gate.v
│   └── clock_gate.v
├── imc22_sdk/              # C SDK + Python 绑定
│   ├── imc22_dvfs.c/h
│   ├── tinyml_adaptive.c/h
│   ├── nn_topology.h
│   └── python/imc22.py
├── mlir_compiler/          # MLIR 优化器
│   ├── optimizer.py
│   ├── qat_trainer.py
│   └── sparsity_optimizer.py
├── tools/                  # 完整工具链
│   ├── train_adaptive_model.py
│   ├── ai_feedback.py
│   ├── analyze_rtl.py
│   └── power_estimator.py
├── fpga/                   # FPGA 综合与测试
│   ├── constraints/
│   ├── vivado/
│   └── tests/
└── docs/                   # 文档
```

---

## 📚 文档

- [实施计划](implementation_plan.md) - 详细技术方案
- [完成报告](walkthrough.md) - 验证结果与使用指南
- [开源就绪评估](OPENSOURCE_READINESS.md) - 开源准备情况
- [贡献指南](CONTRIBUTING.md) - 如何参与贡献

---

## 🧪 测试与验证

### RTL 仿真

```bash
cd sim
python sparse_mac_sim.py
```

**结果**: 5/5 测试通过 (密集、50% 稀疏、80% 稀疏、动态阈值、模式对比)

### Python SDK 测试

```bash
cd imc22_sdk/python
python test_sdk.py
```

**结果**: 7/7 测试通过 (CIM、Power、DVFS、NeuralReflex、Simulator、综合流程)

### 端到端测试

```bash
cd fpga/tests
pytest test_e2e.py -v                 # 所有测试
pytest test_e2e.py --hil --port COM3  # HIL 硬件测试
```

---

## 🔬 高级功能

### AI 反馈循环

```bash
# 收集运行日志
python tools/ai_feedback.py --collect --duration 60

# 云端优化
python tools/ai_feedback.py --optimize

# OTA 部署
python tools/ai_feedback.py --deploy --device dev001

# 自动循环 (每 30 分钟)
python tools/ai_feedback.py --auto --interval 30
```

### TinyML 模型训练

```bash
cd tools
python train_adaptive_model.py --all --samples 10000 --epochs 100
```

**输出**: 
- `models/adaptive_model.pt` (PyTorch 模型)
- `models/adaptive_model_weights.h` (C 头文件)
- `models/adaptive_model.bin` (二进制固件)

### FPGA 综合

```bash
cd fpga/vivado
vivado -mode batch -source build_v21.tcl
```

**生成**:
- 比特流: `output/hive_reflex_top.bit`
- 功耗报告: `reports/power_active.txt`, `reports/power_standby.txt`

---

## 🎓 使用场景

### 机器人关节控制
- 低延迟反射控制 (< 100μs)
- 自适应 PID/神经混合
- 超低待机功耗

### 边缘 AI 推理
- 稀疏神经网络加速
- INT8 量化部署
- 功耗优化 (DVFS)

### 嵌入式 TinyML
- < 10KB 模型部署
- 在线学习与优化
- OTA 固件更新

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 贡献者

感谢所有贡献者！

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源。

---

## 🙏 致谢

- RISC-V 基金会
- Xilinx/AMD FPGA 工具链
- PyTorch 和 ONNX 社区
- 所有贡献者和测试者

---

## 📧 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-org/hive-reflex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/hive-reflex/discussions)
- **Email**: hive-reflex@example.com

---

## 🌟 Star History

如果这个项目对你有帮助，请给一个 Star！

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/hive-reflex&type=Date)](https://star-history.com/#your-org/hive-reflex&Date)

---

**版本**: 2.1.0 | **更新日期**: 2026-01-21
