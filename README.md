# Hive-Reflex v1.0 "Silicon Swarm"

> **Open-Source Computing-in-Memory (CIM) Development Stack**  
> From PyTorch/ONNX to Custom Silicon in Minutes.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](RELEASE_NOTES_v1.0.md)

---

## 🌟 Vision: "Democratizing Custom AI Chips"

Hive-Reflex is an end-to-end framework that allows software engineers to design application-specific AI chips (SoCs). By combining a **Digital CIM Core** with an **MLIR-based Software Stack**, we allow you to:

1.  **Input**: Any ONNX Model (Vision, Audio, NLP, GenAI).
2.  **Input**: A JSON file describing your sensors/IO.
3.  **Output**: A fully verified, optimized compilation of **C Firmware** and **Verilog RTL**.

---

## ✨ Key Features (v1.0)

### 🧬 Silicon Compiler
- **RTL Pruning**: Automatically removes unused hardware engines (e.g., if you only use MLP, the Conv engine is stripped) to save area.
- **Power Optimization**: Auto-generates Power Gating logic for micro-watt standby power.
- **IO Integration**: Configure I2C, SPI, UART, GPIO layout via JSON.

### 🧠 Advanced Model Support
- **Transformer Ready**: Native acceleration for Multi-Head Attention, Softmax, and LayerNorm.
- **GenAI / LLM**: Experimental support for **TinyLlama** (RMSNorm, SiLU, RoPE).
- **Heterogeneous**: Automatic CPU/CIM task partitioning.

### 🛠️ Developer Experience
- **VSCode Extension**: Interactive scheduler visualization and one-click deployment.
- **HEx API**: Standardized Hardware Extension Interface for generic driver integration.

---

## 🔮 Sim-Reflex (v2.0 Beta Features)

### 🖥️ Digital Twin Simulator (`simulator/`)
- **Bit-Accurate**: Simulates `Int8` CIM behavior, including quantization noise.
- **Profiler**: Generates cycle-accurate performance reports and MAC utilization analysis.

### ⚗️ Model Distillation & QAT (`distillation/`)
- **Knowledge Distillation**: Transfer knowledge from large "Teacher" models (e.g. BERT) to tiny "Student" models.
- **Adaptive QAT**: Hardware-in-the-Loop training. The model learns to adapt to the specific "flaws" (quantization errors) of the CIM chip during training.

---

## 🦁 Model Zoo (Included)

| Model | Domain | Architecture | Hardware Target |
| :--- | :--- | :--- | :--- |
| **MobileNet V1** | Vision | Depthwise Separable Conv | CIM Conv Engine |
| **Micro-KWS** | Audio | DS-CNN | CIM Conv Engine |
| **GestureNet** | Sensor | 1D-CNN | CIM Conv Engine |
| **BERT-Tiny** | NLP | Transformer Encoder | CIM Transformer Accel |
| **NanoLlama** | GenAI | Llama Decoder | Hybrid CIM/CPU |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/hive-reflex.git
pip install -r requirements.txt
```

### 2. The "Hello Silicon" Workflow

**Goal**: Create a custom chip for Voice Commands (Wake Word).

```bash
# Step 1: Export Reference Model
python model_zoo/export_micro_kws.py

# Step 2: Define your IO (Microphone needs I2S/SPI)
echo '{"i2c_count": 0, "spi_count": 1, "uart_count": 1}' > my_chip_io.json

# Step 3: Generate Software & Hardware
python mlir_compiler/codegen_cim.py \
    --model model_zoo/micro_kws.onnx \
    --output-firmware firmware/voice_main.c \
    --prune-rtl rtl/voice_soc_config.vh \
    --io-config my_chip_io.json
```

**Output**:
- `firmware/voice_main.c`:  The application code pre-integrated with inference.
- `rtl/voice_soc_config.vh`: The storage/power/IO configuration for the chip.

---

## 🗂️ Project Structure

```
hive-reflex/
├── mlir_compiler/    # The Brain: Optimizer, Allocator, CodeGen
├── model_zoo/        # The Heart: Reference Models (PyTorch->ONNX)
├── imc22_sdk/        # The Body: Drivers, Scheduler, HEx API
├── rtl/              # The Skeleton: Verilog Hardware Sources
├── vscode-extension/ # The Face: IDE Tools
├── examples/         # Demo Applications
└── RELEASE_NOTES_v1.0.md
```

## 📚 Documentation

- [Release Notes v1.0](RELEASE_NOTES_v1.0.md)
- [Model Zoo Guide](MODEL_ZOO.md)
- [TVM Integration Guide](TVM_INTEGRATION_GUIDE.md)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License
[MIT License](LICENSE)
