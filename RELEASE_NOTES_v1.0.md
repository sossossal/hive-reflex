# 🚀 Hive-Reflex v1.0 Release Notes: "Silicon Swarm"

**Date**: January 24, 2026
**Tag**: `v1.0.0-alpha`

We are proud to announce the first major release of **Hive-Reflex**, the world's first open-source, end-to-end **Computing-in-Memory (CIM)** development stack. This release bridges the gap between high-level AI models (PyTorch/ONNX) and custom silicon implementation.

## 🌟 Highlights

- **Model-to-Silicon Compiler**: One-click generation of C firmware and optimized RTL configuration from ONNX models.
- **GenAI Ready**: First micro-architecture to support **TinyLlama** (RMSNorm, SiLU, RoPE) on edge CIM hardware.
- **Custom SoC Factory**: Automated **RTL Pruning** and **Power Gating** generation allows users to create application-specific single-function chips (e.g., a dedicated "Smart Ring" chip) in minutes.

---

## 📦 Key Features

### 1. The Compiler Stack (`mlir_compiler`)
- **Advanced Graph Optimization**: Automatic operator fusion (Conv+Bn+Relu, MatMul+Add), constant folding, and dead code elimination.
- **Memory Planning**: Liveness analysis-based linear scan allocator to minimize SRAM footprint.
- **Transformer Support**: Native mapping for Multi-Head Attention, LayerNorm, Softmax, and **RMSNorm/SiLU** (New in v1.0).
- **Backend Generators**:
  - `inference.c`: Optimized kernel calls.
  - `main_soc.c`: Complete SoC firmware with sensor loops.

### 2. SoC Generation Tools
- **Hardware Pruner (`rtl_pruner.py`)**: Automatically analyzes model requirements to remove unused hardware blocks (e.g., stripping the Convolution engine for NLP-only chips).
- **Power Optimizer**: Generates `POWER_GATE` defines to physically isolate unused power domains.
- **IO Integrator**: Configures I2C/SPI/UART interfaces via JSON (`--io-config`).

### 3. Developer Experience (DX)
- **HEx API**: A standardized Hardware Extension Interface (`Init`, `Read`, `Act`) for plugging in custom sensors/actuators.
- **VSCode Extension**: Interactive scheduler visualization, one-click compile, and IntelliSense setup.
- **TVM Integration**: Preliminary BYOC (Bring Your Own Codegen) support for TVM Relay.

### 4. Model Zoo (Reference Implementations)
Pre-validated models optimized for Hive-Reflex:
- **Vision**: `MobileNet V1` (Depthwise Separable Conv).
- **Audio**: `Micro-KWS` (Keyword Spotting DS-CNN).
- **Sensor**: `GestureNet` (6-axis IMU Gesture Recognition).
- **NLP**: `BERT-Tiny` (Transformer Encoder).
- **GenAI**: `NanoLlama` (Llama2/3 Architecture Tech Demo).

---

## 🛠️ Usage Example

**Generate a Custom "Voice Remote" Chip:**

```bash
# 1. Export Micro-KWS Model
python model_zoo/export_micro_kws.py

# 2. Compile & Generate Silicon Config
python mlir_compiler/codegen_cim.py \
    --model model_zoo/micro_kws.onnx \
    --output-firmware firmware/voice_soc.c \
    --prune-rtl rtl/voice_config.vh \
    --io-config examples/io_config_mic.json
```

**Result:**
- `firmware/voice_soc.c`: Ready-to-flash C code.
- `rtl/voice_config.vh`: `define POWER_GATE_TRANSFORMER_DOMAIN`, `define ENABLE_AGI_AUDIO_FE`.

---

## 🔮 Future Roadmap (v1.x)
- **Int4 Quantization**: Hardware-aware quantization for Llama models.
- **Multi-Core**: Support for "Swarm" configurations (Multi-CIM tiles).
- **FPGA Bitstream Service**: Cloud-based bitstream generation.

Thank you to the open-source community/Deepmind Team for the support!
