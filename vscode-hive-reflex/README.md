# VSCode Hive-Reflex Extension

The official developer tools for the Hive-Reflex AI Accelerator.

## Features

### 1. One-Click Compile
Right-click on any `.onnx` model in your workspace and select **"Hive-Reflex: Compile ONNX Model"**.
This invokes the MLIR compiler to generate:
- C Source Code (Optimized CIM Kernels)
- Weight Binaries (Quantized)
- Model Configuration

### 2. Scheduler Visualization
After compilation, a visualization panel opens automatically. It shows the heterogeneous task graph, highlighting which layers are accelerated by the **CIM Core** and which fallback to the **RISC-V CPU**.

## Setup

1. Open this folder in VSCode.
2. Run `npm install`.
3. Press `F5` to debug (Start Extension Host).
