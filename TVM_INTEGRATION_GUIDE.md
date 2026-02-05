# Hive-Reflex TVM Integration Guide

This guide explains how to use TVM to compile models for the Hive-Reflex CIM accelerator using the BYOC (Bring Your Own Codegen) flow.

## 1. Overview

We leverage Apache TVM to support a wide range of frontend frameworks (PyTorch, TensorFlow, ONNX) and offload supported operators (Dense, Conv2D, Activation) to our hardware accelerator.

```mermaid
graph LR
    A[PyTorch/ONNX] --> B[TVM Relay]
    B --> C{Partition Graph}
    C -->|Supported Ops| D[CIM Compiler]
    C -->|Unsupported| E[RISC-V CPU (LLVM)]
    D --> F[CIM C-Source]
    E --> G[RISC-V Assembly]
    F --> H[Final Firmware]
    G --> H
```

## 2. Setup

Ensure you have TVM installed:
```bash
pip install apache-tvm
```

Python scripts are located in `tvm_integration/`.

## 3. Usage Example

```python
import tvm
from tvm import relay
import onnx
from tvm_integration.cim_backend import annotate_cim_ops
import tvm_integration.codegen_wrapper # Registers the backend

# 1. Load Model
onnx_model = onnx.load("models/mobilenet.onnx")
mod, params = relay.frontend.from_onnx(onnx_model)

# 2. Partition Graph for CIM
mod = annotate_cim_ops(mod, params)

# 3. Compile
# This will invoke our custom codegen for 'cim' target regions
target = tvm.target.Target("c", host="c") # host='c' for generating C source
lib = relay.build(mod, target=target, params=params)

# 4. Export
# The output library contains C source code for the CIM accelerator
print(lib.get_source())
lib.export_library("model_lib.tar")
```

## 4. Supported Layouts

The CIM accelerator supports:
- **Dense/Linear**: `(Batch, In) x (Out, In)^T -> (Batch, Out)`
- **Activation**: ReLU, Gelu, Tanh
