# 🦁 Hive-Reflex Model Zoo

This directory contains reference models optimized for the Hive-Reflex architecture.

## 1. MobileNet V1 Tiny (`mobilenet_v1_tiny.onnx`)
- **Domain**: Computer Vision (Image Classification)
- **Input**: 96x96 Grayscale (1x1x96x96)
- **Architecture**: Depthwise Separable Convolutions
- **Hardware Target**: CIM Conv Engine
- **Use Case**: Low-power object detection, face detection on battery-powered cameras.

## 2. GestureNet (`gesture_net.onnx`)
- **Domain**: Sensor Fusion / HCI
- **Input**: 6-axis IMU @ 32Hz (1x6x32)
- **Architecture**: 1D-CNN + Global Average Pooling
- **Hardware Target**: CIM Conv Engine + RISC-V Post-processing
- **Use Case**: Smart ring/watch gesture control (Swipe, Tap, Circle).
- **Driver Pair**: `examples/driver_mpu6050.c`

## 3. ReflexNet (`reflex_net.onnx`)
- **Domain**: Time-Series Forecasting / Control
- **Architecture**: LSTM + FC
- **Hardware Target**: CIM LSTM Engine
- **Use Case**: Predictive maintenance, motor control loop.

## 4. Micro-KWS (`micro_kws.onnx`)
- **Domain**: Audio / Speech
- **Input**: MFCC Features (1x1x49x10) - *Preprocessing required on CPU*
- **Architecture**: Depthwise Separable CNN (DS-CNN)
- **Hardware Target**: CIM Conv Engine
- **Use Case**: Wake-word detection ("Hey Reflex"), Voice Commands.

## 5. BERT-Tiny (`bert_tiny.onnx`)
- **Domain**: NLP (Text Classification / Intent)
- **Architecture**: Transformer Encoder (Self-Attention)
- **Hardware Target**: CIM Transformer Accelerator
- **Use Case**: On-device voice commands, keyword spotting.

## 6. NanoLlama (`tinyllama_nano.onnx`)
- **Domain**: GenAI / SLM
- **Architecture**: Llama2 Style (RMSNorm, SiLU, SwiGLU)
- **Status**: **Experimental**
- **Hardware Target**: CIM (MatMul) + RISC-V (RMSNorm, SiLU)
- **Use Case**: Tech demo for Generative AI on Edge.

## How to Build
Run the build script to export PyTorch definitions to ONNX:
```bash
python model_zoo/build_zoo.py
```
> Note: Requires `torch`, `onnx`, and `onnxscript` installed.

## How to Compile
Use the Hive-Reflex Compiler to generate C firmware:
```bash
# Example: Compile GestureNet for SoC
python mlir_compiler/codegen_cim.py \
    --model model_zoo/gesture_net.onnx \
    --output-firmware firmware/gesture_main.c \
    --prune-rtl rtl/gesture_soc.vh \
    --io-config examples/io_config_sensor.json
```
