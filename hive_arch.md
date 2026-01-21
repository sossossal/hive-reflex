# Hive-Reflex: 基于 IMC-22 的模块化脊髓反射控制器

**版本:** v0.1 (Concept)
**核心理念:** 分布式感知、本地反射、指令叠加。

---

## 1. 硬件定义 (Hardware Node)

每个关节模块 (Joint Node) 包含：
*   **MCU/NPU:** IMC-22 (RISC-V + Neural Accelerator)
*   **Sensor:** 6-Axis IMU (SPI, 1kHz Sample Rate) + Current Sensor (ADC)
*   **Actuator:** DC/BLDC Motor Driver (PWM Interface)
*   **Bus:** CAN-FD (5Mbps, 2-Wire)

## 2. 控制架构 (Control Loop)

控制频率固定为 **1kHz (1ms)**。

### 信号流
$$ U_{final} = U_{PID} \cdot (1 - \gamma) + f_{NN}(S_{local}) \cdot \gamma \cdot T_{max} $$

*   $U_{PID}$: 来自经典 PID 控制器的输出。
*   $f_{NN}$: 运行在 IMC 上的神经网络 (Reflex Policy)，输出范围 [-1, 1]。
*   $S_{local}$: 本地传感器数据 [Gyro, Accel, Current]。
*   $\gamma$: **柔顺系数** (Compliance Factor)，范围 [0, 1]。
    *   $\gamma = 0$: 完全刚性，仅 PID 控制，无反射。
    *   $\gamma = 1$: 完全柔性，仅反射控制。
*   $T_{max}$: 最大力矩限制。

## 3. 神经网络模型 (ReflexNet)

由于 IMC-22 内存限制 (512KB)，模型必须极致精简。

*   **输入 (12 dims):** 
    *   Gyro[x,y,z], Accel[x,y,z] (当前)
    *   Gyro_Prev[x,y,z] (上一时刻)
    *   Current, Error_Angle
*   **隐藏层:** 
    *   FC (12 -> 32, ReLU)
    *   LSTM (32 -> 16) [捕捉惯性趋势]
*   **输出 (1 dim):**
    *   Torque_Correction (力矩修正量，范围 [-1, 1])

**量化支持:** 模型支持 INT8 量化以进一步减小内存占用（约 2-3KB），详见 `reflex_net.py --quantize`。

## 4. 通信协议 (Hive-CAN Protocol)

**波特率:** 1Mbps / 5Mbps (FD)

| CAN ID (11-bit) | Type | Data Payload (8 Bytes) | Description |
| :--- | :--- | :--- | :--- |
| `0x000` | Sync | `Timestamp` | 全局时间同步 (Heartbeat) |
| `0x100 + NodeID` | Status | `Angle`, `Current`, `Error` | 节点状态广播 (100Hz) |
| `0x200 + NodeID` | Command| `TargetAngle` (int16), `Compliance` (uint8) | 主控下发指令 (使用定点数) |
| `0x300 + NodeID` | Config | `MaxTorque`, `PID_Kp`, `PID_Ki` | 参数配置帧 (用于调试) |
| `0x7FF` | Handshake| `Type`, `UID` | 上电握手/热插拔声明 |

**数据格式说明:**
*   `TargetAngle`: int16_t, 单位 0.01°, 范围 ±327.67°
*   `Compliance`: uint8_t, 单位 1/255, 值 0-255 映射到 0.0-1.0

---
