/**
 * @file example_reflex_node.c
 * @brief 完整的 Hive-Reflex 节点控制程序
 *
 * 展示如何使用 IMC-22 SDK 实现:
 * - CAN 通信
 * - NPU 神经网络推理
 * - 电机 PWM 控制
 * - 周期性控制循环 (1kHz)
 */

#include "imc22.h"
#include <string.h>

/* ========== 配置参数 ========== */
#define MY_NODE_ID 1
#define CONTROL_FREQ_HZ 1000
#define CAN_CMD_ID (0x200 + MY_NODE_ID)
#define MAX_TORQUE 10.0f

/* ========== 全局状态 ========== */
typedef struct {
  float target_angle;
  float compliance;
  float current_angle;
  float current_load;
  uint32_t last_can_rx_time;
} NodeState_t;

NodeState_t g_state = {0};

/* LSTM 隐藏状态 */
static float lstm_hidden[16] = {0};
static float lstm_cell[16] = {0};

/* NPU 模型 */
extern const uint8_t reflex_net_weights[]; // 权重数据 (在 Flash 中)
extern const uint32_t reflex_net_size;

NPU_Model_t reflex_model;
NPU_Context_t npu_ctx;

/* ========== 控制算法 ========== */

float PID_Controller(float target, float current) {
  static float integral = 0;
  static float prev_error = 0;

  const float Kp = 10.0f;
  const float Ki = 0.1f;
  const float Kd = 1.0f;

  float error = target - current;
  integral += error * (1.0f / CONTROL_FREQ_HZ);
  float derivative = (error - prev_error) * CONTROL_FREQ_HZ;
  prev_error = error;

  return Kp * error + Ki * integral + Kd * derivative;
}

/* ========== 1kHz 控制循环 ========== */

void TIMER_IRQHandler(void) {
  // 清除中断
  TIMER->STATUS = 0;

  // 1. 读取传感器 (简化版本)
  g_state.current_angle = 0.0f; // TODO: 从编码器读取
  g_state.current_load = ADC_ReadVoltage(0, 3.3f);

  // 2. 准备 NPU 输入 (12 维)
  float npu_input[12];
  memset(npu_input, 0, sizeof(npu_input));
  npu_input[10] = g_state.current_load;
  npu_input[11] = g_state.target_angle - g_state.current_angle;

  // 3. NPU 推理 (非阻塞)
  float reflex_output = 0.0f;
  NPU_StartInference(&npu_ctx, npu_input);
  NPU_WaitDone(100); // 最多等待 100us

  // 获取输出 (简化)
  // reflex_output = NPU_GetOutput();

  // 4. 通信超时检测
  uint32_t current_time = TIMER->COUNT / (IMC22_SYSCLK_HZ / 1000);
  bool is_timeout = (current_time - g_state.last_can_rx_time) > 100;

  float effective_compliance = g_state.compliance;
  if (is_timeout) {
    effective_compliance = 0.9f;
    g_state.target_angle = g_state.current_angle;
  }

  // 5. 混合控制律
  float pid_out = PID_Controller(g_state.target_angle, g_state.current_angle);
  float final_output = pid_out * (1.0f - effective_compliance) +
                       reflex_output * effective_compliance * MAX_TORQUE;

  // 6. 输出到电机 (转换为 PWM 占空比)
  float duty =
      (final_output / MAX_TORQUE) * 50.0f + 50.0f; // -10~10 映射到 0~100%
  duty = (duty < 0) ? 0 : (duty > 100) ? 100 : duty;
  PWM_SetDuty(0, duty);
}

/* ========== CAN 回调 ========== */

void CAN_RxCallback(CAN_Message_t *msg) {
  if (msg->id == CAN_CMD_ID) {
    // 解析命令 (假设使用定点数)
    int16_t angle_int16 = (msg->data[1] << 8) | msg->data[0];
    uint8_t compliance_u8 = msg->data[2];

    g_state.target_angle = angle_int16 * 0.01f; // 单位: 0.01度
    g_state.compliance = compliance_u8 / 255.0f;

    g_state.last_can_rx_time = TIMER->COUNT / (IMC22_SYSCLK_HZ / 1000);
  }
}

/* ========== 主函数 ========== */

int main(void) {
  /* 1. 初始化外设 */

  // CAN
  CAN_Config_t can_cfg = {
      .baudrate = 1000000, .fd_mode = true, .loopback = false};
  CAN_Init(&can_cfg);
  CAN_SetFilter(0, CAN_CMD_ID, 0x7FF);
  CAN_EnableRxInterrupt(true);

  // ADC
  ADC_Init();

  // PWM (20 kHz)
  PWM_Init(20000);

  // NPU
  NPU_Init();

  /* 2. 加载神经网络 */
  reflex_model.weight_size = reflex_net_size;
  reflex_model.dtype = NPU_DTYPE_INT8;
  reflex_model.has_lstm = true;

  NPU_LoadModel(&reflex_model, reflex_net_weights);

  npu_ctx.model = &reflex_model;
  npu_ctx.lstm_h = lstm_hidden;
  npu_ctx.lstm_c = lstm_cell;
  npu_ctx.lstm_size = 16;

  /* 3. 握手广播 */
  CAN_Message_t handshake = {.id = 0x7FF,
                             .dlc = 2,
                             .data = {0x01, MY_NODE_ID}, // Type=LEG, ID=1
                             .is_extended = false};
  CAN_Send(&handshake);

  /* 4. 启动 1kHz 定时器 */
  TIMER->LOAD = IMC22_SYSCLK_HZ / CONTROL_FREQ_HZ;
  TIMER->CTRL = TIMER_CTRL_EN | TIMER_CTRL_MODE | TIMER_CTRL_IE;
  NVIC_SetPriority(IRQ_TIMER, 0); // 最高优先级
  NVIC_EnableIRQ(IRQ_TIMER);

  /* 5. 主循环 (低优先级任务) */
  while (1) {
    // LED 闪烁
    GPIO->TOGGLE = 1;
    DelayMs(500);
  }

  return 0;
}

/* 虚拟权重数据 (实际需要从 ONNX 转换) */
const uint8_t reflex_net_weights[2048]
    __attribute__((section(".rodata"))) = {0};
const uint32_t reflex_net_size = sizeof(reflex_net_weights);
