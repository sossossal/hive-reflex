/**
 * @file example_reflex_inference.c
 * @brief 完整的 ReflexNet 推理示例
 *
 * 演示端到端的推理流程:
 * 1. 从 FLASH 加载模型
 * 2. 初始化 CIM 硬件
 * 3. 执行实时推理
 * 4. 与 PID 控制器融合
 */

#include "imc22.h"
#include "imc22_can.h"
#include "imc22_cim.h"
#include "imc22_nvs.h"
#include "imc22_power.h"
#include "model_loader.h"
#include <math.h>
#include <stdio.h>


/* ========================================================================= */
/* 配置                                                                      */
/* ========================================================================= */

#define CONTROL_FREQ_HZ 1000 // 1kHz 控制频率
#define MY_NODE_ID 1         // 节点 ID
#define MAX_TORQUE 10.0f     // 最大力矩 (Nm)

/* ========================================================================= */
/* 全局状态                                                                  */
/* ========================================================================= */

typedef struct {
  // 控制目标
  float target_angle;
  float compliance; // 柔顺系数 (0-1)

  // 传感器数据
  float gyro[3];  // 陀螺仪 (rad/s)
  float accel[3]; // 加速度计 (m/s²)
  float current;  // 电流 (A)
  float angle;    // 当前角度 (rad)

  // PID 状态
  float pid_error_sum;
  float pid_error_prev;

  // 统计
  uint32_t loop_count;
  uint32_t inference_count;
} ControlState_t;

static ControlState_t g_state = {0};
static Model_t g_model;
static InferenceContext_t *g_inference_ctx = NULL;

/* ========================================================================= */
/* PID 控制器                                                                */
/* ========================================================================= */

float PID_Update(float target, float current, float dt) {
  // 从 NVS 加载 PID 参数
  static float kp = 0.0f, ki = 0.0f, kd = 0.0f;
  static bool params_loaded = false;

  if (!params_loaded) {
    kp = NVS_ReadFloat(NVS_KEY_PID_KP, 1.5f);
    ki = NVS_ReadFloat(NVS_KEY_PID_KI, 0.2f);
    kd = NVS_ReadFloat(NVS_KEY_PID_KD, 0.08f);
    params_loaded = true;
  }

  float error = target - current;
  g_state.pid_error_sum += error * dt;
  float error_diff = (error - g_state.pid_error_prev) / dt;
  g_state.pid_error_prev = error;

  return kp * error + ki * g_state.pid_error_sum + kd * error_diff;
}

/* ========================================================================= */
/* 传感器读取                                                                */
/* ========================================================================= */

void ReadSensors(void) {
  // 读取 IMU (模拟数据)
  // 实际应用中需要调用 SPI_Read 等硬件接口

  // 陀螺仪
  g_state.gyro[0] = 0.1f * sinf(g_state.loop_count * 0.01f);
  g_state.gyro[1] = 0.05f * cosf(g_state.loop_count * 0.02f);
  g_state.gyro[2] = 0.02f;

  // 加速度计
  g_state.accel[0] = 0.1f;
  g_state.accel[1] = 0.05f;
  g_state.accel[2] = 9.8f;

  // 电流
  g_state.current = 1.2f + 0.3f * sinf(g_state.loop_count * 0.05f);

  // 角度 (模拟编码器)
  g_state.angle =
      g_state.target_angle + 0.1f * sinf(g_state.loop_count * 0.03f);
}

/* ========================================================================= */
/* 神经反射推理                                                              */
/* ========================================================================= */

float NeuroReflexInference(void) {
  // 准备输入 (12 维)
  float input[12];

  // 当前 IMU 数据 (6)
  input[0] = g_state.gyro[0];
  input[1] = g_state.gyro[1];
  input[2] = g_state.gyro[2];
  input[3] = g_state.accel[0];
  input[4] = g_state.accel[1];
  input[5] = g_state.accel[2];

  // 历史数据 (3) - 简化版本,实际应该保存上一时刻的数据
  input[6] = g_state.gyro[0] * 0.9f;
  input[7] = g_state.gyro[1] * 0.9f;
  input[8] = g_state.gyro[2] * 0.9f;

  // 控制状态 (2)
  input[9] = g_state.current;
  input[10] = g_state.target_angle - g_state.angle; // 角度误差
  input[11] = 0.0f;                                 // 保留

  // 执行推理
  float output[1] = {0.0f};
  if (Inference_Run(g_inference_ctx, input, output) == 0) {
    g_state.inference_count++;
    return output[0]; // 范围 [-1, 1]
  }

  return 0.0f; // 推理失败,返回 0
}

/* ========================================================================= */
/* 控制循环 (1kHz)                                                          */
/* ========================================================================= */

void ControlLoop_1kHz(void) {
  // 1. 读取传感器
  ReadSensors();

  // 2. PID 控制
  float dt = 1.0f / CONTROL_FREQ_HZ;
  float pid_torque = PID_Update(g_state.target_angle, g_state.angle, dt);

  // 3. 神经反射推理
  float reflex_torque = NeuroReflexInference();

  // 4. 混合控制律
  float final_torque = pid_torque * (1.0f - g_state.compliance) +
                       reflex_torque * g_state.compliance * MAX_TORQUE;

  // 5. 限幅
  if (final_torque > MAX_TORQUE)
    final_torque = MAX_TORQUE;
  if (final_torque < -MAX_TORQUE)
    final_torque = -MAX_TORQUE;

  // 6. 输出到电机 (PWM)
  float pwm_duty = (final_torque / MAX_TORQUE) * 50.0f + 50.0f; // 0-100%
  PWM_SetDuty(0, pwm_duty);

  // 7. 统计
  g_state.loop_count++;

  // 8. 每秒显示一次状态
  if (g_state.loop_count % CONTROL_FREQ_HZ == 0) {
    printf("[%6lu] 目标:%.2f° | 当前:%.2f° | PID:%.2f | 反射:%.2f | "
           "最终:%.2f Nm | 推理:%lu\n",
           g_state.loop_count / CONTROL_FREQ_HZ, g_state.target_angle * 57.3f,
           g_state.angle * 57.3f, pid_torque, reflex_torque, final_torque,
           g_state.inference_count);
  }
}

/* ========================================================================= */
/* CAN 接收回调                                                              */
/* ========================================================================= */

void CAN_RxCallback(CAN_Message_t *msg) {
  if (msg->id == 0x200 + MY_NODE_ID) {
    // 解析指令
    int16_t target_deg = (int16_t)((msg->data[0] << 8) | msg->data[1]);
    uint8_t compliance = msg->data[2];

    g_state.target_angle = target_deg * 0.01f * (M_PI / 180.0f); // 度 → 弧度
    g_state.compliance = compliance / 255.0f;

    // 通知电源管理有活动
    Power_NotifyActivity();
  }
}

/* ========================================================================= */
/* 主函数                                                                    */
/* ========================================================================= */

int main(void) {
  printf("\n");
  printf("╔═══════════════════════════════════════════╗\n");
  printf("║  Hive-Reflex 2.0 完整推理示例             ║\n");
  printf("║  神经反射 + PID 混合控制                  ║\n");
  printf("╚═══════════════════════════════════════════╝\n\n");

  // 1. 系统初始化
  printf("1. 系统初始化...\n");
  System_Init();

  // 2. 初始化电源管理
  printf("2. 电源管理初始化...\n");
  Power_Init();
  Power_EnableAutoMode(100);

  // 3. 初始化 NVS
  printf("3. NVS 初始化...\n");
  if (NVS_Init() != 0) {
    printf("错误: NVS 初始化失败\n");
    return -1;
  }

  // 4. 初始化 CIM
  printf("4. CIM 硬件初始化...\n");
  if (CIM_Init() != 0) {
    printf("错误: CIM 初始化失败\n");
    return -1;
  }

  // 5. 从 FLASH 加载模型
  printf("5. 从 FLASH 加载神经网络模型...\n");
  if (Model_LoadFromFlash(MODEL_REFLEX_V2, &g_model) != 0) {
    printf("错误: 模型加载失败\n");
    return -1;
  }

  // 显示模型信息
  Model_PrintInfo(&g_model);

  // 6. 将模型加载到 CIM
  printf("\n6. 加载模型到 CIM SRAM...\n");
  if (Model_LoadToCIM(&g_model, 0) != 0) {
    printf("错误: 无法加载到 CIM\n");
    return -1;
  }

  // 7. 创建推理上下文
  printf("7. 创建推理上下文...\n");
  g_inference_ctx = Inference_CreateContext(&g_model);
  if (!g_inference_ctx) {
    printf("错误: 创建推理上下文失败\n");
    return -1;
  }

  // 8. 初始化 CAN
  printf("8. CAN 总线初始化...\n");
  CAN_Config_t can_cfg = {
      .baudrate = 1000000, .fd_mode = true, .loopback = false};
  CAN_Init(&can_cfg);
  CAN_RegisterRxCallback(CAN_RxCallback);

  // 9. 初始化 PWM
  printf("9. PWM 初始化...\n");
  PWM_Init(20000); // 20kHz

  // 10. 设置默认参数
  g_state.target_angle = 0.0f;
  g_state.compliance = 0.7f;

  printf("\n✓ 初始化完成! 进入控制循环...\n\n");
  printf("提示: 通过 CAN 发送指令到 ID 0x%03X\n", 0x200 + MY_NODE_ID);
  printf("      数据格式: [角度高8位][角度低8位][柔顺度]\n\n");

  // 11. 主控制循环
  uint32_t last_tick = Millis();

  while (1) {
    uint32_t current_tick = Millis();

    // 1kHz 控制频率
    if (current_tick - last_tick >= 1) {
      last_tick = current_tick;
      ControlLoop_1kHz();
    }

    // 更新电源管理
    Power_Update();

    // 每 10 秒显示一次性能统计
    if (g_state.loop_count % (CONTROL_FREQ_HZ * 10) == 0 &&
        g_state.loop_count > 0) {
      uint32_t avg_time, fps_int;
      float fps;
      Inference_GetStats(g_inference_ctx, &avg_time, &fps);
      fps_int = (uint32_t)fps;

      CIM_PerfStats_t cim_stats;
      CIM_GetPerfStats(&cim_stats);

      printf("\n性能统计 (10秒):\n");
      printf("  推理延迟: %lu μs\n", avg_time);
      printf("  推理速率: %lu FPS\n", fps_int);
      printf("  CIM GOPS: %.2f\n", cim_stats.gops);
      printf("  总推理次数: %lu\n\n", g_state.inference_count);
    }
  }

  // 清理 (实际不会执行)
  Inference_DestroyContext(g_inference_ctx);
  Model_Unload(&g_model);

  return 0;
}
