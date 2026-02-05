/**
 * @file system_optimization_demo.c
 * @brief 系统级优化综合演示
 *
 * 展示如何使用:
 * 1. 定点滤波器 (10x 加速)
 * 2. CAN SYNC 同步 (消除抖动)
 * 3. TCM 加速 (单周期访问)
 *
 * 场景: 平衡控制循环
 */

#include "middleware/can_sync.h"
#include "middleware/fixed_point_filter.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// ============================================================================
// 函数前置声明
// ============================================================================

// 控制循环
void RunControlLoop(void);
void SendControlCommand(void);

// 传感器读取
float ReadGyro(void);
float ReadAccel(void);

// 硬件抽象
void SetMotorPWM(float duty);
void CAN_EnableRx(void);
void CAN_Transmit(uint16_t id, const void *data, uint8_t dlc);
void CAN_Receive(uint16_t *id, uint8_t *data, uint8_t *dlc);

// 时间函数
uint32_t GetTimestampUs(void);
uint32_t GetTickMs(void);
void DelayMs(uint32_t ms);
void DelayUs(uint32_t us);

// CAN 中断处理
void CAN_RX_IRQHandler(void);

// CAN ID 定义 (从 can_sync.h 复制)
#ifndef CAN_SYNC_ID
#define CAN_SYNC_ID 0x000
#define CAN_ID_CONTROL 0x100
#endif

// ============================================================================
// 全局变量 (将被放置在 TCM 中)
// ============================================================================

// 标记为 TCM section (通过链接脚本实现)
__attribute__((section(".tcm_data"))) static ComplementaryFilter_t angle_filter;

__attribute__((section(".tcm_data"))) static float current_angle = 0.0f;

__attribute__((section(".tcm_data"))) static float target_angle = 0.0f;

// ============================================================================
// CAN SYNC 回调 (时间触发任务)
// ============================================================================

static CANSyncMgr_t sync_mgr;

void on_sync_received(uint32_t phase_us) {
  // SYNC 帧到达，执行时间触发任务

  // Phase 100μs: 采样 IMU
  if (CANSync_IsPhase(&sync_mgr, 100, 20)) {
    float gyro = ReadGyro();   // 陀螺仪
    float accel = ReadAccel(); // 加速度计

    // 使用定点滤波器 (快 10x)
    current_angle = CompFilter_Update(&angle_filter, gyro, accel);
  }

  // Phase 500μs: 运行控制循环
  if (CANSync_IsPhase(&sync_mgr, 500, 20)) {
    RunControlLoop();
  }

  // Phase 800μs: 发送控制指令
  if (CANSync_IsPhase(&sync_mgr, 800, 20)) {
    SendControlCommand();
  }
}

// ============================================================================
// 控制循环 (标记为 TCM 代码)
// ============================================================================

__attribute__((section(".tcm_text"))) void RunControlLoop(void) {
  // PID 控制 (在 TCM 中执行, 单周期访问全局变量)
  float error = target_angle - current_angle;
  float Kp = 1.5f;
  float output = Kp * error;

  // 输出到电机
  SetMotorPWM(output);
}

// ============================================================================
// 主函数
// ============================================================================

int main(void) {
  printf("==============================================\n");
  printf(" System Optimization Demo\n");
  printf("==============================================\n");

  // 1. 初始化定点滤波器
  CompFilter_Init(&angle_filter, 0.98f, 0.001f); // 98% gyro, 1ms
  printf("[✓] Complementary Filter initialized\n");

  // 2. 初始化 CAN SYNC (Slave 模式)
  CANSync_Init(&sync_mgr, false, on_sync_received);
  printf("[✓] CAN SYNC initialized (Slave)\n");

  // 3. 启动 CAN 接收
  CAN_EnableRx();
  printf("[✓] CAN RX enabled\n");

  printf("\n--- Waiting for SYNC frames ---\n");

  // 主循环
  while (1) {
    // 检查 SYNC 超时
    if (CANSync_IsTimeout(&sync_mgr)) {
      printf("[!] SYNC timeout, waiting for Master...\n");
      DelayMs(100);
    }

    // 每秒打印一次统计
    static uint32_t last_print = 0;
    if (GetTickMs() - last_print > 1000) {
      last_print = GetTickMs();

      printf("\n--- Status ---\n");
      printf("Current Angle: %.2f°\n", current_angle);
      printf("Target Angle:  %.2f°\n", target_angle);

      CANSync_PrintStats(&sync_mgr);
    }

    DelayMs(10);
  }

  return 0;
}

// ============================================================================
// HAL 接口实现 (模拟)
// ============================================================================

float ReadGyro(void) {
  // TODO: 实际 SPI 读取
  return 0.5f; // 模拟值
}

float ReadAccel(void) {
  // TODO: 实际 I2C 读取
  return 2.0f; // 模拟值
}

void SetMotorPWM(float duty) {
  // TODO: 实际 PWM 输出
  (void)duty;
}

void SendControlCommand(void) {
  // TODO: 通过 CAN 发送指令到其他关节
  typedef struct {
    float angle;
    float velocity;
  } ControlCmd_t;

  ControlCmd_t cmd = {.angle = current_angle, .velocity = 0.0f};

  CAN_Transmit(CAN_ID_CONTROL, &cmd, sizeof(cmd));
}

// CAN RX 中断
void CAN_RX_IRQHandler(void) {
  uint16_t can_id;
  uint8_t data[8];
  uint8_t dlc;

  CAN_Receive(&can_id, data, &dlc);

  if (can_id == CAN_SYNC_ID) {
    CANSyncFrame_t *sync = (CANSyncFrame_t *)data;
    CANSync_OnSyncReceived(&sync_mgr, sync);
  }
}
