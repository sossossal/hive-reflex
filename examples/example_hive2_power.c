/**
 * @file example_hive2_power.c
 * @brief Hive-Reflex 2.0 电源管理示例
 *
 * 演示 RBB 动态电源管理功能
 */

#include "imc22.h"
#include "imc22_can.h"
#include "imc22_power.h"
#include <stdio.h>


// 配置
#define IDLE_THRESHOLD_MS 100 // 100ms 空闲后进入 Standby

// 全局状态
static uint32_t g_can_rx_count = 0;

// CAN 接收回调
void CAN_RxCallback(CAN_Message_t *msg) {
  // 通知电源管理有活动
  Power_NotifyActivity();

  g_can_rx_count++;

  // 处理 CAN 消息...
}

int main(void) {
  printf("Hive-Reflex 2.0 电源管理示例\n");
  printf("============================\n\n");

  // 1. 初始化系统
  System_Init();

  // 2. 初始化电源管理
  if (Power_Init() != 0) {
    printf("错误: 电源管理初始化失败\n");
    return -1;
  }
  printf("✓ 电源管理初始化完成\n");

  // 3. 配置唤醒源
  Power_SetWakeupSources(RBB_WAKEUP_CAN | RBB_WAKEUP_RTC);
  printf("✓ 唤醒源: CAN + RTC\n");

  // 4. 启用自动电源管理
  Power_EnableAutoMode(IDLE_THRESHOLD_MS);
  printf("✓ 自动电源管理使能 (空闲阈值: %d ms)\n", IDLE_THRESHOLD_MS);

  // 5. 初始化 CAN
  CAN_Config_t can_cfg = {
      .baudrate = 1000000, .fd_mode = true, .loopback = false};
  CAN_Init(&can_cfg);
  CAN_RegisterRxCallback(CAN_RxCallback);
  printf("✓ CAN 总线初始化完成\n\n");

  // 6. 主循环
  uint32_t loop_count = 0;
  PowerState_t power_state;

  printf("进入主循环...\n");
  printf("发送 CAN 消息以唤醒系统\n\n");

  while (1) {
    // 更新电源管理 (检查是否需要切换模式)
    Power_Update();

    // 每秒显示状态
    if (loop_count % 1000 == 0) {
      Power_GetState(&power_state);

      const char *mode_str;
      switch (power_state.mode) {
      case PWR_MODE_ACTIVE:
        mode_str = "Active";
        break;
      case PWR_MODE_STANDBY:
        mode_str = "Standby";
        break;
      case PWR_MODE_DEEPSLEEP:
        mode_str = "DeepSleep";
        break;
      default:
        mode_str = "Unknown";
        break;
      }

      printf("[%6lu] 模式: %-10s | Vbody: %4d mV | 空闲: %4lu ms | "
             "功耗: %.1f mW | CAN RX: %lu\n",
             loop_count / 1000, mode_str, power_state.vbody_mv,
             power_state.idle_time_ms, Power_GetEstimatedPower(),
             g_can_rx_count);
    }

    // 低优先级任务
    Delay_ms(1);
    loop_count++;
  }

  return 0;
}
