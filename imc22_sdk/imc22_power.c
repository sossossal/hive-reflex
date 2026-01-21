/**
 * @file imc22_power.c
 * @brief RBB 电源管理驱动实现
 */

#include "imc22_power.h"
#include "imc22.h"

/* 全局状态 */
static PowerState_t g_power_state = {.mode = PWR_MODE_ACTIVE,
                                     .vbody_mv = 0,
                                     .idle_time_ms = 0,
                                     .wakeup_sources = RBB_WAKEUP_CAN,
                                     .auto_mode = false};

static uint32_t g_idle_threshold_ms = 100; // 默认 100ms 后进入 Standby
static uint32_t g_last_activity_time = 0;

/* ========================================================================= */
/* 公共 API 实现                                                             */
/* ========================================================================= */

int Power_Init(void) {
  // 使能 RBB 模块
  RBB->CTRL = RBB_CTRL_ENABLE;

  // 设置默认唤醒源 (CAN)
  RBB->WAKEUP_SRC = RBB_WAKEUP_CAN;

  // 初始模式为 Active
  _Power_SetVbody(0);

  // 等待稳定
  if (_Power_WaitReady(1000) != 0) {
    return -1; // 初始化失败
  }

  g_power_state.mode = PWR_MODE_ACTIVE;
  g_last_activity_time = Millis();

  return 0;
}

int Power_SetMode(PowerMode_t mode) {
  int16_t target_vbody = 0;

  switch (mode) {
  case PWR_MODE_ACTIVE:
    target_vbody = 0;
    RBB->CTRL |= RBB_CTRL_FORCE_ACTIVE;
    break;

  case PWR_MODE_STANDBY:
    target_vbody = -300; // -0.3V
    RBB->CTRL |= RBB_CTRL_FORCE_STANDBY;
    break;

  case PWR_MODE_DEEPSLEEP:
    target_vbody = -500; // -0.5V
    RBB->CTRL |= RBB_CTRL_FORCE_SLEEP;
    break;

  default:
    return -1;
  }

  _Power_SetVbody(target_vbody);

  // 等待 Body Bias 稳定 (约 100μs)
  if (_Power_WaitReady(200) != 0) {
    return -1;
  }

  g_power_state.mode = mode;
  g_power_state.vbody_mv = target_vbody;

  return 0;
}

PowerMode_t Power_GetMode(void) { return g_power_state.mode; }

void Power_SetWakeupSources(uint32_t sources) {
  RBB->WAKEUP_SRC = sources;
  g_power_state.wakeup_sources = sources;
}

void Power_EnableAutoMode(uint32_t idle_threshold_ms) {
  g_idle_threshold_ms = idle_threshold_ms;
  g_power_state.auto_mode = true;
  RBB->CTRL |= RBB_CTRL_AUTO_MODE;
  RBB->TIMEOUT = idle_threshold_ms;
}

void Power_DisableAutoMode(void) {
  g_power_state.auto_mode = false;
  RBB->CTRL &= ~RBB_CTRL_AUTO_MODE;
}

void Power_Update(void) {
  if (!g_power_state.auto_mode) {
    return;
  }

  uint32_t current_time = Millis();
  g_power_state.idle_time_ms = current_time - g_last_activity_time;

  // 自动电源管理状态机
  if (g_power_state.idle_time_ms > g_idle_threshold_ms) {
    if (g_power_state.mode == PWR_MODE_ACTIVE) {
      // Active → Standby
      Power_SetMode(PWR_MODE_STANDBY);
    } else if (g_power_state.idle_time_ms > g_idle_threshold_ms * 10) {
      // Standby → Deep Sleep (长时间空闲)
      if (g_power_state.mode == PWR_MODE_STANDBY) {
        Power_SetMode(PWR_MODE_DEEPSLEEP);
      }
    }
  }
}

void Power_NotifyActivity(void) {
  g_last_activity_time = Millis();
  g_power_state.idle_time_ms = 0;

  // 如果处于低功耗模式，立即切换回 Active
  if (g_power_state.mode != PWR_MODE_ACTIVE) {
    Power_SetMode(PWR_MODE_ACTIVE);
  }
}

int16_t Power_GetVbody(void) {
  // 读取实际的 Body 电压
  return (int16_t)RBB->VBODY_READ - 500; // 寄存器值 0-500 映射到 -500mV 到 0mV
}

float Power_GetEstimatedPower(void) {
  // 功耗估计模型 (基于模式)
  switch (g_power_state.mode) {
  case PWR_MODE_ACTIVE:
    return 50.0f; // 50 mW
  case PWR_MODE_STANDBY:
    return 5.0f; // 5 mW
  case PWR_MODE_DEEPSLEEP:
    return 0.1f; // 100 μW
  default:
    return 0.0f;
  }
}

void Power_GetState(PowerState_t *state) {
  if (state) {
    *state = g_power_state;
  }
}

/* ========================================================================= */
/* 内部函数实现                                                              */
/* ========================================================================= */

void _Power_SetVbody(int16_t vbody_mv) {
  // 限制范围 [-500, 0]
  if (vbody_mv < -500)
    vbody_mv = -500;
  if (vbody_mv > 0)
    vbody_mv = 0;

  // 将 mV 转换为寄存器值 (0-500)
  uint32_t reg_val = 500 + vbody_mv;
  RBB->VBODY_SET = reg_val;
}

int _Power_WaitReady(uint32_t timeout_us) {
  uint32_t start = GetCycleCount();
  uint32_t timeout_cycles = timeout_us * (IMC22_SYSCLK_HZ / 1000000);

  while (!(RBB->STATUS & RBB_STATUS_READY)) {
    if (GetCycleCount() - start > timeout_cycles) {
      return -1; // 超时
    }
  }

  return 0;
}
