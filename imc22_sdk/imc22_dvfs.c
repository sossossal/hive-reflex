/**
 * @file imc22_dvfs.c
 * @brief DVFS 动态电压频率缩放驱动实现
 */

#include "imc22_dvfs.h"
#include "imc22.h"

/* ========================================================================= */
/* 全局状态                                                                  */
/* ========================================================================= */

static DVFSConfig_t g_dvfs_config = {.enable = false,
                                     .auto_scale = true,
                                     .util_threshold_high = 200,
                                     .util_threshold_low = 50,
                                     .idle_timeout_ms = 1000,
                                     .initial_mode = DVFS_MODE_ACTIVE};

static DVFSState_t g_dvfs_state = {.current_mode = DVFS_MODE_ACTIVE,
                                   .voltage_level = DVFS_VOLTAGE_1_0V,
                                   .freq_level = DVFS_FREQ_100MHZ,
                                   .in_transition = false,
                                   .transition_count = 0,
                                   .time_active_ms = 0,
                                   .time_standby_ms = 0,
                                   .time_deepsleep_ms = 0};

static volatile uint32_t *DVFS_CTRL = (uint32_t *)DVFS_REG_CTRL;
static volatile uint32_t *DVFS_STATUS = (uint32_t *)DVFS_REG_STATUS;
static volatile uint32_t *DVFS_TARGET = (uint32_t *)DVFS_REG_TARGET;
static volatile uint32_t *DVFS_THRESHOLD = (uint32_t *)DVFS_REG_THRESHOLD;
static volatile uint32_t *DVFS_TIMEOUT = (uint32_t *)DVFS_REG_TIMEOUT;

/* 功耗估计表 (mW) */
static const float POWER_TABLE[3] = {
    0.0001f, /* DeepSleep: 100 nW */
    0.5f,    /* Standby: 500 μW */
    50.0f    /* Active: 50 mW */
};

/* ========================================================================= */
/* 公共 API 实现                                                             */
/* ========================================================================= */

int DVFS_Init(const DVFSConfig_t *config) {
  if (config != NULL) {
    g_dvfs_config = *config;
  }

  /* 配置阈值 */
  *DVFS_THRESHOLD = (g_dvfs_config.util_threshold_high << 8) |
                    g_dvfs_config.util_threshold_low;

  /* 配置超时 */
  *DVFS_TIMEOUT = g_dvfs_config.idle_timeout_ms;

  /* 设置初始模式 */
  DVFS_SetMode(g_dvfs_config.initial_mode);

  /* 启用 DVFS */
  if (g_dvfs_config.enable) {
    uint32_t ctrl = DVFS_CTRL_ENABLE;
    if (g_dvfs_config.auto_scale) {
      ctrl |= DVFS_CTRL_AUTO_SCALE;
    }
    *DVFS_CTRL = ctrl;
  }

  g_dvfs_state.transition_count = 0;

  return 0;
}

int DVFS_Enable(void) {
  g_dvfs_config.enable = true;
  *DVFS_CTRL |= DVFS_CTRL_ENABLE;
  return 0;
}

int DVFS_Disable(void) {
  g_dvfs_config.enable = false;
  *DVFS_CTRL &= ~DVFS_CTRL_ENABLE;

  /* 切回 Active 模式 */
  return DVFS_SetMode(DVFS_MODE_ACTIVE);
}

int DVFS_SetMode(DVFSMode_t mode) {
  uint32_t target = 0;

  switch (mode) {
  case DVFS_MODE_DEEPSLEEP:
    target = (DVFS_VOLTAGE_0_4V << 4) | DVFS_FREQ_1MHZ;
    break;
  case DVFS_MODE_STANDBY:
    target = (DVFS_VOLTAGE_0_6V << 4) | DVFS_FREQ_10MHZ;
    break;
  case DVFS_MODE_ACTIVE:
  default:
    target = (DVFS_VOLTAGE_1_0V << 4) | DVFS_FREQ_100MHZ;
    break;
  }

  *DVFS_CTRL |= DVFS_CTRL_FORCE_MODE;
  *DVFS_TARGET = target;

  /* 等待转换完成 */
  if (DVFS_WaitReady(100) != 0) {
    return -1;
  }

  g_dvfs_state.current_mode = mode;
  g_dvfs_state.voltage_level = (target >> 4) & 0x3;
  g_dvfs_state.freq_level = target & 0x3;
  g_dvfs_state.transition_count++;

  *DVFS_CTRL &= ~DVFS_CTRL_FORCE_MODE;

  return 0;
}

DVFSMode_t DVFS_GetMode(void) { return g_dvfs_state.current_mode; }

int DVFS_SetFrequency(DVFSFreq_t freq) {
  uint32_t target = (*DVFS_TARGET & 0xF0) | freq;

  *DVFS_CTRL |= DVFS_CTRL_FORCE_MODE;
  *DVFS_TARGET = target;

  if (DVFS_WaitReady(50) != 0) {
    return -1;
  }

  g_dvfs_state.freq_level = freq;
  *DVFS_CTRL &= ~DVFS_CTRL_FORCE_MODE;

  return 0;
}

int DVFS_SetVoltage(DVFSVoltage_t voltage) {
  uint32_t target = (voltage << 4) | (*DVFS_TARGET & 0x0F);

  *DVFS_CTRL |= DVFS_CTRL_FORCE_MODE;
  *DVFS_TARGET = target;

  if (DVFS_WaitReady(100) != 0) {
    return -1;
  }

  g_dvfs_state.voltage_level = voltage;
  *DVFS_CTRL &= ~DVFS_CTRL_FORCE_MODE;

  return 0;
}

int DVFS_EnableAutoScale(bool enable, uint8_t util_low, uint8_t util_high) {
  g_dvfs_config.auto_scale = enable;
  g_dvfs_config.util_threshold_low = util_low;
  g_dvfs_config.util_threshold_high = util_high;

  *DVFS_THRESHOLD = (util_high << 8) | util_low;

  if (enable) {
    *DVFS_CTRL |= DVFS_CTRL_AUTO_SCALE;
  } else {
    *DVFS_CTRL &= ~DVFS_CTRL_AUTO_SCALE;
  }

  return 0;
}

int DVFS_SetIdleTimeout(uint16_t timeout_ms) {
  g_dvfs_config.idle_timeout_ms = timeout_ms;
  *DVFS_TIMEOUT = timeout_ms;
  return 0;
}

int DVFS_WaitReady(uint32_t timeout_ms) {
  uint32_t start = Millis();

  while ((*DVFS_STATUS & 0x02) != 0) { /* in_transition bit */
    if (Millis() - start > timeout_ms) {
      return -1; /* 超时 */
    }
  }

  g_dvfs_state.in_transition = false;
  return 0;
}

void DVFS_GetState(DVFSState_t *state) {
  if (state != NULL) {
    /* 更新状态 */
    uint32_t status = *DVFS_STATUS;
    g_dvfs_state.current_mode = (status >> 4) & 0x3;
    g_dvfs_state.in_transition = (status & 0x02) != 0;

    /* 读取时间统计 */
    g_dvfs_state.time_active_ms =
        *(volatile uint32_t *)DVFS_REG_TIME_ACTIVE / 100000;
    g_dvfs_state.time_standby_ms =
        *(volatile uint32_t *)DVFS_REG_TIME_STANDBY / 100000;
    g_dvfs_state.time_deepsleep_ms =
        *(volatile uint32_t *)DVFS_REG_TIME_DEEPSLEEP / 100000;
    g_dvfs_state.transition_count = *(volatile uint32_t *)DVFS_REG_TRANS_COUNT;

    *state = g_dvfs_state;
  }
}

void DVFS_GetStats(DVFSStats_t *stats) {
  if (stats == NULL)
    return;

  DVFSState_t state;
  DVFS_GetState(&state);

  uint32_t total_time =
      state.time_active_ms + state.time_standby_ms + state.time_deepsleep_ms;

  if (total_time == 0) {
    stats->avg_power_mw = POWER_TABLE[DVFS_MODE_ACTIVE];
    stats->energy_saved_percent = 0;
    stats->active_ratio_percent = 100;
    return;
  }

  /* 计算平均功耗 */
  float total_energy =
      (state.time_active_ms * POWER_TABLE[DVFS_MODE_ACTIVE]) +
      (state.time_standby_ms * POWER_TABLE[DVFS_MODE_STANDBY]) +
      (state.time_deepsleep_ms * POWER_TABLE[DVFS_MODE_DEEPSLEEP]);

  stats->avg_power_mw = total_energy / total_time;

  /* 计算节能百分比（相对于始终 Active） */
  float always_active_energy = total_time * POWER_TABLE[DVFS_MODE_ACTIVE];
  stats->energy_saved_percent =
      ((always_active_energy - total_energy) / always_active_energy) * 100.0f;

  /* Active 占比 */
  stats->active_ratio_percent = (state.time_active_ms * 100) / total_time;
}

void DVFS_ReportUtilization(uint8_t utilization) {
  *(volatile uint32_t *)DVFS_REG_UTIL = utilization;
}

void DVFS_NotifyActivity(void) {
  /* 重置空闲计时器 */
  *(volatile uint32_t *)(DVFS_BASE_ADDR + 0x28) = 0;

  /* 如果在低功耗模式，切换到 Active */
  if (g_dvfs_state.current_mode != DVFS_MODE_ACTIVE) {
    DVFS_SetMode(DVFS_MODE_ACTIVE);
  }
}

float DVFS_GetEstimatedPower(void) {
  return POWER_TABLE[g_dvfs_state.current_mode];
}

void DVFS_ResetStats(void) {
  g_dvfs_state.time_active_ms = 0;
  g_dvfs_state.time_standby_ms = 0;
  g_dvfs_state.time_deepsleep_ms = 0;
  g_dvfs_state.transition_count = 0;

  /* 重置硬件计数器 */
  *(volatile uint32_t *)DVFS_REG_TIME_ACTIVE = 0;
  *(volatile uint32_t *)DVFS_REG_TIME_STANDBY = 0;
  *(volatile uint32_t *)DVFS_REG_TIME_DEEPSLEEP = 0;
  *(volatile uint32_t *)DVFS_REG_TRANS_COUNT = 0;
}
