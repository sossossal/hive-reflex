/**
 * @file imc22_dvfs.h
 * @brief DVFS 动态电压频率缩放 API
 * 
 * 提供 DVFS 控制接口，支持：
 * - 3 级电压域：Active (1.0V), Standby (0.6V), DeepSleep (0.4V)
 * - 4 级频率：100MHz, 50MHz, 10MHz, 1MHz
 * - 自动负载感知缩放
 * 
 * @version 2.1.0
 */

#ifndef IMC22_DVFS_H
#define IMC22_DVFS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 电源模式定义                                                              */
/* ========================================================================= */

typedef enum {
    DVFS_MODE_DEEPSLEEP = 0,    ///< 深度睡眠 (0.4V, 1MHz) - nW 级功耗
    DVFS_MODE_STANDBY   = 1,    ///< 待机 (0.6V, 10MHz) - μW 级功耗
    DVFS_MODE_ACTIVE    = 2,    ///< 活跃 (1.0V, 100MHz) - mW 级功耗
} DVFSMode_t;

typedef enum {
    DVFS_FREQ_1MHZ   = 0,       ///< 1 MHz (分频 /100)
    DVFS_FREQ_10MHZ  = 1,       ///< 10 MHz (分频 /10)
    DVFS_FREQ_50MHZ  = 2,       ///< 50 MHz (分频 /2)
    DVFS_FREQ_100MHZ = 3,       ///< 100 MHz (直通)
} DVFSFreq_t;

typedef enum {
    DVFS_VOLTAGE_0_4V = 0,      ///< 0.4V (DeepSleep)
    DVFS_VOLTAGE_0_6V = 1,      ///< 0.6V (Standby)
    DVFS_VOLTAGE_1_0V = 2,      ///< 1.0V (Active)
} DVFSVoltage_t;

/* ========================================================================= */
/* 配置结构体                                                                */
/* ========================================================================= */

/**
 * @brief DVFS 配置
 */
typedef struct {
    bool enable;                    ///< 是否启用 DVFS
    bool auto_scale;                ///< 是否启用自动缩放
    uint8_t util_threshold_high;    ///< 高利用率阈值 (0-255, 默认 200)
    uint8_t util_threshold_low;     ///< 低利用率阈值 (0-255, 默认 50)
    uint16_t idle_timeout_ms;       ///< 空闲超时进入 DeepSleep (ms)
    DVFSMode_t initial_mode;        ///< 初始模式
} DVFSConfig_t;

/**
 * @brief DVFS 状态
 */
typedef struct {
    DVFSMode_t current_mode;        ///< 当前电源模式
    DVFSVoltage_t voltage_level;    ///< 当前电压等级
    DVFSFreq_t freq_level;          ///< 当前频率等级
    bool in_transition;             ///< 是否正在转换
    uint32_t transition_count;      ///< 模式切换次数
    uint32_t time_active_ms;        ///< Active 模式累计时间
    uint32_t time_standby_ms;       ///< Standby 模式累计时间
    uint32_t time_deepsleep_ms;     ///< DeepSleep 模式累计时间
} DVFSState_t;

/**
 * @brief DVFS 统计
 */
typedef struct {
    float avg_power_mw;             ///< 平均功耗估计 (mW)
    float energy_saved_percent;     ///< 节能百分比
    uint32_t active_ratio_percent;  ///< Active 模式占比
} DVFSStats_t;

/* ========================================================================= */
/* DVFS 寄存器地址                                                           */
/* ========================================================================= */

#define DVFS_BASE_ADDR          0x50000000

#define DVFS_REG_CTRL           (DVFS_BASE_ADDR + 0x00)
#define DVFS_REG_STATUS         (DVFS_BASE_ADDR + 0x04)
#define DVFS_REG_TARGET         (DVFS_BASE_ADDR + 0x08)
#define DVFS_REG_THRESHOLD      (DVFS_BASE_ADDR + 0x0C)
#define DVFS_REG_TIMEOUT        (DVFS_BASE_ADDR + 0x10)
#define DVFS_REG_UTIL           (DVFS_BASE_ADDR + 0x14)
#define DVFS_REG_TIME_ACTIVE    (DVFS_BASE_ADDR + 0x18)
#define DVFS_REG_TIME_STANDBY   (DVFS_BASE_ADDR + 0x1C)
#define DVFS_REG_TIME_DEEPSLEEP (DVFS_BASE_ADDR + 0x20)
#define DVFS_REG_TRANS_COUNT    (DVFS_BASE_ADDR + 0x24)

/* 控制寄存器位定义 */
#define DVFS_CTRL_ENABLE        (1 << 0)
#define DVFS_CTRL_AUTO_SCALE    (1 << 1)
#define DVFS_CTRL_FORCE_MODE    (1 << 2)
#define DVFS_CTRL_IRQ_ENABLE    (1 << 3)

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化 DVFS 控制器
 * @param config 配置参数，NULL 使用默认配置
 * @return 0 成功，负数失败
 */
int DVFS_Init(const DVFSConfig_t *config);

/**
 * @brief 启用 DVFS
 * @return 0 成功
 */
int DVFS_Enable(void);

/**
 * @brief 禁用 DVFS（回到 Active 模式）
 * @return 0 成功
 */
int DVFS_Disable(void);

/**
 * @brief 设置目标电源模式
 * @param mode 目标模式
 * @return 0 成功，负数失败
 */
int DVFS_SetMode(DVFSMode_t mode);

/**
 * @brief 获取当前电源模式
 * @return 当前电源模式
 */
DVFSMode_t DVFS_GetMode(void);

/**
 * @brief 强制设置频率（忽略自动缩放）
 * @param freq 目标频率
 * @return 0 成功
 */
int DVFS_SetFrequency(DVFSFreq_t freq);

/**
 * @brief 强制设置电压（忽略自动缩放）
 * @param voltage 目标电压
 * @return 0 成功
 */
int DVFS_SetVoltage(DVFSVoltage_t voltage);

/**
 * @brief 启用/禁用自动缩放
 * @param enable 是否启用
 * @param util_low 低利用率阈值 (0-255)
 * @param util_high 高利用率阈值 (0-255)
 * @return 0 成功
 */
int DVFS_EnableAutoScale(bool enable, uint8_t util_low, uint8_t util_high);

/**
 * @brief 设置空闲超时
 * @param timeout_ms 超时时间 (ms)
 * @return 0 成功
 */
int DVFS_SetIdleTimeout(uint16_t timeout_ms);

/**
 * @brief 等待 DVFS 转换完成
 * @param timeout_ms 超时时间
 * @return 0 成功，-1 超时
 */
int DVFS_WaitReady(uint32_t timeout_ms);

/**
 * @brief 获取 DVFS 状态
 * @param state 输出状态
 */
void DVFS_GetState(DVFSState_t *state);

/**
 * @brief 获取 DVFS 统计信息
 * @param stats 输出统计
 */
void DVFS_GetStats(DVFSStats_t *stats);

/**
 * @brief 报告 CIM 利用率（供自动缩放使用）
 * @param utilization 利用率 (0-255, 255 = 100%)
 */
void DVFS_ReportUtilization(uint8_t utilization);

/**
 * @brief 通知系统活动（防止进入低功耗）
 */
void DVFS_NotifyActivity(void);

/**
 * @brief 获取预估功耗
 * @return 当前功耗估计 (mW)
 */
float DVFS_GetEstimatedPower(void);

/**
 * @brief 重置统计计数器
 */
void DVFS_ResetStats(void);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_DVFS_H */
