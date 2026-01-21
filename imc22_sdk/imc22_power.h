/**
 * @file imc22_power.h
 * @brief RBB (Reverse Body Bias) 电源管理驱动
 * @version 2.0
 * @date 2026-01-19
 * 
 * 支持动态 Body Bias 控制，实现超低功耗待机模式
 */

#ifndef IMC22_POWER_H
#define IMC22_POWER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 寄存器定义                                                                */
/* ========================================================================= */

#define RBB_BASE_ADDR       0x40010000

typedef struct {
    volatile uint32_t CTRL;         // 控制寄存器
    volatile uint32_t STATUS;       // 状态寄存器
    volatile uint32_t VBODY_SET;    // Body 电压设置 (mV, 负值)
    volatile uint32_t VBODY_READ;   // Body 电压读取 (mV)
    volatile uint32_t TIMEOUT;      // 自动进入低功耗的超时时间 (ms)
    volatile uint32_t WAKEUP_SRC;   // 唤醒源配置
} RBB_TypeDef;

#define RBB ((RBB_TypeDef*)RBB_BASE_ADDR)

/* CTRL 寄存器位定义 */
#define RBB_CTRL_ENABLE         (1U << 0)   // 使能 RBB
#define RBB_CTRL_AUTO_MODE      (1U << 1)   // 自动模式
#define RBB_CTRL_FORCE_ACTIVE   (1U << 2)   // 强制 Active 模式
#define RBB_CTRL_FORCE_STANDBY  (1U << 3)   // 强制 Standby 模式
#define RBB_CTRL_FORCE_SLEEP    (1U << 4)   // 强制 Deep Sleep 模式

/* STATUS 寄存器位定义 */
#define RBB_STATUS_ACTIVE       (0U << 0)   // 当前处于 Active 模式
#define RBB_STATUS_STANDBY      (1U << 0)   // 当前处于 Standby 模式
#define RBB_STATUS_SLEEP        (2U << 0)   // 当前处于 Deep Sleep 模式
#define RBB_STATUS_READY        (1U << 8)   // Body Bias 稳定

/* 唤醒源定义 */
#define RBB_WAKEUP_CAN          (1U << 0)   // CAN 消息唤醒
#define RBB_WAKEUP_GPIO         (1U << 1)   // GPIO 中断唤醒
#define RBB_WAKEUP_RTC          (1U << 2)   // RTC 定时器唤醒
#define RBB_WAKEUP_UART         (1U << 3)   // UART 唤醒

/* ========================================================================= */
/* 电源模式定义                                                              */
/* ========================================================================= */

typedef enum {
    PWR_MODE_ACTIVE = 0,    /**< 全速运行，Vbody = 0V，功耗 ~50mW */
    PWR_MODE_STANDBY,       /**< 降频运行，Vbody = -300mV，功耗 ~5mW */
    PWR_MODE_DEEPSLEEP      /**< 仅 CAN 唤醒，Vbody = -500mV，功耗 ~100μW */
} PowerMode_t;

typedef struct {
    PowerMode_t mode;           /**< 当前电源模式 */
    int16_t vbody_mv;           /**< Body 电压 (mV, 负值) */
    uint32_t idle_time_ms;      /**< 系统空闲时间 (ms) */
    uint32_t wakeup_sources;    /**< 使能的唤醒源掩码 */
    bool auto_mode;             /**< 是否启用自动电源管理 */
} PowerState_t;

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化电源管理模块
 * @return 0 成功, -1 失败
 */
int Power_Init(void);

/**
 * @brief 设置电源模式
 * @param mode 目标电源模式
 * @return 0 成功, -1 失败
 */
int Power_SetMode(PowerMode_t mode);

/**
 * @brief 获取当前电源模式
 * @return 当前电源模式
 */
PowerMode_t Power_GetMode(void);

/**
 * @brief 配置唤醒源
 * @param sources 唤醒源掩码 (RBB_WAKEUP_xxx 的组合)
 */
void Power_SetWakeupSources(uint32_t sources);

/**
 * @brief 启用自动电源管理
 * @param idle_threshold_ms 空闲多久后自动进入低功耗 (ms)
 */
void Power_EnableAutoMode(uint32_t idle_threshold_ms);

/**
 * @brief 禁用自动电源管理
 */
void Power_DisableAutoMode(void);

/**
 * @brief 更新电源管理状态 (在主循环中定期调用)
 */
void Power_Update(void);

/**
 * @brief 通知系统有活动发生 (重置空闲计时器)
 */
void Power_NotifyActivity(void);

/**
 * @brief 获取当前 Body 电压
 * @return Body 电压 (mV, 负值)
 */
int16_t Power_GetVbody(void);

/**
 * @brief 获取当前功耗估计
 * @return 功耗估计 (mW)
 */
float Power_GetEstimatedPower(void);

/**
 * @brief 获取电源状态
 * @param state 状态结构体指针
 */
void Power_GetState(PowerState_t *state);

/* ========================================================================= */
/* 内部函数 (供 SDK 内部使用)                                                */
/* ========================================================================= */

/**
 * @brief 设置 Body Bias 电压
 * @param vbody_mv Body 电压 (mV, -500 到 0)
 */
void _Power_SetVbody(int16_t vbody_mv);

/**
 * @brief 等待 Body Bias 稳定
 * @param timeout_us 超时时间 (μs)
 * @return 0 成功, -1 超时
 */
int _Power_WaitReady(uint32_t timeout_us);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_POWER_H */
