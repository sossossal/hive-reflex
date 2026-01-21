/**
 * @file imc22_pwm.h
 * @brief IMC-22 PWM 驱动接口 (用于电机控制)
 */

#ifndef IMC22_PWM_H
#define IMC22_PWM_H

#include "imc22.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== PWM 寄存器结构 ========== */
typedef struct {
  vuint32_t CTRL;     // 控制寄存器
  vuint32_t PERIOD;   // 周期值
  vuint32_t DUTY[4];  // 占空比值 (4 通道)
  vuint32_t DEADTIME; // 死区时间
} PWM_TypeDef;

#define PWM ((PWM_TypeDef *)PWM_BASE)

/* PWM 控制位 */
#define PWM_CTRL_EN (1 << 0)     // 使能
#define PWM_CTRL_CH0_EN (1 << 1) // 通道 0 使能
#define PWM_CTRL_CH1_EN (1 << 2) // 通道 1 使能
#define PWM_CTRL_INVERT (1 << 8) // 反相输出

/* ========== 函数声明 ========== */

/**
 * @brief 初始化 PWM
 * @param freq_hz PWM 频率 (Hz)
 */
void PWM_Init(uint32_t freq_hz);

/**
 * @brief 设置 PWM 占空比
 * @param channel 通道 (0-3)
 * @param duty_percent 占空比 (0.0 - 100.0)
 */
void PWM_SetDuty(uint8_t channel, float duty_percent);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_PWM_H */
