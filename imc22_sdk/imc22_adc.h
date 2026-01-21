/**
 * @file imc22_adc.h
 * @brief IMC-22 ADC 驱动接口 (用于电流传感器)
 */

#ifndef IMC22_ADC_H
#define IMC22_ADC_H

#include "imc22.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== ADC 寄存器结构 ========== */
typedef struct {
  vuint32_t CTRL;        // 控制寄存器
  vuint32_t STATUS;      // 状态寄存器
  vuint32_t DATA[8];     // 数据寄存器 (8 通道)
  vuint32_t SAMPLE_TIME; // 采样时间配置
} ADC_TypeDef;

#define ADC ((ADC_TypeDef *)ADC_BASE)

/* ========== 函数声明 ========== */

void ADC_Init(void);
uint16_t ADC_Read(uint8_t channel);
float ADC_ReadVoltage(uint8_t channel, float vref);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_ADC_H */
