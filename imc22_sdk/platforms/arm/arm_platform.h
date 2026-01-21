/**
 * @file arm_platform.h
 * @brief ARM 平台适配层
 *
 * 为 ARM 设备（如 Raspberry Pi）提供 CIM 软件模拟层
 *
 * @version 2.1.0
 */

#ifndef ARM_PLATFORM_H
#define ARM_PLATFORM_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 平台检测                                                                  */
/* ========================================================================= */

#if defined(__arm__) || defined(__aarch64__)
#define ARM_PLATFORM 1
#else
#define ARM_PLATFORM 0
#endif

#ifdef __linux__
#define ARM_LINUX 1
#else
#define ARM_LINUX 0
#endif

/* ========================================================================= */
/* 平台初始化                                                                */
/* ========================================================================= */

/**
 * @brief 初始化 ARM 平台
 * @return 0 成功
 */
int ARM_Platform_Init(void);

/**
 * @brief 检测 Raspberry Pi 型号
 * @return 型号字符串 (如 "Pi 4 Model B")
 */
const char *ARM_DetectRaspberryPi(void);

/**
 * @brief 获取 CPU 频率
 * @return 频率 (Hz)
 */
uint32_t ARM_GetCPUFrequency(void);

/* ========================================================================= */
/* 时间函数                                                                  */
/* ========================================================================= */

/**
 * @brief 获取高精度时间戳 (纳秒)
 */
uint64_t ARM_GetTimeNs(void);

/**
 * @brief 获取毫秒时间戳
 */
uint32_t ARM_GetTimeMs(void);

/**
 * @brief 微秒延迟
 */
void ARM_DelayUs(uint32_t us);

/* ========================================================================= */
/* GPIO 接口 (用于传感器/执行器)                                             */
/* ========================================================================= */

/**
 * @brief GPIO 模式
 */
typedef enum {
  GPIO_INPUT = 0,
  GPIO_OUTPUT,
  GPIO_ALT0,
  GPIO_ALT1,
  GPIO_ALT2
} GPIO_Mode_t;

/**
 * @brief 设置 GPIO 模式
 */
int ARM_GPIO_SetMode(int pin, GPIO_Mode_t mode);

/**
 * @brief 读取 GPIO
 */
int ARM_GPIO_Read(int pin);

/**
 * @brief 写入 GPIO
 */
int ARM_GPIO_Write(int pin, int value);

/**
 * @brief PWM 输出
 */
int ARM_PWM_Write(int pin, int duty_cycle);

/* ========================================================================= */
/* I2C 接口 (用于传感器)                                                     */
/* ========================================================================= */

/**
 * @brief 初始化 I2C
 */
int ARM_I2C_Init(int bus);

/**
 * @brief I2C 读取
 */
int ARM_I2C_Read(int bus, uint8_t addr, uint8_t *data, size_t len);

/**
 * @brief I2C 写入
 */
int ARM_I2C_Write(int bus, uint8_t addr, const uint8_t *data, size_t len);

/* ========================================================================= */
/* SPI 接口                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化 SPI
 */
int ARM_SPI_Init(int bus, uint32_t speed_hz);

/**
 * @brief SPI 传输
 */
int ARM_SPI_Transfer(int bus, const uint8_t *tx, uint8_t *rx, size_t len);

/* ========================================================================= */
/* CIM 软件模拟接口                                                          */
/* ========================================================================= */

/**
 * @brief 初始化 CIM 模拟器
 */
int ARM_CIM_Emulator_Init(void);

/**
 * @brief 设置模拟器参数
 */
void ARM_CIM_Emulator_SetConfig(int mac_count, int data_width);

/**
 * @brief 模拟 CIM MAC 运算
 */
int32_t ARM_CIM_Emulator_MAC(const int8_t *input, const int8_t *weights,
                             size_t count, bool sparse, int threshold);

/**
 * @brief 模拟 CIM 矩阵乘法
 */
int ARM_CIM_Emulator_MatMul(const int8_t *A, const int8_t *B, int32_t *C, int M,
                            int N, int K, bool sparse);

/**
 * @brief 获取模拟器统计
 */
void ARM_CIM_Emulator_GetStats(uint32_t *total_ops, uint32_t *skipped_ops,
                               float *latency_ms);

/* ========================================================================= */
/* NEON SIMD 优化                                                            */
/* ========================================================================= */

#if ARM_PLATFORM && defined(__ARM_NEON)

#include <arm_neon.h>

/**
 * @brief NEON 优化的 int8 向量 MAC
 */
int32_t ARM_NEON_MAC_Int8(const int8_t *a, const int8_t *b, size_t len);

/**
 * @brief NEON 优化的 float32 向量点积
 */
float ARM_NEON_DotProduct_F32(const float *a, const float *b, size_t len);

#endif /* ARM_NEON */

/* ========================================================================= */
/* 扩展接口预留                                                              */
/* ========================================================================= */

/**
 * @brief 注册扩展平台
 * @param name 平台名称
 * @param init_fn 初始化函数
 * @param cleanup_fn 清理函数
 */
int ARM_RegisterPlatformExtension(const char *name, int (*init_fn)(void),
                                  void (*cleanup_fn)(void));

/**
 * @brief 列出可用扩展
 */
void ARM_ListPlatformExtensions(void);

#ifdef __cplusplus
}
#endif

#endif /* ARM_PLATFORM_H */
