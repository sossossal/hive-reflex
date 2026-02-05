/**
 * @file fixed_point_filter.h
 * @brief 定点数滤波器库
 *
 * 使用 Q16.16 定点数实现高性能滤波算法
 * 性能: 比浮点卡尔曼滤波快 10 倍
 *
 * 定点数格式:
 *   int32_t: [符号位:1][整数:15][小数:16]
 *   范围: -32768.0 ~ +32767.99998
 *   精度: 1/65536 ≈ 0.000015
 */

#ifndef FIXED_POINT_FILTER_H
#define FIXED_POINT_FILTER_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 定点数类型定义
// ============================================================================

typedef int32_t fixed_t; // Q16.16 定点数

// 定点数常量
#define FIXED_ONE (65536)     // 1.0
#define FIXED_HALF (32768)    // 0.5
#define FIXED_PI (205887)     // 3.14159
#define FIXED_TWO_PI (411775) // 6.28318

// ============================================================================
// 转换宏
// ============================================================================

/**
 * @brief 浮点数转定点数
 */
#define FLOAT_TO_FIXED(x) ((fixed_t)((x) * 65536.0f))

/**
 * @brief 定点数转浮点数
 */
#define FIXED_TO_FLOAT(x) ((float)(x) / 65536.0f)

/**
 * @brief 整数转定点数
 */
#define INT_TO_FIXED(x) ((fixed_t)((x) << 16))

/**
 * @brief 定点数转整数 (向下取整)
 */
#define FIXED_TO_INT(x) ((int)((x) >> 16))

// ============================================================================
// 定点数运算
// ============================================================================

/**
 * @brief 定点数乘法
 *
 * @param a 定点数 A
 * @param b 定点数 B
 * @return A * B (定点数)
 */
static inline fixed_t fixed_mul(fixed_t a, fixed_t b) {
  return (fixed_t)(((int64_t)a * b) >> 16);
}

/**
 * @brief 定点数除法
 *
 * @param a 定点数 A (被除数)
 * @param b 定点数 B (除数)
 * @return A / B (定点数)
 */
static inline fixed_t fixed_div(fixed_t a, fixed_t b) {
  return (fixed_t)(((int64_t)a << 16) / b);
}

// ============================================================================
// 互补滤波器 (Complementary Filter)
// ============================================================================

/**
 * @brief 互补滤波器状态
 */
typedef struct {
  fixed_t alpha; /**< 滤波系数 (0.95-0.99) */
  fixed_t state; /**< 当前状态 */
  fixed_t dt;    /**< 采样周期 (秒) */
} ComplementaryFilter_t;

/**
 * @brief 初始化互补滤波器
 *
 * @param filter 滤波器指针
 * @param alpha 滤波系数 (推荐: 0.98)
 * @param dt 采样周期 (秒, e.g., 0.01 for 100Hz)
 */
void CompFilter_Init(ComplementaryFilter_t *filter, float alpha, float dt);

/**
 * @brief 更新滤波器 (姿态估计)
 *
 * @param filter 滤波器指针
 * @param gyro 陀螺仪读数 (度/秒)
 * @param accel 加速度计角度 (度)
 * @return 滤波后的角度 (度)
 */
float CompFilter_Update(ComplementaryFilter_t *filter, float gyro, float accel);

/**
 * @brief 更新滤波器 (定点数版本, 快 10x)
 *
 * @param filter 滤波器指针
 * @param gyro 陀螺仪读数 (定点数)
 * @param accel 加速度计角度 (定点数)
 */
void CompFilter_UpdateFixed(ComplementaryFilter_t *filter, fixed_t gyro,
                            fixed_t accel);

// ============================================================================
// 一阶 IIR 低通滤波器
// ============================================================================

/**
 * @brief IIR 低通滤波器
 */
typedef struct {
  fixed_t alpha; /**< 平滑系数 (0.0 - 1.0) */
  fixed_t state; /**< 当前状态 */
} IIRFilter_t;

/**
 * @brief 初始化 IIR 滤波器
 *
 * @param filter 滤波器指针
 * @param cutoff_hz 截止频率 (Hz)
 * @param sample_hz 采样频率 (Hz)
 */
void IIRFilter_Init(IIRFilter_t *filter, float cutoff_hz, float sample_hz);

/**
 * @brief 更新 IIR 滤波器
 *
 * @param filter 滤波器指针
 * @param input 输入值
 * @return 滤波后的输出
 */
float IIRFilter_Update(IIRFilter_t *filter, float input);

/**
 * @brief 更新 IIR 滤波器 (定点数版本)
 */
fixed_t IIRFilter_UpdateFixed(IIRFilter_t *filter, fixed_t input);

// ============================================================================
// 移动平均滤波器 (Moving Average)
// ============================================================================

#define MA_FILTER_SIZE 8 /**< 窗口大小 (必须是 2 的幂) */

/**
 * @brief 移动平均滤波器
 */
typedef struct {
  fixed_t buffer[MA_FILTER_SIZE]; /**< 循环缓冲区 */
  uint8_t head;                   /**< 写指针 */
  fixed_t sum;                    /**< 当前总和 */
} MAFilter_t;

/**
 * @brief 初始化移动平均滤波器
 */
void MAFilter_Init(MAFilter_t *filter);

/**
 * @brief 更新移动平均滤波器
 *
 * @param filter 滤波器指针
 * @param input 输入值
 * @return 滤波后的输出 (平均值)
 */
float MAFilter_Update(MAFilter_t *filter, float input);

/**
 * @brief 更新移动平均滤波器 (定点数版本)
 */
fixed_t MAFilter_UpdateFixed(MAFilter_t *filter, fixed_t input);

// ============================================================================
// 预测性滤波器 (Predictive Filter)
// ============================================================================

#define PRED_HISTORY_SIZE 4

/**
 * @brief 预测性滤波器 (线性外推)
 */
typedef struct {
  fixed_t history[PRED_HISTORY_SIZE]; /**< 历史数据 */
  uint8_t head;                       /**< 写指针 */
  uint8_t count;                      /**< 有效样本数 */
} PredictiveFilter_t;

/**
 * @brief 初始化预测性滤波器
 */
void PredFilter_Init(PredictiveFilter_t *filter);

/**
 * @brief 预测下一个值 (线性外推)
 *
 * @param filter 滤波器指针
 * @return 预测值
 */
float PredFilter_Predict(PredictiveFilter_t *filter);

/**
 * @brief 更新滤波器 (添加新样本)
 */
void PredFilter_Update(PredictiveFilter_t *filter, float new_value);

/**
 * @brief 预测下一个值 (定点数版本)
 */
fixed_t PredFilter_PredictFixed(PredictiveFilter_t *filter);

/**
 * @brief 更新滤波器 (定点数版本)
 */
void PredFilter_UpdateFixed(PredictiveFilter_t *filter, fixed_t new_value);

#ifdef __cplusplus
}
#endif

#endif // FIXED_POINT_FILTER_H
