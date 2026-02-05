/**
 * @file fixed_point_filter.c
 * @brief 定点数滤波器实现
 */

#include "fixed_point_filter.h"
#include <string.h>

// ============================================================================
// 互补滤波器
// ============================================================================

void CompFilter_Init(ComplementaryFilter_t *filter, float alpha, float dt) {
  if (!filter)
    return;

  filter->alpha = FLOAT_TO_FIXED(alpha);
  filter->dt = FLOAT_TO_FIXED(dt);
  filter->state = 0;
}

float CompFilter_Update(ComplementaryFilter_t *filter, float gyro,
                        float accel) {
  if (!filter)
    return 0.0f;

  // 转换为定点数
  fixed_t gyro_fixed = FLOAT_TO_FIXED(gyro);
  fixed_t accel_fixed = FLOAT_TO_FIXED(accel);

  // 定点数计算
  CompFilter_UpdateFixed(filter, gyro_fixed, accel_fixed);

  // 转回浮点数
  return FIXED_TO_FLOAT(filter->state);
}

void CompFilter_UpdateFixed(ComplementaryFilter_t *filter, fixed_t gyro,
                            fixed_t accel) {
  if (!filter)
    return;

  // state = alpha * (state + gyro*dt) + (1-alpha) * accel

  // Step 1: state + gyro * dt
  fixed_t gyro_integration = filter->state + fixed_mul(gyro, filter->dt);

  // Step 2: alpha * (state + gyro*dt)
  fixed_t high_pass = fixed_mul(filter->alpha, gyro_integration);

  // Step 3: (1 - alpha) * accel
  fixed_t one_minus_alpha = FIXED_ONE - filter->alpha;
  fixed_t low_pass = fixed_mul(one_minus_alpha, accel);

  // Step 4: 合并
  filter->state = high_pass + low_pass;
}

// ============================================================================
// IIR 低通滤波器
// ============================================================================

void IIRFilter_Init(IIRFilter_t *filter, float cutoff_hz, float sample_hz) {
  if (!filter)
    return;

  // alpha = dt / (RC + dt), where RC = 1 / (2*pi*fc)
  float dt = 1.0f / sample_hz;
  float RC = 1.0f / (2.0f * 3.14159f * cutoff_hz);
  float alpha = dt / (RC + dt);

  filter->alpha = FLOAT_TO_FIXED(alpha);
  filter->state = 0;
}

float IIRFilter_Update(IIRFilter_t *filter, float input) {
  if (!filter)
    return 0.0f;

  fixed_t input_fixed = FLOAT_TO_FIXED(input);
  IIRFilter_UpdateFixed(filter, input_fixed);

  return FIXED_TO_FLOAT(filter->state);
}

fixed_t IIRFilter_UpdateFixed(IIRFilter_t *filter, fixed_t input) {
  if (!filter)
    return 0;

  // state = alpha * input + (1 - alpha) * state
  fixed_t weighted_input = fixed_mul(filter->alpha, input);
  fixed_t one_minus_alpha = FIXED_ONE - filter->alpha;
  fixed_t weighted_state = fixed_mul(one_minus_alpha, filter->state);

  filter->state = weighted_input + weighted_state;
  return filter->state;
}

// ============================================================================
// 移动平均滤波器
// ============================================================================

void MAFilter_Init(MAFilter_t *filter) {
  if (!filter)
    return;

  memset(filter->buffer, 0, sizeof(filter->buffer));
  filter->head = 0;
  filter->sum = 0;
}

float MAFilter_Update(MAFilter_t *filter, float input) {
  if (!filter)
    return 0.0f;

  fixed_t input_fixed = FLOAT_TO_FIXED(input);
  fixed_t result = MAFilter_UpdateFixed(filter, input_fixed);

  return FIXED_TO_FLOAT(result);
}

fixed_t MAFilter_UpdateFixed(MAFilter_t *filter, fixed_t input) {
  if (!filter)
    return 0;

  // 移除最旧的样本
  filter->sum -= filter->buffer[filter->head];

  // 添加新样本
  filter->buffer[filter->head] = input;
  filter->sum += input;

  // 更新指针 (循环)
  filter->head = (filter->head + 1) & (MA_FILTER_SIZE - 1);

  // 返回平均值 (除以 MA_FILTER_SIZE, 使用移位代替)
  return filter->sum >> 3; // 假设 MA_FILTER_SIZE = 8
}

// ============================================================================
// 预测性滤波器
// ============================================================================

void PredFilter_Init(PredictiveFilter_t *filter) {
  if (!filter)
    return;

  memset(filter->history, 0, sizeof(filter->history));
  filter->head = 0;
  filter->count = 0;
}

float PredFilter_Predict(PredictiveFilter_t *filter) {
  if (!filter)
    return 0.0f;

  fixed_t result = PredFilter_PredictFixed(filter);
  return FIXED_TO_FLOAT(result);
}

void PredFilter_Update(PredictiveFilter_t *filter, float new_value) {
  if (!filter)
    return;

  fixed_t value_fixed = FLOAT_TO_FIXED(new_value);
  PredFilter_UpdateFixed(filter, value_fixed);
}

fixed_t PredFilter_PredictFixed(PredictiveFilter_t *filter) {
  if (!filter || filter->count < 2) {
    // 样本不足，返回最新值
    return filter->history[filter->head];
  }

  // 线性外推: y_next = 2*y[0] - y[1]
  uint8_t idx0 = filter->head;
  uint8_t idx1 = (filter->head + PRED_HISTORY_SIZE - 1) % PRED_HISTORY_SIZE;

  fixed_t y0 = filter->history[idx0];
  fixed_t y1 = filter->history[idx1];

  return (y0 << 1) - y1; // 2*y0 - y1
}

void PredFilter_UpdateFixed(PredictiveFilter_t *filter, fixed_t new_value) {
  if (!filter)
    return;

  // 更新指针
  filter->head = (filter->head + 1) % PRED_HISTORY_SIZE;

  // 存储新值
  filter->history[filter->head] = new_value;

  // 更新计数
  if (filter->count < PRED_HISTORY_SIZE) {
    filter->count++;
  }
}
