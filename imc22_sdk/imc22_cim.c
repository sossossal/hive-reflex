/**
 * @file imc22_cim.c
 * @brief Digital CIM 硬件加速器实现
 */

#include "imc22_cim.h"
#include "imc22.h"
#include "imc22_nvs.h"
#include <math.h>
#include <string.h>


/* 全局状态 */
static bool g_cim_initialized = false;
static CIM_PerfStats_t g_perf_stats = {0};

/* SRAM Bank 管理 */
#define CIM_NUM_BANKS 4
#define CIM_BANK_SIZE (128 * 1024) // 128KB per bank

static uint32_t g_bank_usage[CIM_NUM_BANKS] = {0};

/* ========================================================================= */
/* 内部辅助函数                                                              */
/* ========================================================================= */

static uint32_t _CIM_GetCycleCount(void) {
  // 读取 CPU 周期计数器
  uint32_t cycles;
  __asm__ volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

static void _CIM_UpdatePerfStats(uint32_t ops, uint32_t cycles) {
  g_perf_stats.total_ops += ops;
  g_perf_stats.total_cycles += cycles;
  g_perf_stats.total_time_us += cycles / (IMC22_SYSCLK_HZ / 1000000);

  // 计算 GOPS
  if (g_perf_stats.total_time_us > 0) {
    g_perf_stats.gops = (float)g_perf_stats.total_ops /
                        (g_perf_stats.total_time_us / 1000000.0f) / 1e9f;
  }
}

/* ========================================================================= */
/* 公共 API 实现                                                             */
/* ========================================================================= */

int CIM_Init(void) {
  if (g_cim_initialized) {
    return 0; // 已经初始化
  }

  // 复位 CIM
  CIM->CTRL = CIM_CTRL_RESET;
  Delay_us(10);
  CIM->CTRL = 0;

  // 等待就绪
  uint32_t timeout = 10000;
  while ((CIM->STATUS & CIM_STATUS_BUSY) && timeout--) {
    Delay_us(1);
  }

  if (timeout == 0) {
    return -1; // 超时
  }

  // 清空 Bank 使用记录
  memset(g_bank_usage, 0, sizeof(g_bank_usage));

  // 重置性能统计
  CIM_ResetPerfStats();

  g_cim_initialized = true;
  return 0;
}

void CIM_Reset(void) {
  CIM->CTRL = CIM_CTRL_RESET;
  Delay_us(10);
  CIM->CTRL = 0;
  g_cim_initialized = false;
}

int CIM_MatMul(const CIM_Matrix_t *A, const CIM_Matrix_t *B, CIM_Matrix_t *C,
               const CIM_QuantParam_t *quant) {
  if (!g_cim_initialized) {
    return -1;
  }

  // 维度检查
  if (A->cols != B->rows) {
    return -1; // 维度不匹配
  }

  uint32_t M = A->rows;
  uint32_t K = A->cols;
  uint32_t N = B->cols;

  // 配置 CIM
  CIM->DIM_M = M;
  CIM->DIM_N = N;
  CIM->DIM_K = K;
  CIM->INPUT_ADDR = (uint32_t)A->data;
  CIM->WEIGHT_ADDR = (uint32_t)B->data;
  CIM->OUTPUT_ADDR = (uint32_t)C->data;
  CIM->OP_TYPE = CIM_OP_MATMUL;

  // 量化参数
  if (quant) {
    CIM->CTRL |= CIM_CTRL_INT8_MODE;
    CIM->QUANT_SCALE = *(uint32_t *)&quant->scale; // 类型双关
    CIM->QUANT_ZERO = quant->zero_point;
  } else {
    CIM->CTRL &= ~CIM_CTRL_INT8_MODE;
  }

  // 性能计数
  uint32_t start_cycles = _CIM_GetCycleCount();

  // 启动计算
  CIM->CTRL |= CIM_CTRL_START;

  // 等待完成
  if (CIM_WaitDone(1000) != 0) {
    return -1; // 超时
  }

  // 更新性能统计
  uint32_t cycles = _CIM_GetCycleCount() - start_cycles;
  uint32_t ops = 2 * M * N * K; // 乘法 + 加法
  _CIM_UpdatePerfStats(ops, cycles);

  // 输出维度
  C->rows = M;
  C->cols = N;
  C->dtype = quant ? CIM_DTYPE_INT8 : CIM_DTYPE_FP32;

  return 0;
}

int CIM_LSTM(const float *input, const float *h_prev, const float *c_prev,
             float *h_next, float *c_next, void *weights) {
  if (!g_cim_initialized) {
    return -1;
  }

  // 配置 LSTM 操作
  CIM->INPUT_ADDR = (uint32_t)input;
  CIM->WEIGHT_ADDR = (uint32_t)weights;
  CIM->OUTPUT_ADDR = (uint32_t)h_next;
  CIM->OP_TYPE = CIM_OP_LSTM_GATE;

  // 性能计数
  uint32_t start_cycles = _CIM_GetCycleCount();

  // 启动计算
  CIM->CTRL |= CIM_CTRL_START;

  // 等待完成
  if (CIM_WaitDone(2000) != 0) {
    return -1; // 超时
  }

  // 更新性能统计
  uint32_t cycles = _CIM_GetCycleCount() - start_cycles;
  _CIM_UpdatePerfStats(1000, cycles); // LSTM 操作近似 1000 ops

  return 0;
}

int CIM_LoadWeights(const void *weights, uint32_t size, uint8_t bank) {
  if (bank >= CIM_NUM_BANKS) {
    return -1; // 无效的 Bank
  }

  if (size > CIM_BANK_SIZE) {
    return -1; // 超出 Bank 容量
  }

  // 计算目标地址
  uint32_t bank_addr = CIM_BASE_ADDR + (bank * CIM_BANK_SIZE);

  // 复制权重到 CIM SRAM
  memcpy((void *)bank_addr, weights, size);

  // 更新使用记录
  g_bank_usage[bank] = size;

  return 0;
}

int CIM_WaitDone(uint32_t timeout_us) {
  uint32_t start = _CIM_GetCycleCount();
  uint32_t timeout_cycles = timeout_us * (IMC22_SYSCLK_HZ / 1000000);

  while (CIM->STATUS & CIM_STATUS_BUSY) {
    if ((_CIM_GetCycleCount() - start) > timeout_cycles) {
      return -1; // 超时
    }
  }

  // 检查错误
  if (CIM->STATUS & CIM_STATUS_ERROR) {
    return -1;
  }

  return 0;
}

bool CIM_IsBusy(void) { return (CIM->STATUS & CIM_STATUS_BUSY) != 0; }

void CIM_EnableIRQ(bool enable) {
  if (enable) {
    CIM->IRQ_ENABLE = 1;
  } else {
    CIM->IRQ_ENABLE = 0;
  }
}

/* ========================================================================= */
/* 高级 API 实现                                                             */
/* ========================================================================= */

int CIM_FullyConnected(const float *input, float *output, const void *weights,
                       const float *bias, uint32_t input_size,
                       uint32_t output_size, uint8_t activation) {
  // 准备矩阵
  CIM_Matrix_t A = {.data = (void *)input,
                    .rows = 1,
                    .cols = input_size,
                    .dtype = CIM_DTYPE_FP32};

  CIM_Matrix_t B = {.data = (void *)weights,
                    .rows = input_size,
                    .cols = output_size,
                    .dtype = CIM_DTYPE_FP32};

  CIM_Matrix_t C = {
      .data = output, .rows = 1, .cols = output_size, .dtype = CIM_DTYPE_FP32};

  // 矩阵乘法
  if (CIM_MatMul(&A, &B, &C, NULL) != 0) {
    return -1;
  }

  // 加偏置
  if (bias) {
    for (uint32_t i = 0; i < output_size; i++) {
      output[i] += bias[i];
    }
  }

  // 激活函数
  switch (activation) {
  case 1: // ReLU
    CIM_ReLU(output, output_size);
    break;
  case 2: // Sigmoid
    CIM_Sigmoid(output, output_size);
    break;
  case 3: // Tanh
    CIM_Tanh(output, output_size);
    break;
  default:
    break;
  }

  return 0;
}

void CIM_ReLU(float *data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (data[i] < 0.0f) {
      data[i] = 0.0f;
    }
  }
}

void CIM_Sigmoid(float *data, uint32_t size) {
  // 使用查找表 (LUT) 加速
  for (uint32_t i = 0; i < size; i++) {
    data[i] = 1.0f / (1.0f + expf(-data[i]));
  }
}

void CIM_Tanh(float *data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    data[i] = tanhf(data[i]);
  }
}

/* ========================================================================= */
/* 性能分析                                                                  */
/* ========================================================================= */

void CIM_GetPerfStats(CIM_PerfStats_t *stats) {
  if (stats) {
    *stats = g_perf_stats;
  }
}

void CIM_ResetPerfStats(void) {
  memset(&g_perf_stats, 0, sizeof(g_perf_stats));
}
