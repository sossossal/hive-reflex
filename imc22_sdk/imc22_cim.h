/**
 * @file imc22_cim.h
 * @brief Digital CIM (Compute-in-Memory) 硬件加速器接口
 * @version 2.0
 * @date 2026-01-19
 *
 * 支持矩阵乘法、LSTM 单元的存内计算加速
 */

#ifndef IMC22_CIM_H
#define IMC22_CIM_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 寄存器定义                                                                */
/* ========================================================================= */

#define CIM_BASE_ADDR 0x50000000
#define CIM_SRAM_SIZE (512 * 1024) // 512KB

typedef struct {
  volatile uint32_t CTRL;        // 控制寄存器
  volatile uint32_t STATUS;      // 状态寄存器
  volatile uint32_t INPUT_ADDR;  // 输入数据地址
  volatile uint32_t WEIGHT_ADDR; // 权重地址
  volatile uint32_t OUTPUT_ADDR; // 输出缓冲区地址
  volatile uint32_t DIM_M;       // 矩阵维度 M
  volatile uint32_t DIM_N;       // 矩阵维度 N
  volatile uint32_t DIM_K;       // 矩阵维度 K
  volatile uint32_t QUANT_SCALE; // INT8 量化缩放因子 (FP32)
  volatile uint32_t QUANT_ZERO;  // INT8 量化零点
  volatile uint32_t OP_TYPE;     // 操作类型
  volatile uint32_t IRQ_ENABLE;  // 中断使能
} CIM_TypeDef;

#define CIM ((CIM_TypeDef *)CIM_BASE_ADDR)

/* CTRL 寄存器位定义 */
#define CIM_CTRL_START (1U << 0)      // 启动计算
#define CIM_CTRL_RESET (1U << 1)      // 复位 CIM
#define CIM_CTRL_INT8_MODE (1U << 2)  // INT8 模式
#define CIM_CTRL_FP32_MODE (0U << 2)  // FP32 模式
#define CIM_CTRL_ACCUMULATE (1U << 3) // 累加模式

/* STATUS 寄存器位定义 */
#define CIM_STATUS_BUSY (1U << 0)  // CIM 忙碌
#define CIM_STATUS_DONE (1U << 1)  // 计算完成
#define CIM_STATUS_ERROR (1U << 2) // 错误

/* 操作类型 */
#define CIM_OP_MATMUL 0    // 矩阵乘法
#define CIM_OP_LSTM_GATE 1 // LSTM 门控运算
#define CIM_OP_CONV2D 2    // 2D 卷积
#define CIM_OP_POOL 3      // 池化

/* ========================================================================= */
/* 数据类型定义                                                              */
/* ========================================================================= */

typedef enum {
  CIM_DTYPE_INT8 = 0, /**< 8-bit 整数 */
  CIM_DTYPE_FP16,     /**< 16-bit 浮点 */
  CIM_DTYPE_FP32      /**< 32-bit 浮点 */
} CIM_DataType_t;

typedef struct {
  void *data;           /**< 数据指针 */
  uint32_t rows;        /**< 行数 (M) */
  uint32_t cols;        /**< 列数 (N) */
  CIM_DataType_t dtype; /**< 数据类型 */
} CIM_Matrix_t;

typedef struct {
  float scale;        /**< 量化缩放因子 */
  int32_t zero_point; /**< 量化零点 */
} CIM_QuantParam_t;

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化 CIM 模块
 * @return 0 成功, -1 失败
 */
int CIM_Init(void);

/**
 * @brief 复位 CIM 模块
 */
void CIM_Reset(void);

/**
 * @brief 矩阵乘法: C = A × B
 * @param A 矩阵 A (M x K)
 * @param B 矩阵 B (K x N)
 * @param C 输出矩阵 C (M x N)
 * @param quant 量化参数 (NULL 表示 FP32 模式)
 * @return 0 成功, -1 失败
 */
int CIM_MatMul(const CIM_Matrix_t *A, const CIM_Matrix_t *B, CIM_Matrix_t *C,
               const CIM_QuantParam_t *quant);

/**
 * @brief LSTM 门控运算 (硬件加速)
 * @param input 输入向量 (batch x input_size)
 * @param h_prev 上一时刻隐藏状态 (batch x hidden_size)
 * @param c_prev 上一时刻细胞状态 (batch x hidden_size)
 * @param h_next 输出隐藏状态 (batch x hidden_size)
 * @param c_next 输出细胞状态 (batch x hidden_size)
 * @param weights LSTM 权重 (预加载到 CIM SRAM)
 * @return 0 成功, -1 失败
 */
int CIM_LSTM(const float *input, const float *h_prev, const float *c_prev,
             float *h_next, float *c_next, void *weights);

/**
 * @brief 加载权重到 CIM SRAM
 * @param weights 权重数据
 * @param size 数据大小 (字节)
 * @param bank 目标 Bank (0-3)
 * @return 0 成功, -1 失败
 */
int CIM_LoadWeights(const void *weights, uint32_t size, uint8_t bank);

/**
 * @brief 等待 CIM 计算完成
 * @param timeout_us 超时时间 (μs), 0 表示无限等待
 * @return 0 成功, -1 超时
 */
int CIM_WaitDone(uint32_t timeout_us);

/**
 * @brief 检查 CIM 是否忙碌
 * @return true 忙碌, false 空闲
 */
bool CIM_IsBusy(void);

/**
 * @brief 使能 CIM 完成中断
 * @param enable true 使能, false 禁用
 */
void CIM_EnableIRQ(bool enable);

/**
 * @brief CIM 中断回调 (由用户实现)
 */
void CIM_IRQCallback(void);

/* ========================================================================= */
/* 高级 API (用于神经网络推理)                                               */
/* ========================================================================= */

/**
 * @brief 全连接层推理
 * @param input 输入向量 (input_size)
 * @param output 输出向量 (output_size)
 * @param weights 权重矩阵 (output_size x input_size)
 * @param bias 偏置 (output_size)
 * @param input_size 输入维度
 * @param output_size 输出维度
 * @param activation 激活函数类型 (0=None, 1=ReLU, 2=Sigmoid, 3=Tanh)
 * @return 0 成功, -1 失败
 */
int CIM_FullyConnected(const float *input, float *output, const void *weights,
                       const float *bias, uint32_t input_size,
                       uint32_t output_size, uint8_t activation);

/**
 * @brief ReLU 激活函数 (硬件加速)
 * @param data 数据数组
 * @param size 数据长度
 */
void CIM_ReLU(float *data, uint32_t size);

/**
 * @brief Sigmoid 激活函数 (查找表)
 * @param data 数据数组
 * @param size 数据长度
 */
void CIM_Sigmoid(float *data, uint32_t size);

/**
 * @brief Tanh 激活函数 (查找表)
 * @param data 数据数组
 * @param size 数据长度
 */
void CIM_Tanh(float *data, uint32_t size);

/* ========================================================================= */
/* 性能分析                                                                  */
/* ========================================================================= */

typedef struct {
  uint32_t total_ops;     /**< 总操作数 */
  uint32_t total_cycles;  /**< 总周期数 */
  uint32_t total_time_us; /**< 总时间 (μs) */
  float gops;             /**< GOPS (十亿次操作/秒) */
  float power_mw;         /**< 功耗 (mW) */
} CIM_PerfStats_t;

/**
 * @brief 获取性能统计
 * @param stats 统计结构体指针
 */
void CIM_GetPerfStats(CIM_PerfStats_t *stats);

/**
 * @brief 重置性能统计
 */
void CIM_ResetPerfStats(void);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_CIM_H */
