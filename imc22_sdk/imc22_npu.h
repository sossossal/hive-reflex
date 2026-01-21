/**
 * @file imc22_npu.h
 * @brief IMC-22 神经加速器 (NPU) 驱动接口
 *
 * NPU 特性:
 * - 存内计算架构
 * - 支持 INT8/FP16 推理
 * - 专用 LSTM 加速单元
 * - 权重存储: 128KB SRAM
 */

#ifndef IMC22_NPU_H
#define IMC22_NPU_H

#include "imc22.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== NPU 寄存器结构 ========== */
typedef struct {
  vuint32_t CTRL;        // 控制寄存器
  vuint32_t STATUS;      // 状态寄存器
  vuint32_t CMD;         // 命令寄存器
  vuint32_t INPUT_ADDR;  // 输入数据地址
  vuint32_t OUTPUT_ADDR; // 输出数据地址
  vuint32_t WEIGHT_ADDR; // 权重地址
  vuint32_t LAYER_CFG;   // 层配置
  vuint32_t PERF_CNT;    // 性能计数器 (推理周期数)
} NPU_TypeDef;

#define NPU ((NPU_TypeDef *)IMC22_NPU_BASE)

/* NPU 控制位 */
#define NPU_CTRL_EN (1 << 0)    // NPU 使能
#define NPU_CTRL_START (1 << 1) // 启动推理
#define NPU_CTRL_IE (1 << 2)    // 中断使能
#define NPU_CTRL_INT8 (1 << 3)  // INT8 模式
#define NPU_CTRL_FP16 (1 << 4)  // FP16 模式

/* NPU 状态位 */
#define NPU_STATUS_BUSY (1 << 0)  // 忙碌
#define NPU_STATUS_DONE (1 << 1)  // 推理完成
#define NPU_STATUS_ERROR (1 << 2) // 错误

/* ========== NPU 数据类型 ========== */
typedef enum { NPU_DTYPE_INT8, NPU_DTYPE_FP16, NPU_DTYPE_FP32 } NPU_DataType_t;

/* ========== NPU 模型句柄 ========== */
typedef struct {
  uint32_t weight_addr;    // 权重在 NPU SRAM 中的地址
  uint32_t weight_size;    // 权重大小 (字节)
  uint16_t input_dims[4];  // 输入维度 [batch, seq, channel, feature]
  uint16_t output_dims[4]; // 输出维度
  NPU_DataType_t dtype;    // 数据类型
  bool has_lstm;           // 是否包含 LSTM 层
} NPU_Model_t;

/* ========== NPU 推理上下文 ========== */
typedef struct {
  const NPU_Model_t *model;
  float *lstm_h;      // LSTM 隐藏状态 h (需要用户提供缓冲)
  float *lstm_c;      // LSTM 单元状态 c
  uint16_t lstm_size; // LSTM 隐藏层大小
} NPU_Context_t;

/* ========== 函数声明 ========== */

/**
 * @brief 初始化 NPU
 * @return 0=成功, -1=失败
 */
int NPU_Init(void);

/**
 * @brief 从 Flash 加载模型权重到 NPU SRAM
 * @param model 模型句柄
 * @param weight_data 权重数据指针 (通常在 Flash 中)
 * @return 0=成功, -1=失败
 */
int NPU_LoadModel(NPU_Model_t *model, const void *weight_data);

/**
 * @brief 执行推理 (阻塞)
 * @param ctx 推理上下文
 * @param input 输入数据指针
 * @param output 输出数据指针
 * @return 0=成功, -1=错误
 */
int NPU_Inference(NPU_Context_t *ctx, const void *input, void *output);

/**
 * @brief 启动推理 (非阻塞)
 */
void NPU_StartInference(NPU_Context_t *ctx, const void *input);

/**
 * @brief 检查 NPU 是否忙碌
 */
static inline bool NPU_IsBusy(void) {
  return (NPU->STATUS & NPU_STATUS_BUSY) != 0;
}

/**
 * @brief 等待推理完成
 * @param timeout_us 超时时间 (微秒), 0=无限等待
 * @return 0=成功, -1=超时
 */
int NPU_WaitDone(uint32_t timeout_us);

/**
 * @brief 获取最后一次推理的周期数
 */
static inline uint32_t NPU_GetPerfCount(void) { return NPU->PERF_CNT; }

/**
 * @brief NPU 中断回调 (需要用户实现)
 */
extern void NPU_DoneCallback(void);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_NPU_H */
