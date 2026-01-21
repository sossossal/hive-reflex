/**
 * @file tinyml_adaptive.h
 * @brief TinyML 自适应控制引擎
 *
 * 使用嵌入式 ML 模型实时计算 compliance，动态调整 PID 与神经反射权重
 *
 * 特性：
 * - 轻量级 MLP 推理 (< 10KB Flash)
 * - 实时更新周期：1ms
 * - 输入特征：力矩、速度、位置误差、历史偏差
 * - 输出：PID/Neural blend ratio
 *
 * @version 2.1.0
 */

#ifndef TINYML_ADAPTIVE_H
#define TINYML_ADAPTIVE_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 常量定义                                                                  */
/* ========================================================================= */

#define TINYML_MAX_LAYERS 8   ///< 最大层数
#define TINYML_MAX_NEURONS 64 ///< 每层最大神经元数
#define TINYML_FEATURE_DIM 8  ///< 输入特征维度
#define TINYML_HISTORY_SIZE 4 ///< 历史窗口大小

/* ========================================================================= */
/* 类型定义                                                                  */
/* ========================================================================= */

/**
 * @brief 传感器反馈数据
 */
typedef struct {
  float torque;         ///< 当前力矩 (Nm)
  float velocity;       ///< 当前速度 (rad/s)
  float position;       ///< 当前位置 (rad)
  float position_error; ///< 位置误差
  float velocity_error; ///< 速度误差
  float external_force; ///< 外部力 (N)
  float temperature;    ///< 温度 (°C)
  float voltage;        ///< 电压 (V)
} SensorFeedback_t;

/**
 * @brief 自适应状态输出
 */
typedef struct {
  float pid_weight;    ///< PID 权重 (0.0-1.0)
  float neural_weight; ///< 神经反射权重 (0.0-1.0)
  float compliance;    ///< 当前合规度 (0.0-1.0)
  float confidence;    ///< 预测置信度 (0.0-1.0)
  bool high_load_mode; ///< 是否高负载模式
} AdaptiveState_t;

/**
 * @brief 激活函数类型
 */
typedef enum {
  ACTIVATION_NONE = 0,
  ACTIVATION_RELU,
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH,
  ACTIVATION_SOFTMAX
} ActivationType_t;

/**
 * @brief 层配置
 */
typedef struct {
  uint16_t input_size;         ///< 输入大小
  uint16_t output_size;        ///< 输出大小
  ActivationType_t activation; ///< 激活函数
  const int8_t *weights;       ///< 权重指针 (量化 int8)
  const int32_t *bias;         ///< 偏置指针 (int32)
  float scale;                 ///< 量化缩放因子
  int8_t zero_point;           ///< 量化零点
} TinyMLLayer_t;

/**
 * @brief TinyML 模型配置
 */
typedef struct {
  uint8_t num_layers; ///< 层数
  TinyMLLayer_t layers[TINYML_MAX_LAYERS];
  float input_scale;  ///< 输入缩放
  float output_scale; ///< 输出缩放
} TinyMLModel_t;

/**
 * @brief 自适应控制配置
 */
typedef struct {
  float update_rate_hz;        ///< 更新频率 (Hz)
  float blend_smoothing;       ///< 混合平滑系数 (0.0-1.0)
  float high_load_threshold;   ///< 高负载阈值
  float low_compliance_limit;  ///< 最低合规度限制
  bool enable_safety_override; ///< 启用安全覆盖
  float max_neural_weight;     ///< 最大神经权重
  float min_pid_weight;        ///< 最小 PID 权重
} AdaptiveConfig_t;

/**
 * @brief 运行时统计
 */
typedef struct {
  uint32_t inference_count;   ///< 推理次数
  uint32_t inference_time_us; ///< 最后推理时间 (μs)
  float avg_compliance;       ///< 平均合规度
  float avg_pid_weight;       ///< 平均 PID 权重
  uint32_t high_load_events;  ///< 高负载事件次数
} AdaptiveStats_t;

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化 TinyML 推理引擎
 * @param model_data 模型二进制数据指针
 * @param model_size 模型大小 (字节)
 * @return 0 成功，负数失败
 */
int TinyML_Init(const uint8_t *model_data, size_t model_size);

/**
 * @brief 使用预定义模型初始化
 * @return 0 成功
 */
int TinyML_InitDefault(void);

/**
 * @brief 配置自适应控制器
 * @param config 配置参数，NULL 使用默认
 * @return 0 成功
 */
int TinyML_Configure(const AdaptiveConfig_t *config);

/**
 * @brief 计算自适应状态（主函数）
 * @param sensors 传感器反馈
 * @return 自适应状态
 */
AdaptiveState_t TinyML_ComputeAdaptive(const SensorFeedback_t *sensors);

/**
 * @brief 更新混合权重
 * @param state 自适应状态
 */
void TinyML_UpdateBlend(const AdaptiveState_t *state);

/**
 * @brief 获取当前 PID 权重
 * @return PID 权重 (0.0-1.0)
 */
float TinyML_GetPIDWeight(void);

/**
 * @brief 获取当前神经反射权重
 * @return 神经权重 (0.0-1.0)
 */
float TinyML_GetNeuralWeight(void);

/**
 * @brief 获取当前合规度
 * @return 合规度 (0.0-1.0)
 */
float TinyML_GetCompliance(void);

/**
 * @brief 强制设置 PID 权重（手动模式）
 * @param weight 权重 (0.0-1.0)
 */
void TinyML_ForcePIDWeight(float weight);

/**
 * @brief 启用/禁用自适应模式
 * @param enable 是否启用
 */
void TinyML_EnableAdaptive(bool enable);

/**
 * @brief 检查是否处于高负载模式
 * @return true 高负载
 */
bool TinyML_IsHighLoadMode(void);

/**
 * @brief 重置自适应状态
 */
void TinyML_Reset(void);

/**
 * @brief 获取统计信息
 * @param stats 输出统计
 */
void TinyML_GetStats(AdaptiveStats_t *stats);

/**
 * @brief 执行单次推理（低层 API）
 * @param input 输入特征数组
 * @param output 输出数组
 * @return 推理时间 (μs)
 */
uint32_t TinyML_Inference(const float *input, float *output);

/**
 * @brief 加载新模型
 * @param model_data 模型数据
 * @param model_size 模型大小
 * @return 0 成功
 */
int TinyML_LoadModel(const uint8_t *model_data, size_t model_size);

/**
 * @brief 获取模型信息
 * @param num_layers 输出层数
 * @param model_size 输出模型大小
 */
void TinyML_GetModelInfo(uint8_t *num_layers, size_t *model_size);

#ifdef __cplusplus
}
#endif

#endif /* TINYML_ADAPTIVE_H */
