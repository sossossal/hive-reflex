/**
 * @file nn_topology.h
 * @brief 模块化神经网络拓扑接口
 *
 * 允许用户自定义神经网络结构，而非使用固定拓扑
 *
 * 支持：
 * - 动态层定义（FC、Conv、LSTM 等）
 * - 自定义激活函数
 * - 模块化权重加载
 * - 运行时拓扑修改
 *
 * @version 2.1.0
 */

#ifndef NN_TOPOLOGY_H
#define NN_TOPOLOGY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 常量定义                                                                  */
/* ========================================================================= */

#define NN_MAX_LAYERS 32   ///< 最大层数
#define NN_MAX_NAME_LEN 32 ///< 层名称最大长度
#define NN_MAX_HANDLES 8   ///< 最大模型句柄数

/* ========================================================================= */
/* 枚举定义                                                                  */
/* ========================================================================= */

/**
 * @brief 层类型
 */
typedef enum {
  LAYER_NONE = 0,
  LAYER_INPUT,          ///< 输入层
  LAYER_DENSE,          ///< 全连接层
  LAYER_CONV1D,         ///< 1D 卷积
  LAYER_CONV2D,         ///< 2D 卷积
  LAYER_DEPTHWISE_CONV, ///< 深度可分离卷积
  LAYER_LSTM,           ///< LSTM
  LAYER_GRU,            ///< GRU
  LAYER_POOLING,        ///< 池化
  LAYER_BATCHNORM,      ///< BatchNorm
  LAYER_DROPOUT,        ///< Dropout
  LAYER_FLATTEN,        ///< 展平
  LAYER_RESHAPE,        ///< 重塑
  LAYER_CONCAT,         ///< 拼接
  LAYER_ADD,            ///< 逐元素加
  LAYER_SOFTMAX,        ///< Softmax
  LAYER_CUSTOM          ///< 自定义层
} LayerType_t;

/**
 * @brief 激活函数类型
 */
typedef enum {
  ACTIVATION_NONE = 0,
  ACTIVATION_RELU,
  ACTIVATION_RELU6,
  ACTIVATION_LEAKY_RELU,
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH,
  ACTIVATION_SOFTMAX,
  ACTIVATION_SWISH,
  ACTIVATION_GELU
} ActivationType_t;

/**
 * @brief 权重格式
 */
typedef enum {
  WEIGHT_FORMAT_FLOAT32 = 0,
  WEIGHT_FORMAT_FLOAT16,
  WEIGHT_FORMAT_INT8,
  WEIGHT_FORMAT_INT4,
  WEIGHT_FORMAT_BINARY
} WeightFormat_t;

/**
 * @brief 池化类型
 */
typedef enum {
  POOL_MAX = 0,
  POOL_AVG,
  POOL_GLOBAL_MAX,
  POOL_GLOBAL_AVG
} PoolType_t;

/* ========================================================================= */
/* 结构体定义                                                                */
/* ========================================================================= */

/**
 * @brief 张量形状
 */
typedef struct {
  uint16_t dims[4]; ///< 维度 [batch, height/seq, width/features, channels]
  uint8_t num_dims; ///< 维度数量
} TensorShape_t;

/**
 * @brief 卷积参数
 */
typedef struct {
  uint16_t kernel_size[2]; ///< 卷积核大小 [H, W]
  uint16_t stride[2];      ///< 步长
  uint16_t padding[2];     ///< 填充
  uint16_t dilation[2];    ///< 膨胀
  uint16_t groups;         ///< 分组数
} ConvParams_t;

/**
 * @brief 池化参数
 */
typedef struct {
  PoolType_t type;         ///< 池化类型
  uint16_t kernel_size[2]; ///< 池化核大小
  uint16_t stride[2];      ///< 步长
} PoolParams_t;

/**
 * @brief LSTM/GRU 参数
 */
typedef struct {
  uint16_t hidden_size; ///< 隐藏层大小
  uint8_t num_layers;   ///< 层数
  bool bidirectional;   ///< 是否双向
  bool batch_first;     ///< 批次维度是否在前
} RecurrentParams_t;

/**
 * @brief 层配置
 */
typedef struct {
  char name[NN_MAX_NAME_LEN];  ///< 层名称
  LayerType_t type;            ///< 层类型
  TensorShape_t input_shape;   ///< 输入形状
  TensorShape_t output_shape;  ///< 输出形状
  ActivationType_t activation; ///< 激活函数

  /* 层特定参数 (union 节省空间) */
  union {
    struct {
      uint16_t units; ///< Dense: 神经元数
      bool use_bias;
    } dense;

    ConvParams_t conv;     ///< Conv 参数
    PoolParams_t pool;     ///< Pool 参数
    RecurrentParams_t rnn; ///< RNN 参数

    struct {
      int target_shape[4]; ///< Reshape 目标形状
    } reshape;
  } params;

  /* 权重信息 */
  WeightFormat_t weight_format;
  const void *weights;     ///< 权重指针 (NULL = 在线加载)
  const void *bias;        ///< 偏置指针
  size_t weight_size;      ///< 权重大小 (字节)
  float quant_scale;       ///< 量化缩放因子
  int8_t quant_zero_point; ///< 量化零点

} LayerConfig_t;

/**
 * @brief 模型句柄
 */
typedef struct {
  uint32_t id;           ///< 句柄 ID
  uint8_t num_layers;    ///< 层数
  LayerConfig_t *layers; ///< 层配置数组
  bool initialized;      ///< 是否已初始化
  void *runtime_ctx;     ///< 运行时上下文
} NNHandle_t;

/**
 * @brief 推理配置
 */
typedef struct {
  bool use_sparse;          ///< 使用稀疏计算
  uint8_t sparse_threshold; ///< 稀疏阈值
  bool enable_profiling;    ///< 启用性能分析
  uint16_t batch_size;      ///< 批次大小
} InferConfig_t;

/**
 * @brief 推理统计
 */
typedef struct {
  uint32_t total_time_us;               ///< 总推理时间
  uint32_t per_layer_us[NN_MAX_LAYERS]; ///< 每层时间
  uint32_t ops_executed;                ///< 执行的操作数
  uint32_t ops_skipped;                 ///< 跳过的操作数（稀疏）
  float sparsity_ratio;                 ///< 稀疏率
} InferStats_t;

/* ========================================================================= */
/* 回调类型（用于自定义层）                                                  */
/* ========================================================================= */

/**
 * @brief 自定义层前向函数类型
 */
typedef int (*CustomLayerForward_t)(const void *input, void *output,
                                    const LayerConfig_t *config,
                                    void *user_data);

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 创建网络拓扑
 * @param layers 层配置数组
 * @param num_layers 层数
 * @param handle 输出句柄
 * @return 0 成功，负数失败
 */
int NN_CreateTopology(const LayerConfig_t *layers, size_t num_layers,
                      NNHandle_t *handle);

/**
 * @brief 从 JSON 创建拓扑
 * @param json_str JSON 字符串
 * @param handle 输出句柄
 * @return 0 成功
 */
int NN_CreateFromJSON(const char *json_str, NNHandle_t *handle);

/**
 * @brief 销毁网络拓扑
 * @param handle 句柄
 */
void NN_DestroyTopology(NNHandle_t *handle);

/**
 * @brief 添加层
 * @param handle 句柄
 * @param config 层配置
 * @return 层索引，负数失败
 */
int NN_AddLayer(NNHandle_t *handle, const LayerConfig_t *config);

/**
 * @brief 移除层
 * @param handle 句柄
 * @param layer_idx 层索引
 * @return 0 成功
 */
int NN_RemoveLayer(NNHandle_t *handle, size_t layer_idx);

/**
 * @brief 获取层配置
 * @param handle 句柄
 * @param layer_idx 层索引
 * @param config 输出配置
 * @return 0 成功
 */
int NN_GetLayerConfig(NNHandle_t *handle, size_t layer_idx,
                      LayerConfig_t *config);

/**
 * @brief 设置层权重
 * @param handle 句柄
 * @param layer_idx 层索引
 * @param weights 权重数据
 * @param size 权重大小
 * @param format 权重格式
 * @return 0 成功
 */
int NN_SetWeights(NNHandle_t *handle, size_t layer_idx, const void *weights,
                  size_t size, WeightFormat_t format);

/**
 * @brief 从文件加载权重
 * @param handle 句柄
 * @param filepath 权重文件路径
 * @return 0 成功
 */
int NN_LoadWeightsFromFile(NNHandle_t *handle, const char *filepath);

/**
 * @brief 初始化网络（分配运行时资源）
 * @param handle 句柄
 * @return 0 成功
 */
int NN_Initialize(NNHandle_t *handle);

/**
 * @brief 执行推理
 * @param handle 句柄
 * @param input 输入数据
 * @param output 输出数据
 * @param config 推理配置（NULL 使用默认）
 * @return 0 成功
 */
int NN_Infer(NNHandle_t *handle, const void *input, void *output,
             const InferConfig_t *config);

/**
 * @brief 获取推理统计
 * @param handle 句柄
 * @param stats 输出统计
 */
void NN_GetStats(NNHandle_t *handle, InferStats_t *stats);

/**
 * @brief 注册自定义层
 * @param type_name 类型名称
 * @param forward_fn 前向函数
 * @param user_data 用户数据
 * @return 0 成功
 */
int NN_RegisterCustomLayer(const char *type_name,
                           CustomLayerForward_t forward_fn, void *user_data);

/**
 * @brief 打印网络结构
 * @param handle 句柄
 */
void NN_PrintTopology(NNHandle_t *handle);

/**
 * @brief 获取网络信息
 * @param handle 句柄
 * @param num_layers 输出层数
 * @param total_params 输出总参数数
 * @param total_memory 输出内存占用
 */
void NN_GetInfo(NNHandle_t *handle, size_t *num_layers, size_t *total_params,
                size_t *total_memory);

/* ========================================================================= */
/* 便捷层创建宏                                                              */
/* ========================================================================= */

/**
 * @brief 创建 Dense 层配置
 */
#define NN_DENSE_LAYER(name, in_features, out_features, act)                   \
  (LayerConfig_t) {                                                            \
    .name = name, .type = LAYER_DENSE,                                         \
    .input_shape = {.dims = {1, in_features, 0, 0}, .num_dims = 2},            \
    .output_shape = {.dims = {1, out_features, 0, 0}, .num_dims = 2},          \
    .activation = act,                                                         \
    .params.dense = {.units = out_features, .use_bias = true},                 \
    .weight_format = WEIGHT_FORMAT_INT8                                        \
  }

/**
 * @brief 创建 Conv2D 层配置
 */
#define NN_CONV2D_LAYER(name, in_ch, out_ch, k_h, k_w, act)                    \
  (LayerConfig_t) {                                                            \
    .name = name, .type = LAYER_CONV2D, .activation = act,                     \
    .params.conv = {.kernel_size = {k_h, k_w},                                 \
                    .stride = {1, 1},                                          \
                    .padding = {0, 0},                                         \
                    .groups = 1},                                              \
    .weight_format = WEIGHT_FORMAT_INT8                                        \
  }

/**
 * @brief 创建 LSTM 层配置
 */
#define NN_LSTM_LAYER(name, input_size, hidden_size, n_layers)                 \
  (LayerConfig_t) {                                                            \
    .name = name, .type = LAYER_LSTM,                                          \
    .params.rnn = {.hidden_size = hidden_size,                                 \
                   .num_layers = n_layers,                                     \
                   .bidirectional = false,                                     \
                   .batch_first = true},                                       \
    .weight_format = WEIGHT_FORMAT_INT8                                        \
  }

#ifdef __cplusplus
}
#endif

#endif /* NN_TOPOLOGY_H */
