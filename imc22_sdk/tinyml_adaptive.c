/**
 * @file tinyml_adaptive.c
 * @brief TinyML 自适应控制引擎实现
 *
 * 轻量级嵌入式 ML 推理引擎，用于实时计算 compliance 和权重混合
 */

#include "tinyml_adaptive.h"
#include "imc22.h"
#include <math.h>
#include <string.h>


/* ========================================================================= */
/* 默认模型权重（预训练的 MLP: 8->16->8->2）                                 */
/* ========================================================================= */

/* 层 1: 8 -> 16 (量化 int8) */
static const int8_t DEFAULT_W1[8 * 16] = {
    /* 简化的预训练权重 */
    12, -8, 5,  3,  -2, 7,  -4, 9,  6,  -3, 8,  -5, 2,  4,  -6, 11, -5, 10, -3,
    8,  4,  -7, 6,  2,  -9, 5,  3,  -4, 8,  -2, 7,  -6, 8,  3,  -7, 12, -4, 5,
    9,  -2, 6,  -8, 4,  7,  -3, 10, -5, 2,  -4, 7,  10, -6, 3,  8,  -5, 11, -2,
    4,  9,  -7, 5,  -3, 6,  8,  9,  -5, 4,  7,  -8, 3,  12, -4, 5,  6,  -9, 2,
    8,  -6, 3,  10, -7, 8,  6,  -3, 11, -5, 2,  9,  -4, 7,  5,  -8, 3,  6,  -2,
    4,  5,  -9, 8,  4,  -6, 10, -3, 7,  2,  -5, 9,  6,  -4, 8,  -7, 3,  -3, 6,
    -8, 5,  9,  -4, 7,  -2, 10, 3,  -6, 8,  4,  -5, 12, -7};
static const int32_t DEFAULT_B1[16] = {10, -5, 8,  3, -7, 12, 4,  -9,
                                       6,  -2, 11, 5, -4, 8,  -6, 3};

/* 层 2: 16 -> 8 */
static const int8_t DEFAULT_W2[16 * 8] = {
    7,  -4, 9,  3,  -6, 8,  5,  -2, 10, -7, 4,  6,  -3, 9,  -5,  8,  -6, 10, 4,
    -8, 5,  3,  -7, 9,  2,  -4, 7,  -5, 11, 6,  -3, 8,  5,  -3,  8,  6,  -9, 4,
    10, -5, 7,  3,  -6, 9,  -2, 5,  8,  -4, -8, 5,  3,  -7, 9,   6,  -4, 8,  -5,
    10, 2,  -6, 7,  -3, 4,  9,  9,  -6, 7,  4,  -5, 8,  3,  -10, 6,  -2, 9,  5,
    -7, 4,  8,  -3, 4,  8,  -5, 10, 3,  -7, 6,  9,  -4, 5,  -8,  7,  2,  -6, 10,
    4,  -5, 7,  9,  -3, 8,  4,  -6, 5,  10, -8, 3,  -4, 9,  6,   -2, 7,  8,  -4,
    6,  9,  -7, 5,  3,  -8, 4,  7,  -5, 10, -3, 8,  6,  -9};
static const int32_t DEFAULT_B2[8] = {5, -3, 8, 2, -6, 9, 4, -7};

/* 层 3: 8 -> 2 (输出: pid_weight, compliance) */
static const int8_t DEFAULT_W3[8 * 2] = {15, -12, 8, 5,   -9, 10, -6, 7,
                                         -8, 14,  6, -10, 7,  -5, 11, -9};
static const int32_t DEFAULT_B3[2] = {64, 64}; /* 输出中心化到 0.5 */

/* ========================================================================= */
/* 全局状态                                                                  */
/* ========================================================================= */

static TinyMLModel_t g_model = {
    .num_layers = 3,
    .input_scale = 0.1f,
    .output_scale = 1.0f / 127.0f,
    .layers = {{8, 16, ACTIVATION_RELU, DEFAULT_W1, DEFAULT_B1, 0.05f, 0},
               {16, 8, ACTIVATION_RELU, DEFAULT_W2, DEFAULT_B2, 0.05f, 0},
               {8, 2, ACTIVATION_SIGMOID, DEFAULT_W3, DEFAULT_B3, 0.05f, 0}}};

static AdaptiveConfig_t g_config = {.update_rate_hz = 1000.0f,
                                    .blend_smoothing = 0.9f,
                                    .high_load_threshold = 0.8f,
                                    .low_compliance_limit = 0.1f,
                                    .enable_safety_override = true,
                                    .max_neural_weight = 0.9f,
                                    .min_pid_weight = 0.1f};

static AdaptiveState_t g_state = {.pid_weight = 0.5f,
                                  .neural_weight = 0.5f,
                                  .compliance = 0.5f,
                                  .confidence = 1.0f,
                                  .high_load_mode = false};

static AdaptiveStats_t g_stats = {0};

static float g_history[TINYML_HISTORY_SIZE][TINYML_FEATURE_DIM] = {0};
static uint8_t g_history_idx = 0;

static bool g_initialized = false;
static bool g_adaptive_enabled = true;
static float g_forced_pid_weight = -1.0f; /* -1 = 自动 */

/* 工作缓冲区 */
static float g_layer_buffer[2][TINYML_MAX_NEURONS];

/* ========================================================================= */
/* 内部函数                                                                  */
/* ========================================================================= */

/* ReLU 激活 */
static inline float relu(float x) { return x > 0 ? x : 0; }

/* Sigmoid 激活 */
static inline float sigmoid(float x) {
  if (x < -10.0f)
    return 0.0f;
  if (x > 10.0f)
    return 1.0f;
  return 1.0f / (1.0f + expf(-x));
}

/* Tanh 激活 */
static inline float tanh_act(float x) { return tanhf(x); }

/* 应用激活函数 */
static void apply_activation(float *data, size_t size, ActivationType_t type) {
  size_t i;
  float sum;

  switch (type) {
  case ACTIVATION_RELU:
    for (i = 0; i < size; i++) {
      data[i] = relu(data[i]);
    }
    break;
  case ACTIVATION_SIGMOID:
    for (i = 0; i < size; i++) {
      data[i] = sigmoid(data[i]);
    }
    break;
  case ACTIVATION_TANH:
    for (i = 0; i < size; i++) {
      data[i] = tanh_act(data[i]);
    }
    break;
  case ACTIVATION_SOFTMAX:
    sum = 0;
    for (i = 0; i < size; i++) {
      data[i] = expf(data[i]);
      sum += data[i];
    }
    for (i = 0; i < size; i++) {
      data[i] /= sum;
    }
    break;
  default:
    break;
  }
}

/* 量化矩阵乘法 */
static void quantized_matmul(const float *input, size_t input_size,
                             const TinyMLLayer_t *layer, float *output) {
  size_t i, j;
  int32_t acc;

  for (i = 0; i < layer->output_size; i++) {
    acc = layer->bias[i];

    for (j = 0; j < input_size; j++) {
      /* 输入量化 */
      int8_t q_input = (int8_t)(input[j] / layer->scale);
      acc += (int32_t)layer->weights[j * layer->output_size + i] * q_input;
    }

    /* 反量化 */
    output[i] = (float)acc * layer->scale * layer->scale;
  }
}

/* 提取特征 */
static void extract_features(const SensorFeedback_t *sensors, float *features) {
  /* 归一化传感器数据 */
  features[0] = sensors->torque / 10.0f;         /* 力矩归一化 */
  features[1] = sensors->velocity / 5.0f;        /* 速度归一化 */
  features[2] = sensors->position_error / 0.5f;  /* 位置误差 */
  features[3] = sensors->velocity_error / 1.0f;  /* 速度误差 */
  features[4] = sensors->external_force / 20.0f; /* 外力 */
  features[5] = fabsf(sensors->position_error) +
                fabsf(sensors->velocity_error); /* 总误差 */

  /* 历史特征 */
  features[6] = g_history[(g_history_idx + TINYML_HISTORY_SIZE - 1) %
                          TINYML_HISTORY_SIZE][0];
  features[7] = g_history[(g_history_idx + TINYML_HISTORY_SIZE - 2) %
                          TINYML_HISTORY_SIZE][0];

  /* 更新历史 */
  memcpy(g_history[g_history_idx], features,
         TINYML_FEATURE_DIM * sizeof(float));
  g_history_idx = (g_history_idx + 1) % TINYML_HISTORY_SIZE;
}

/* ========================================================================= */
/* 公共 API 实现                                                             */
/* ========================================================================= */

int TinyML_Init(const uint8_t *model_data, size_t model_size) {
  if (model_data != NULL && model_size > 0) {
    /* 解析自定义模型 */
    return TinyML_LoadModel(model_data, model_size);
  }

  /* 使用默认模型 */
  return TinyML_InitDefault();
}

int TinyML_InitDefault(void) {
  /* 重置状态 */
  g_state.pid_weight = 0.5f;
  g_state.neural_weight = 0.5f;
  g_state.compliance = 0.5f;
  g_state.high_load_mode = false;

  memset(&g_stats, 0, sizeof(g_stats));
  memset(g_history, 0, sizeof(g_history));
  g_history_idx = 0;

  g_initialized = true;
  g_adaptive_enabled = true;
  g_forced_pid_weight = -1.0f;

  return 0;
}

int TinyML_Configure(const AdaptiveConfig_t *config) {
  if (config != NULL) {
    g_config = *config;
  }
  return 0;
}

AdaptiveState_t TinyML_ComputeAdaptive(const SensorFeedback_t *sensors) {
  if (!g_initialized || sensors == NULL) {
    return g_state;
  }

  uint32_t start_time = GetCycleCount();

  /* 提取特征 */
  float features[TINYML_FEATURE_DIM];
  extract_features(sensors, features);

  /* 推理 */
  float output[2];
  TinyML_Inference(features, output);

  /* 解析输出 */
  float raw_pid_weight = output[0];
  float raw_compliance = output[1];

  /* 平滑更新 */
  g_state.compliance = g_config.blend_smoothing * g_state.compliance +
                       (1 - g_config.blend_smoothing) * raw_compliance;

  /* 高负载检测 */
  float load_level = fabsf(sensors->torque) / 10.0f;
  g_state.high_load_mode = (load_level > g_config.high_load_threshold);

  if (g_state.high_load_mode) {
    g_stats.high_load_events++;
  }

  /* 计算权重 */
  if (g_forced_pid_weight >= 0) {
    /* 手动模式 */
    g_state.pid_weight = g_forced_pid_weight;
    g_state.neural_weight = 1.0f - g_forced_pid_weight;
  } else if (g_adaptive_enabled) {
    /* 自适应模式 */
    if (g_state.high_load_mode) {
      /* 高负载偏向 PID（更稳定） */
      g_state.pid_weight = 0.8f + 0.2f * raw_pid_weight;
    } else {
      /* 正常模式 */
      g_state.pid_weight = raw_pid_weight;
    }

    /* 应用限制 */
    if (g_state.pid_weight < g_config.min_pid_weight) {
      g_state.pid_weight = g_config.min_pid_weight;
    }

    g_state.neural_weight = 1.0f - g_state.pid_weight;

    if (g_state.neural_weight > g_config.max_neural_weight) {
      g_state.neural_weight = g_config.max_neural_weight;
      g_state.pid_weight = 1.0f - g_state.neural_weight;
    }
  }

  /* 合规度安全限制 */
  if (g_config.enable_safety_override &&
      g_state.compliance < g_config.low_compliance_limit) {
    g_state.compliance = g_config.low_compliance_limit;
  }

  /* 更新统计 */
  g_stats.inference_count++;
  g_stats.inference_time_us =
      (GetCycleCount() - start_time) / (IMC22_SYSCLK_HZ / 1000000);
  g_stats.avg_compliance =
      (g_stats.avg_compliance * (g_stats.inference_count - 1) +
       g_state.compliance) /
      g_stats.inference_count;
  g_stats.avg_pid_weight =
      (g_stats.avg_pid_weight * (g_stats.inference_count - 1) +
       g_state.pid_weight) /
      g_stats.inference_count;

  return g_state;
}

void TinyML_UpdateBlend(const AdaptiveState_t *state) {
  if (state != NULL) {
    /* 通知其他模块更新权重 */
    /* 这里可以写入硬件寄存器或调用其他 API */
  }
}

float TinyML_GetPIDWeight(void) { return g_state.pid_weight; }

float TinyML_GetNeuralWeight(void) { return g_state.neural_weight; }

float TinyML_GetCompliance(void) { return g_state.compliance; }

void TinyML_ForcePIDWeight(float weight) {
  if (weight < 0) {
    g_forced_pid_weight = -1.0f; /* 恢复自动 */
  } else if (weight > 1.0f) {
    g_forced_pid_weight = 1.0f;
  } else {
    g_forced_pid_weight = weight;
  }
}

void TinyML_EnableAdaptive(bool enable) { g_adaptive_enabled = enable; }

bool TinyML_IsHighLoadMode(void) { return g_state.high_load_mode; }

void TinyML_Reset(void) { TinyML_InitDefault(); }

void TinyML_GetStats(AdaptiveStats_t *stats) {
  if (stats != NULL) {
    *stats = g_stats;
  }
}

uint32_t TinyML_Inference(const float *input, float *output) {
  uint32_t start = GetCycleCount();

  /* 复制输入到缓冲区 */
  memcpy(g_layer_buffer[0], input,
         g_model.layers[0].input_size * sizeof(float));

  /* 前向传播 */
  int src = 0;
  for (uint8_t l = 0; l < g_model.num_layers; l++) {
    int dst = 1 - src;

    quantized_matmul(g_layer_buffer[src], g_model.layers[l].input_size,
                     &g_model.layers[l], g_layer_buffer[dst]);

    apply_activation(g_layer_buffer[dst], g_model.layers[l].output_size,
                     g_model.layers[l].activation);

    src = dst;
  }

  /* 复制输出 */
  memcpy(output, g_layer_buffer[src],
         g_model.layers[g_model.num_layers - 1].output_size * sizeof(float));

  return (GetCycleCount() - start) / (IMC22_SYSCLK_HZ / 1000000);
}

int TinyML_LoadModel(const uint8_t *model_data, size_t model_size) {
  /* 简化的模型格式解析 */
  /* 实际实现需要支持完整的 TFLite/ONNX 格式 */

  if (model_data == NULL || model_size < 16) {
    return -1;
  }

  /* 头部验证 */
  if (model_data[0] != 'T' || model_data[1] != 'M' || model_data[2] != 'L' ||
      model_data[3] != '1') {
    return -2; /* 无效魔数 */
  }

  /* TODO: 实现完整的模型解析 */

  g_initialized = true;
  return 0;
}

void TinyML_GetModelInfo(uint8_t *num_layers, size_t *model_size) {
  if (num_layers)
    *num_layers = g_model.num_layers;
  if (model_size) {
    /* 估算模型大小 */
    size_t size = 0;
    for (uint8_t i = 0; i < g_model.num_layers; i++) {
      size += g_model.layers[i].input_size * g_model.layers[i].output_size;
      size += g_model.layers[i].output_size * sizeof(int32_t);
    }
    *model_size = size;
  }
}
