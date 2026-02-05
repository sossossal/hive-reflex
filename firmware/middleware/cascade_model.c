/**
 * @file cascade_model.c
 * @brief Strategy 3: 条件加载 - 级联模型实现
 */

#include "cascade_model.h"
#include <math.h>
#include <stdio.h>
#include <string.h>


// ============================================================================
// 初始化
// ============================================================================

bool Cascade_Init(CascadeModel_t *model, uint8_t num_exits,
                  uint8_t total_layers) {
  if (!model || num_exits > CASCADE_MAX_EXIT_POINTS || total_layers == 0) {
    return false;
  }

  memset(model, 0, sizeof(CascadeModel_t));
  model->num_exit_points = num_exits;
  model->total_layers = total_layers;
  model->enable_adaptive_threshold = false;
  model->threshold_adjust_factor = 1.0f;

  return true;
}

bool Cascade_ConfigureExit(CascadeModel_t *model, uint8_t exit_index,
                           uint8_t layer_index, float threshold,
                           uint32_t feature_dim, const float *weights,
                           const float *bias, uint32_t output_dim) {
  if (!model || exit_index >= model->num_exit_points) {
    return false;
  }

  if (layer_index >= model->total_layers) {
    return false;
  }

  ExitPoint_t *ep = &model->exit_points[exit_index];
  ep->layer_index = layer_index;
  ep->confidence_threshold = threshold;
  ep->feature_dim = feature_dim;
  ep->classifier_weights = weights;
  ep->classifier_bias = bias;
  ep->classifier_output_dim = output_dim;

  return true;
}

void Cascade_SetLayerSizes(CascadeModel_t *model, const uint32_t *layer_sizes) {
  if (model && layer_sizes) {
    model->layer_flash_sizes = layer_sizes;
  }
}

void Cascade_EnableAdaptiveThreshold(CascadeModel_t *model, bool enable) {
  if (model) {
    model->enable_adaptive_threshold = enable;
    // 可根据历史数据自动调整 threshold_adjust_factor（此处简化）
  }
}

// ============================================================================
// 推理逻辑
// ============================================================================

bool Cascade_ShouldExit(CascadeModel_t *model, uint8_t current_layer,
                        const float *features, uint32_t feature_dim,
                        uint8_t *predicted_class, float *confidence) {
  if (!model || !features || !predicted_class || !confidence) {
    return false;
  }

  // 查找是否存在该层的退出点
  ExitPoint_t *ep = NULL;
  for (uint8_t i = 0; i < model->num_exit_points; i++) {
    if (model->exit_points[i].layer_index == current_layer) {
      ep = &model->exit_points[i];
      break;
    }
  }

  if (!ep) {
    return false; // 该层无退出点
  }

  if (feature_dim != ep->feature_dim) {
    return false; // 特征维度不匹配
  }

  // 计算分类器输出（简化的线性分类 + Softmax）
  float logits[16]; // 假设最多 16 个类别
  if (ep->classifier_output_dim > 16) {
    return false;
  }

  // logits = weights @ features + bias
  for (uint32_t c = 0; c < ep->classifier_output_dim; c++) {
    float sum = ep->classifier_bias ? ep->classifier_bias[c] : 0.0f;
    for (uint32_t d = 0; d < feature_dim; d++) {
      sum += ep->classifier_weights[c * feature_dim + d] * features[d];
    }
    logits[c] = sum;
  }

  // 计算置信度
  Cascade_ComputeConfidence(logits, ep->classifier_output_dim, predicted_class,
                            confidence);

  // 应用自适应阈值
  float effective_threshold =
      ep->confidence_threshold * model->threshold_adjust_factor;

  return (*confidence >= effective_threshold);
}

uint32_t Cascade_CalculateSavedBytes(CascadeModel_t *model,
                                     uint8_t exit_layer) {
  if (!model || !model->layer_flash_sizes) {
    return 0;
  }

  uint32_t saved = 0;
  for (uint8_t i = exit_layer + 1; i < model->total_layers; i++) {
    saved += model->layer_flash_sizes[i];
  }

  return saved;
}

void Cascade_UpdateStats(CascadeModel_t *model, bool did_early_exit,
                         uint32_t saved_bytes) {
  if (!model)
    return;

  if (did_early_exit) {
    model->early_exit_count++;
    model->total_flash_saved_bytes += saved_bytes;
  } else {
    model->full_inference_count++;
  }
}

// ============================================================================
// 统计
// ============================================================================

float Cascade_GetEarlyExitRatio(const CascadeModel_t *model) {
  if (!model)
    return 0.0f;

  uint32_t total = model->early_exit_count + model->full_inference_count;
  if (total == 0)
    return 0.0f;

  return (float)model->early_exit_count / total;
}

uint32_t Cascade_GetAverageSavedBytes(const CascadeModel_t *model) {
  if (!model)
    return 0;

  uint32_t total = model->early_exit_count + model->full_inference_count;
  if (total == 0)
    return 0;

  return (uint32_t)(model->total_flash_saved_bytes / total);
}

void Cascade_PrintStats(const CascadeModel_t *model) {
  if (!model)
    return;

  printf("\n=== Cascade Model Statistics ===\n");
  printf("Early Exits:       %lu\n", (unsigned long)model->early_exit_count);
  printf("Full Inferences:   %lu\n",
         (unsigned long)model->full_inference_count);
  printf("Early Exit Ratio:  %.1f%%\n", Cascade_GetEarlyExitRatio(model) * 100);
  printf("Total Flash Saved: %lu KB\n",
         (unsigned long)(model->total_flash_saved_bytes / 1024));
  printf("Avg Saved/Frame:   %lu bytes\n",
         (unsigned long)Cascade_GetAverageSavedBytes(model));
  printf("================================\n");
}

void Cascade_ResetStats(CascadeModel_t *model) {
  if (!model)
    return;

  model->early_exit_count = 0;
  model->full_inference_count = 0;
  model->total_flash_saved_bytes = 0;
}

// ============================================================================
// 辅助函数
// ============================================================================

void Cascade_ComputeConfidence(const float *logits, uint32_t num_classes,
                               uint8_t *predicted_class, float *confidence) {
  if (!logits || !predicted_class || !confidence || num_classes == 0) {
    return;
  }

  // 找到最大 logit
  float max_logit = logits[0];
  uint8_t max_idx = 0;

  for (uint32_t i = 1; i < num_classes; i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      max_idx = i;
    }
  }

  // 计算 Softmax
  float sum_exp = 0.0f;
  for (uint32_t i = 0; i < num_classes; i++) {
    sum_exp += expf(logits[i] - max_logit); // 数值稳定性
  }

  *predicted_class = max_idx;
  *confidence =
      1.0f / sum_exp; // exp(max_logit - max_logit) / sum_exp = 1 / sum_exp
}
