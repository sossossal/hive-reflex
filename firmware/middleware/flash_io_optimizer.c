/**
 * @file flash_io_optimizer.c
 * @brief Flash IO 优化集成框架实现
 */

#include "flash_io_optimizer.h"
#include <stdio.h>
#include <string.h>


// 模拟的 Flash 读取函数（需替换为实际硬件接口）
extern void Flash_Read(uint32_t addr, void *buffer, uint32_t size);

// 模拟的 CIM 推理函数（需替换为实际 CIM HAL）
extern void CIM_Compute(const float *input, float *output, const void *weights,
                        uint32_t layer_idx);

// 模拟的时间戳函数（需替换为实际定时器）
static inline uint64_t GetTimestampUs(void) {
  // TODO: 实现微秒级时间戳
  return 0;
}

// ============================================================================
// 初始化
// ============================================================================

bool FlashOpt_Init(FlashIOOptimizer_t *opt, OptimizationFlags_t flags) {
  if (!opt)
    return false;

  memset(opt, 0, sizeof(FlashIOOptimizer_t));
  opt->flags = flags;

  return true;
}

bool FlashOpt_ConfigPipeline(FlashIOOptimizer_t *opt,
                             PipelineController_t *pipeline_ctrl) {
  if (!opt || !(opt->flags & OPT_PIPELINE)) {
    return false;
  }

  opt->pipeline_ctrl = pipeline_ctrl;
  return true;
}

bool FlashOpt_ConfigCompression(FlashIOOptimizer_t *opt, CompressionType_t type,
                                uint32_t buffer_size) {
  if (!opt || !(opt->flags & OPT_COMPRESS)) {
    return false;
  }

  opt->compression_type = type;
  opt->decompress_buffer_size = buffer_size;
  return true;
}

bool FlashOpt_ConfigCascade(FlashIOOptimizer_t *opt,
                            CascadeModel_t *cascade_model) {
  if (!opt || !(opt->flags & OPT_CASCADE)) {
    return false;
  }

  opt->cascade_model = cascade_model;
  return true;
}

// ============================================================================
// 推理逻辑
// ============================================================================

bool FlashOpt_InferLayer(FlashIOOptimizer_t *opt, uint8_t layer_index,
                         uint32_t flash_addr, uint32_t flash_size,
                         const float *input, float *output, bool *should_exit) {
  if (!opt || !input || !output) {
    return false;
  }

  uint64_t start_time = GetTimestampUs();

  // ========================================================================
  // Strategy 1: Pipeline - 异步加载下一层
  // ========================================================================
  if (opt->flags & OPT_PIPELINE && opt->pipeline_ctrl) {
    // 同步等待当前层加载完成
    Pipeline_Sync(opt->pipeline_ctrl, layer_index);

    // 获取当前层权重
    const void *weights =
        Pipeline_GetLayerWeights(opt->pipeline_ctrl, layer_index);

    // 启动下一层的异步加载
    if (layer_index + 1 < opt->pipeline_ctrl->model.num_layers) {
      Pipeline_LoadLayerAsync(opt->pipeline_ctrl, layer_index + 1);
    }

    // CIM 计算
    CIM_Compute(input, output, weights, layer_index);

  } else {
    // ====================================================================
    // 无 Pipeline: 同步读取
    // ====================================================================
    static uint8_t weight_buffer[256 * 1024]; // 256KB 临时缓冲区
    void *weights = weight_buffer;

    // Strategy 2: Compression - 解压缩
    if (opt->flags & OPT_COMPRESS) {
      static uint8_t decompress_buffer[512 * 1024]; // 512KB 解压缓冲

      // 读取压缩数据
      Flash_Read(flash_addr, weight_buffer, flash_size);
      opt->total_flash_read_bytes += flash_size;

      // 解压
      DecompressResult_t result;
      if (!Decompress_Auto(weight_buffer, flash_size, decompress_buffer,
                           opt->decompress_buffer_size, &result)) {
        return false;
      }

      weights = decompress_buffer;

    } else {
      // 直接读取未压缩权重
      Flash_Read(flash_addr, weight_buffer, flash_size);
      opt->total_flash_read_bytes += flash_size;
    }

    // CIM 计算
    CIM_Compute(input, output, weights, layer_index);
  }

  // ========================================================================
  // Strategy 3: Cascade - 检查早退出
  // ========================================================================
  if (should_exit) {
    *should_exit = false;
  }

  if (opt->flags & OPT_CASCADE && opt->cascade_model && should_exit) {
    uint8_t predicted_class;
    float confidence;

    // 假设 output 的维度与特征维度一致（需根据实际模型调整）
    uint32_t feature_dim = 128; // 示例值

    if (Cascade_ShouldExit(opt->cascade_model, layer_index, output, feature_dim,
                           &predicted_class, &confidence)) {
      *should_exit = true;

      // 计算节省的 Flash 读取量
      uint32_t saved =
          Cascade_CalculateSavedBytes(opt->cascade_model, layer_index);
      Cascade_UpdateStats(opt->cascade_model, true, saved);
    }
  }

  uint64_t end_time = GetTimestampUs();
  opt->total_inference_time_us += (end_time - start_time);

  return true;
}

bool FlashOpt_RunInference(FlashIOOptimizer_t *opt, uint8_t num_layers,
                           const uint32_t *layer_flash_addrs,
                           const uint32_t *layer_flash_sizes,
                           const float *input, float *output) {
  if (!opt || !layer_flash_addrs || !layer_flash_sizes || !input || !output) {
    return false;
  }

  static float layer_outputs[2][4096]; // Ping-pong 输出缓冲区
  const float *current_input = input;
  float *current_output = layer_outputs[0];

  for (uint8_t i = 0; i < num_layers; i++) {
    bool should_exit = false;

    if (!FlashOpt_InferLayer(opt, i, layer_flash_addrs[i], layer_flash_sizes[i],
                             current_input, current_output, &should_exit)) {
      return false;
    }

    // 早退出
    if (should_exit) {
      memcpy(output, current_output, sizeof(float) * 4096); // 复制最终输出
      break;
    }

    // Ping-pong 切换
    current_input = current_output;
    current_output = (current_output == layer_outputs[0]) ? layer_outputs[1]
                                                          : layer_outputs[0];
  }

  // 更新统计
  opt->inference_count++;

  // 如果未早退出，更新 Cascade 统计
  if (opt->flags & OPT_CASCADE && opt->cascade_model) {
    Cascade_UpdateStats(opt->cascade_model, false, 0);
  }

  return true;
}

// ============================================================================
// 性能分析
// ============================================================================

void FlashOpt_PrintStats(const FlashIOOptimizer_t *opt) {
  if (!opt)
    return;

  printf("\n======= Flash IO Optimizer Statistics =======\n");
  printf("Enabled Strategies: ");
  if (opt->flags & OPT_PIPELINE)
    printf("Pipeline ");
  if (opt->flags & OPT_COMPRESS)
    printf("Compression ");
  if (opt->flags & OPT_CASCADE)
    printf("Cascade ");
  if (opt->flags == OPT_NONE)
    printf("None (Baseline)");
  printf("\n");

  printf("\nOverall Performance:\n");
  printf("  Inference Count:     %lu\n", (unsigned long)opt->inference_count);
  printf("  Avg Inference Time:  %lu us\n",
         (unsigned long)FlashOpt_GetAverageInferenceTime(opt));
  printf("  Total Flash Read:    %lu KB\n",
         (unsigned long)(opt->total_flash_read_bytes / 1024));
  printf("  Flash Bandwidth:     %lu KB/s\n",
         (unsigned long)(FlashOpt_GetFlashBandwidth(opt) / 1024));

  // Pipeline 统计
  if (opt->flags & OPT_PIPELINE && opt->pipeline_ctrl) {
    printf("\nPipeline Statistics:\n");
    Pipeline_PrintStats(opt->pipeline_ctrl);
  }

  // Compression 统计
  if (opt->flags & OPT_COMPRESS) {
    printf("\nCompression Statistics:\n");
    Compression_PrintStats();
  }

  // Cascade 统计
  if (opt->flags & OPT_CASCADE && opt->cascade_model) {
    Cascade_PrintStats(opt->cascade_model);
  }

  printf("==============================================\n");
}

uint32_t FlashOpt_GetAverageInferenceTime(const FlashIOOptimizer_t *opt) {
  if (!opt || opt->inference_count == 0) {
    return 0;
  }

  return (uint32_t)(opt->total_inference_time_us / opt->inference_count);
}

uint64_t FlashOpt_GetFlashBandwidth(const FlashIOOptimizer_t *opt) {
  if (!opt || opt->total_inference_time_us == 0) {
    return 0;
  }

  // bytes/s = (total_bytes * 1000000) / time_us
  return (opt->total_flash_read_bytes * 1000000ULL) /
         opt->total_inference_time_us;
}

void FlashOpt_ResetStats(FlashIOOptimizer_t *opt) {
  if (!opt)
    return;

  opt->total_inference_time_us = 0;
  opt->total_flash_read_bytes = 0;
  opt->inference_count = 0;

  if (opt->pipeline_ctrl) {
    Pipeline_ResetStats(opt->pipeline_ctrl);
  }

  if (opt->cascade_model) {
    Cascade_ResetStats(opt->cascade_model);
  }

  Compression_ResetStats();
}
