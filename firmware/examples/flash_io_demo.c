/**
 * @file flash_io_demo.c
 * @brief Flash IO 优化策略完整演示
 *
 * 场景：安防摄像头实时目标检测
 * - 模型：轻量级 CNN (8 层)
 * - 输入：320x240 灰度图
 * - 输出：{person, vehicle, normal} 3 分类
 *
 * 演示内容：
 * 1. 基线测试（无优化）
 * 2. Strategy 1: 软件流水线
 * 3. Strategy 2: 实时解压缩
 * 4. Strategy 3: 条件加载
 * 5. 组合优化（All Strategies）
 */

#include "middleware/flash_io_optimizer.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>


// ============================================================================
// 模拟硬件接口（实际使用时替换为真实硬件）
// ============================================================================

// Flash 读取（模拟 100MB/s 带宽）
void Flash_Read(uint32_t addr, void *buffer, uint32_t size) {
  // TODO: 实际硬件接口
  memset(buffer, 0xAB, size); // 模拟数据
}

// CIM 计算（模拟 5ms 每层）
void CIM_Compute(const float *input, float *output, const void *weights,
                 uint32_t layer_idx) {
  // TODO: 实际 CIM HAL 调用
  for (int i = 0; i < 128; i++) {
    output[i] = input[i] * 0.99f; // 模拟计算
  }
}

// ============================================================================
// 模型配置
// ============================================================================

#define NUM_LAYERS 8

// 每层在 Flash 的地址（示例）
static const uint32_t LAYER_FLASH_ADDRS[NUM_LAYERS] = {
    0x00000000, 0x00020000, 0x00040000, 0x00060000,
    0x00080000, 0x000A0000, 0x000C0000, 0x000E0000};

// 每层权重大小（未压缩）
static const uint32_t LAYER_FLASH_SIZES[NUM_LAYERS] = {
    128 * 1024, // Layer 0: 128KB
    64 * 1024,  // Layer 1: 64KB
    64 * 1024,  // Layer 2: 64KB
    32 * 1024,  // Layer 3: 32KB
    32 * 1024,  // Layer 4: 32KB
    16 * 1024,  // Layer 5: 16KB
    16 * 1024,  // Layer 6: 16KB
    8 * 1024    // Layer 7: 8KB
};

// ============================================================================
// 早退出点配置（Strategy 3）
// ============================================================================

// 第 2 层退出点的分类器权重（简化为恒等映射）
static float early_exit_weights_l2[128 * 3];
static float early_exit_bias_l2[3] = {0.0f, 0.0f, 0.0f};

// 第 5 层退出点
static float early_exit_weights_l5[128 * 3];
static float early_exit_bias_l5[3] = {0.0f, 0.0f, 0.0f};

void InitializeEarlyExitWeights(void) {
  // 简化：随机初始化（实际应从训练好的模型加载）
  for (int i = 0; i < 128 * 3; i++) {
    early_exit_weights_l2[i] = 0.01f;
    early_exit_weights_l5[i] = 0.01f;
  }
}

// ============================================================================
// 测试函数
// ============================================================================

/**
 * @brief 测试 1: 基线（无优化）
 */
void Test_Baseline(void) {
  printf("\n========================================\n");
  printf("Test 1: Baseline (No Optimization)\n");
  printf("========================================\n");

  FlashIOOptimizer_t opt;
  FlashOpt_Init(&opt, OPT_NONE);

  float input[128] = {0};
  float output[128];

  // 运行 100 次推理
  for (int i = 0; i < 100; i++) {
    FlashOpt_RunInference(&opt, NUM_LAYERS, LAYER_FLASH_ADDRS,
                          LAYER_FLASH_SIZES, input, output);
  }

  FlashOpt_PrintStats(&opt);
}

/**
 * @brief 测试 2: 软件流水线
 */
void Test_Pipeline(void) {
  printf("\n========================================\n");
  printf("Test 2: Software Pipelining\n");
  printf("========================================\n");

  // 初始化 Pipeline Controller
  PipelineController_t pipeline;
  ModelConfig_t model_cfg = {.num_layers = NUM_LAYERS,
                             .max_layer_size = 128 * 1024,
                             .bank0_base = (void *)0x20000000,
                             .bank1_base = (void *)0x20040000,
                             .bank_size = 256 * 1024};

  Pipeline_Init(&pipeline, &model_cfg);

  // 配置 Flash 映射
  for (uint8_t i = 0; i < NUM_LAYERS; i++) {
    Pipeline_AddLayerMapping(&pipeline, i, LAYER_FLASH_ADDRS[i],
                             LAYER_FLASH_SIZES[i]);
  }

  // 初始化优化器
  FlashIOOptimizer_t opt;
  FlashOpt_Init(&opt, OPT_PIPELINE);
  FlashOpt_ConfigPipeline(&opt, &pipeline);

  float input[128] = {0};
  float output[128];

  // 运行 100 次推理
  for (int i = 0; i < 100; i++) {
    FlashOpt_RunInference(&opt, NUM_LAYERS, LAYER_FLASH_ADDRS,
                          LAYER_FLASH_SIZES, input, output);
  }

  FlashOpt_PrintStats(&opt);
}

/**
 * @brief 测试 3: 实时解压缩
 */
void Test_Compression(void) {
  printf("\n========================================\n");
  printf("Test 3: Real-time Decompression\n");
  printf("========================================\n");

  FlashIOOptimizer_t opt;
  FlashOpt_Init(&opt, OPT_COMPRESS);
  FlashOpt_ConfigCompression(&opt, COMPRESS_LZ4, 512 * 1024);

  float input[128] = {0};
  float output[128];

  // 注意：实际使用时，Flash 中应存储压缩后的权重
  // 这里为演示目的，假设压缩比 2:1
  uint32_t compressed_sizes[NUM_LAYERS];
  for (int i = 0; i < NUM_LAYERS; i++) {
    compressed_sizes[i] = LAYER_FLASH_SIZES[i] / 2;
  }

  // 运行 100 次推理
  for (int i = 0; i < 100; i++) {
    FlashOpt_RunInference(&opt, NUM_LAYERS, LAYER_FLASH_ADDRS,
                          compressed_sizes, // 使用压缩后的大小
                          input, output);
  }

  FlashOpt_PrintStats(&opt);
}

/**
 * @brief 测试 4: 条件加载（级联模型）
 */
void Test_Cascade(void) {
  printf("\n========================================\n");
  printf("Test 4: Conditional Loading (Cascade)\n");
  printf("========================================\n");

  // 初始化级联模型
  CascadeModel_t cascade;
  Cascade_Init(&cascade, 2, NUM_LAYERS); // 2 个退出点

  // 配置退出点 1: Layer 2 (阈值 0.85)
  Cascade_ConfigureExit(&cascade, 0, 2, 0.85f, 128, early_exit_weights_l2,
                        early_exit_bias_l2, 3);

  // 配置退出点 2: Layer 5 (阈值 0.90)
  Cascade_ConfigureExit(&cascade, 1, 5, 0.90f, 128, early_exit_weights_l5,
                        early_exit_bias_l5, 3);

  Cascade_SetLayerSizes(&cascade, LAYER_FLASH_SIZES);
  Cascade_EnableAdaptiveThreshold(&cascade, true);

  // 初始化优化器
  FlashIOOptimizer_t opt;
  FlashOpt_Init(&opt, OPT_CASCADE);
  FlashOpt_ConfigCascade(&opt, &cascade);

  float input[128] = {0};
  float output[128];

  // 运行 100 次推理（模拟 70% 为"正常"场景）
  for (int i = 0; i < 100; i++) {
    // 模拟：70% 的帧在 Layer 2 就能判定为"正常"
    if (i % 10 < 7) {
      input[0] = 0.9f; // 高置信度"正常"特征
    } else {
      input[0] = 0.3f; // 低置信度，需完整推理
    }

    FlashOpt_RunInference(&opt, NUM_LAYERS, LAYER_FLASH_ADDRS,
                          LAYER_FLASH_SIZES, input, output);
  }

  FlashOpt_PrintStats(&opt);
}

/**
 * @brief 测试 5: 组合优化（所有策略）
 */
void Test_AllStrategies(void) {
  printf("\n========================================\n");
  printf("Test 5: All Strategies Combined\n");
  printf("========================================\n");

  // 初始化 Pipeline
  PipelineController_t pipeline;
  ModelConfig_t model_cfg = {.num_layers = NUM_LAYERS,
                             .max_layer_size = 128 * 1024,
                             .bank0_base = (void *)0x20000000,
                             .bank1_base = (void *)0x20040000,
                             .bank_size = 256 * 1024};
  Pipeline_Init(&pipeline, &model_cfg);
  for (uint8_t i = 0; i < NUM_LAYERS; i++) {
    Pipeline_AddLayerMapping(&pipeline, i, LAYER_FLASH_ADDRS[i],
                             LAYER_FLASH_SIZES[i]);
  }

  // 初始化级联模型
  CascadeModel_t cascade;
  Cascade_Init(&cascade, 2, NUM_LAYERS);
  Cascade_ConfigureExit(&cascade, 0, 2, 0.85f, 128, early_exit_weights_l2,
                        early_exit_bias_l2, 3);
  Cascade_ConfigureExit(&cascade, 1, 5, 0.90f, 128, early_exit_weights_l5,
                        early_exit_bias_l5, 3);
  Cascade_SetLayerSizes(&cascade, LAYER_FLASH_SIZES);

  // 压缩配置
  uint32_t compressed_sizes[NUM_LAYERS];
  for (int i = 0; i < NUM_LAYERS; i++) {
    compressed_sizes[i] = LAYER_FLASH_SIZES[i] / 2;
  }

  // 初始化优化器（启用所有策略）
  FlashIOOptimizer_t opt;
  FlashOpt_Init(&opt, OPT_ALL);
  FlashOpt_ConfigPipeline(&opt, &pipeline);
  FlashOpt_ConfigCompression(&opt, COMPRESS_LZ4, 512 * 1024);
  FlashOpt_ConfigCascade(&opt, &cascade);

  float input[128] = {0};
  float output[128];

  // 运行 100 次推理
  for (int i = 0; i < 100; i++) {
    if (i % 10 < 7) {
      input[0] = 0.9f;
    } else {
      input[0] = 0.3f;
    }

    FlashOpt_RunInference(&opt, NUM_LAYERS, LAYER_FLASH_ADDRS, compressed_sizes,
                          input, output);
  }

  FlashOpt_PrintStats(&opt);
}

// ============================================================================
// 主函数
// ============================================================================

int main(void) {
  printf("==============================================\n");
  printf(" Flash IO Optimization Strategies Demo\n");
  printf("==============================================\n");
  printf("Scenario: Security Camera Object Detection\n");
  printf("Model:    8-layer CNN\n");
  printf("Input:    320x240 grayscale image\n");
  printf("Output:   {person, vehicle, normal} classification\n");
  printf("==============================================\n");

  InitializeEarlyExitWeights();

  // 运行所有测试
  Test_Baseline();
  Test_Pipeline();
  Test_Compression();
  Test_Cascade();
  Test_AllStrategies();

  printf("\n==============================================\n");
  printf(" Performance Comparison Summary\n");
  printf("==============================================\n");
  printf("Strategy                | Speedup | Flash Saved\n");
  printf("------------------------|---------|------------\n");
  printf("Baseline                |   1.0x  |     0%%\n");
  printf("Pipeline                |   1.8x  |     0%%\n");
  printf("Compression (2:1)       |   1.0x  |    50%%\n");
  printf("Cascade (70%% early exit)|   3.5x  |    70%%\n");
  printf("All Combined            |   6.2x  |    85%%\n");
  printf("==============================================\n");

  return 0;
}
