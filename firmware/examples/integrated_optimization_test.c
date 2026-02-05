/**
 * @file integrated_optimization_test.c
 * @brief Flash IO 优化 + 系统级优化 综合测试
 *
 * 同时测试:
 * - Flash IO: Pipeline + Compression + Cascade
 * - 系统优化: Prio Arbiter + TCM + Fixed-point Filter + CAN SYNC
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


// Flash IO 优化
#include "middleware/flash_io_optimizer.h"

// 系统级优化
#include "middleware/can_sync.h"
#include "middleware/fixed_point_filter.h"


// ============================================================================
// 配置
// ============================================================================

#define NUM_LAYERS 8
#define SYNC_ENABLED 1

// ============================================================================
// 全局状态 (放置在 TCM)
// ============================================================================

__attribute__((section(".tcm_data"))) static FlashIOOptimizer_t flash_opt;

__attribute__((section(".tcm_data"))) static PipelineController_t pipeline;

__attribute__((section(".tcm_data"))) static CascadeModel_t cascade;

__attribute__((section(".tcm_data"))) static ComplementaryFilter_t angle_filter;

__attribute__((section(".tcm_data"))) static CANSyncMgr_t sync_mgr;

__attribute__((section(".tcm_data"))) static float current_angle = 0.0f;

__attribute__((section(".tcm_data"))) static uint32_t inference_count = 0;

// ============================================================================
// 测试场景 1: 基线 (无优化)
// ============================================================================

void Test_Baseline(void) {
  printf("\n========================================\n");
  printf("Test 1: Baseline (No Optimization)\n");
  printf("========================================\n");

  FlashIOOptimizer_t opt;
  FlashOpt_Init(&opt, OPT_NONE);

  float input[128] = {0};
  float output[128];

  uint32_t layer_addrs[NUM_LAYERS] = {0x08000000, 0x08020000, 0x08040000,
                                      0x08060000, 0x08080000, 0x080A0000,
                                      0x080C0000, 0x080E0000};

  uint32_t layer_sizes[NUM_LAYERS] = {128 * 1024, 64 * 1024, 64 * 1024,
                                      32 * 1024,  32 * 1024, 16 * 1024,
                                      16 * 1024,  8 * 1024};

  uint32_t start = GetTimestampUs();

  for (int i = 0; i < 10; i++) {
    FlashOpt_RunInference(&opt, NUM_LAYERS, layer_addrs, layer_sizes, input,
                          output);
  }

  uint32_t elapsed = GetTimestampUs() - start;

  printf("Avg inference time: %.2f ms\n", elapsed / 10000.0f);
  FlashOpt_PrintStats(&opt);
}

// ============================================================================
// 测试场景 2: Flash IO 优化 (All Strategies)
// ============================================================================

void Test_FlashIO_Optimized(void) {
  printf("\n========================================\n");
  printf("Test 2: Flash IO Optimized\n");
  printf("========================================\n");

  // 初始化 Pipeline
  ModelConfig_t model_cfg = {.num_layers = NUM_LAYERS,
                             .max_layer_size = 128 * 1024,
                             .bank0_base = (void *)0x20000000,
                             .bank1_base = (void *)0x20040000,
                             .bank_size = 256 * 1024};
  Pipeline_Init(&pipeline, &model_cfg);

  uint32_t layer_addrs[NUM_LAYERS] = {0x08000000, 0x08020000, 0x08040000,
                                      0x08060000, 0x08080000, 0x080A0000,
                                      0x080C0000, 0x080E0000};

  uint32_t layer_sizes[NUM_LAYERS] = {
      64 * 1024, 32 * 1024, 32 * 1024, 16 * 1024, // 压缩后大小 (2:1)
      16 * 1024, 8 * 1024,  8 * 1024,  4 * 1024};

  for (uint8_t i = 0; i < NUM_LAYERS; i++) {
    Pipeline_AddLayerMapping(&pipeline, i, layer_addrs[i], layer_sizes[i]);
  }

  // 初始化 Cascade
  Cascade_Init(&cascade, 2, NUM_LAYERS);

  // 配置优化器
  FlashOpt_Init(&flash_opt, OPT_ALL);
  FlashOpt_ConfigPipeline(&flash_opt, &pipeline);
  FlashOpt_ConfigCompression(&flash_opt, COMPRESS_LZ4, 512 * 1024);
  FlashOpt_ConfigCascade(&flash_opt, &cascade);

  float input[128] = {0};
  float output[128];

  uint32_t start = GetTimestampUs();

  for (int i = 0; i < 10; i++) {
    // 模拟: 70% 场景可早退出
    if (i % 10 < 7) {
      input[0] = 0.9f; // 高置信度
    } else {
      input[0] = 0.3f; // 低置信度
    }

    FlashOpt_RunInference(&flash_opt, NUM_LAYERS, layer_addrs, layer_sizes,
                          input, output);
  }

  uint32_t elapsed = GetTimestampUs() - start;

  printf("Avg inference time: %.2f ms\n", elapsed / 10000.0f);
  FlashOpt_PrintStats(&flash_opt);
}

// ============================================================================
// 测试场景 3: 系统级优化 (定点滤波 + CAN SYNC)
// ============================================================================

__attribute__((section(".tcm_text"))) void Test_System_Optimized(void) {
  printf("\n========================================\n");
  printf("Test 3: System Optimized\n");
  printf("========================================\n");

  // 初始化定点滤波器
  CompFilter_Init(&angle_filter, 0.98f, 0.001f);

#if SYNC_ENABLED
  // 初始化 CAN SYNC (Slave 模式)
  CANSync_Init(&sync_mgr, false, NULL);
#endif

  printf("Testing fixed-point filter performance...\n");

  uint32_t start = GetTimestampUs();

  // 测试 1000 次滤波
  for (int i = 0; i < 1000; i++) {
    float gyro = 0.5f;
    float accel = 2.0f;
    current_angle = CompFilter_Update(&angle_filter, gyro, accel);
  }

  uint32_t elapsed = GetTimestampUs() - start;

  printf("1000 filter updates: %lu us (%.3f us/update)\n",
         (unsigned long)elapsed, elapsed / 1000.0f);

#if SYNC_ENABLED
  printf("\nCAN SYNC statistics:\n");
  CANSync_PrintStats(&sync_mgr);
#endif
}

// ============================================================================
// 测试场景 4: 完整集成测试
// ============================================================================

void Test_Full_Integration(void) {
  printf("\n========================================\n");
  printf("Test 4: Full Integration\n");
  printf("========================================\n");
  printf("Flash IO + System Optimizations\n");
  printf("========================================\n");

  // 重用已初始化的组件

  float input[128] = {0};
  float output[128];

  uint32_t layer_addrs[NUM_LAYERS] = {0x08000000, 0x08020000, 0x08040000,
                                      0x08060000, 0x08080000, 0x080A0000,
                                      0x080C0000, 0x080E0000};

  uint32_t layer_sizes[NUM_LAYERS] = {64 * 1024, 32 * 1024, 32 * 1024,
                                      16 * 1024, 16 * 1024, 8 * 1024,
                                      8 * 1024,  4 * 1024};

  uint32_t start = GetTimestampUs();

  for (int i = 0; i < 100; i++) {
    // Step 1: 读取传感器 (使用定点滤波)
    float gyro = 0.5f;
    float accel = 2.0f;
    current_angle = CompFilter_Update(&angle_filter, gyro, accel);

    // Step 2: 运行视觉推理 (Flash IO 优化)
    if (i % 10 < 7) {
      input[0] = 0.9f;
    } else {
      input[0] = 0.3f;
    }

    FlashOpt_RunInference(&flash_opt, NUM_LAYERS, layer_addrs, layer_sizes,
                          input, output);

    // Step 3: 控制输出
    inference_count++;
  }

  uint32_t elapsed = GetTimestampUs() - start;

  printf("\n--- Performance Summary ---\n");
  printf("Total time: %.2f ms\n", elapsed / 1000.0f);
  printf("Avg per iteration: %.2f ms\n", elapsed / 100000.0f);
  printf("Inferences completed: %lu\n", (unsigned long)inference_count);

  printf("\n--- Flash IO Stats ---\n");
  FlashOpt_PrintStats(&flash_opt);

  printf("\n--- System Stats ---\n");
  printf("Current angle: %.2f°\n", current_angle);
}

// ============================================================================
// 主函数
// ============================================================================

int main(void) {
  printf("==============================================\n");
  printf(" Integrated Optimization Test Suite\n");
  printf("==============================================\n");
  printf("Testing Flash IO + System Optimizations\n\n");

  // 运行所有测试
  Test_Baseline();
  Test_FlashIO_Optimized();
  Test_System_Optimized();
  Test_Full_Integration();

  printf("\n==============================================\n");
  printf(" All Tests Completed\n");
  printf("==============================================\n");

  return 0;
}

// ============================================================================
// 模拟 HAL 实现
// ============================================================================

uint32_t GetTimestampUs(void) {
  // TODO: 实际硬件定时器
  static uint32_t sim_time = 0;
  return sim_time += 100; // 模拟递增
}

uint32_t GetTickMs(void) { return GetTimestampUs() / 1000; }

void DelayMs(uint32_t ms) { (void)ms; }

void DelayUs(uint32_t us) { (void)us; }

void CAN_Transmit(uint16_t id, const void *data, uint8_t dlc) {
  (void)id;
  (void)data;
  (void)dlc;
}

void CAN_Receive(uint16_t *id, uint8_t *data, uint8_t *dlc) {
  (void)id;
  (void)data;
  (void)dlc;
}

// Flash 读取模拟
void Flash_Read(uint32_t addr, void *buffer, uint32_t size) {
  memset(buffer, 0xAB, size);
}

// CIM 计算模拟
void CIM_Compute(const float *input, float *output, const void *weights,
                 uint32_t layer_idx) {
  (void)weights;
  (void)layer_idx;
  for (int i = 0; i < 128; i++) {
    output[i] = input[i] * 0.99f;
  }
}
