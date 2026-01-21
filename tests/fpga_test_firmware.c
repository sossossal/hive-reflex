/**
 * @file fpga_test_firmware.c
 * @brief Hive-Reflex 2.1 FPGA 硬件测试固件
 *
 * 用于验证 FPGA 上的稀疏 MAC、DVFS 和 CIM 功能
 * 目标平台: ZCU102 (RISC-V 软核或 ARM Cortex-A53)
 */

#include "imc22.h"
#include "imc22_cim.h"
#include "imc22_dvfs.h"
#include "tinyml_adaptive.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>


/* ========================================================================= */
/* 测试配置                                                                  */
/* ========================================================================= */

#define TEST_MAC_COUNT 256
#define TEST_ITERATIONS 1000
#define SPARSE_THRESHOLD 2
#define VERBOSE_OUTPUT 1

/* ========================================================================= */
/* 测试结果结构                                                              */
/* ========================================================================= */

typedef struct {
  const char *name;
  bool passed;
  uint32_t latency_us;
  float result_value;
  char message[128];
} TestResult_t;

static TestResult_t g_test_results[32];
static int g_test_count = 0;

/* ========================================================================= */
/* 辅助函数                                                                  */
/* ========================================================================= */

static void log_test(const char *name, bool passed, const char *msg) {
  if (g_test_count < 32) {
    TestResult_t *r = &g_test_results[g_test_count++];
    r->name = name;
    r->passed = passed;
    strncpy(r->message, msg, sizeof(r->message) - 1);

#if VERBOSE_OUTPUT
    printf("[%s] %s: %s\n", passed ? "PASS" : "FAIL", name, msg);
#endif
  }
}

static uint32_t get_time_us(void) {
  /* 硬件定时器读取 */
  return GetCycleCount() / (IMC22_SYSCLK_HZ / 1000000);
}

/* ========================================================================= */
/* 测试 1: CIM 基础功能                                                      */
/* ========================================================================= */

static void test_cim_basic(void) {
  printf("\n=== Test 1: CIM Basic Functionality ===\n");

  int8_t input[TEST_MAC_COUNT];
  int8_t weights[TEST_MAC_COUNT];

  /* 初始化测试数据 */
  for (int i = 0; i < TEST_MAC_COUNT; i++) {
    input[i] = (i % 127) + 1;
    weights[i] = ((i * 3) % 127) + 1;
  }

  /* 加载权重 */
  int ret = CIM_LoadWeights(weights, TEST_MAC_COUNT);
  if (ret != 0) {
    log_test("CIM_LoadWeights", false, "Failed to load weights");
    return;
  }
  log_test("CIM_LoadWeights", true, "Weights loaded successfully");

  /* 执行计算 */
  int32_t result;
  uint32_t start = get_time_us();
  ret = CIM_Compute(input, TEST_MAC_COUNT, &result);
  uint32_t elapsed = get_time_us() - start;

  if (ret != 0) {
    log_test("CIM_Compute", false, "Compute failed");
    return;
  }

  /* 验证结果 */
  int32_t expected = 0;
  for (int i = 0; i < TEST_MAC_COUNT; i++) {
    expected += (int32_t)input[i] * (int32_t)weights[i];
  }

  char msg[128];
  if (result == expected) {
    snprintf(msg, sizeof(msg), "Result correct: %d, Latency: %u us", result,
             elapsed);
    log_test("CIM_Compute", true, msg);
  } else {
    snprintf(msg, sizeof(msg), "Result mismatch: got %d, expected %d", result,
             expected);
    log_test("CIM_Compute", false, msg);
  }
}

/* ========================================================================= */
/* 测试 2: 稀疏计算                                                          */
/* ========================================================================= */

static void test_sparse_compute(void) {
  printf("\n=== Test 2: Sparse Computation ===\n");

  int8_t input[TEST_MAC_COUNT];
  int8_t weights[TEST_MAC_COUNT];

  /* 初始化 50% 稀疏数据 */
  for (int i = 0; i < TEST_MAC_COUNT; i++) {
    input[i] = (i % 2 == 0) ? 0 : ((i % 63) + 5);
    weights[i] = ((i * 7) % 127) + 1;
  }

  CIM_LoadWeights(weights, TEST_MAC_COUNT);

  /* 启用稀疏模式 */
  CIM_EnableSparse(true, SPARSE_THRESHOLD);

  int32_t result;
  uint32_t start = get_time_us();
  CIM_Compute(input, TEST_MAC_COUNT, &result);
  uint32_t elapsed_sparse = get_time_us() - start;

  /* 读取稀疏统计 */
  uint16_t total_ops, skipped_ops;
  CIM_GetSparseStats(&total_ops, &skipped_ops);

  float sparsity = (float)skipped_ops / total_ops * 100;

  char msg[128];
  snprintf(msg, sizeof(msg), "Sparsity: %.1f%%, Skipped: %u/%u, Latency: %u us",
           sparsity, skipped_ops, total_ops, elapsed_sparse);

  /* 验证稀疏率 */
  if (sparsity > 40.0f) {
    log_test("Sparse_Compute", true, msg);
  } else {
    log_test("Sparse_Compute", false, msg);
  }

  /* 禁用稀疏模式对比 */
  CIM_EnableSparse(false, 0);

  start = get_time_us();
  CIM_Compute(input, TEST_MAC_COUNT, &result);
  uint32_t elapsed_dense = get_time_us() - start;

  float speedup = (float)elapsed_dense / elapsed_sparse;
  snprintf(msg, sizeof(msg), "Sparse: %u us, Dense: %u us, Speedup: %.2fx",
           elapsed_sparse, elapsed_dense, speedup);
  log_test("Sparse_Speedup", speedup > 1.0f, msg);
}

/* ========================================================================= */
/* 测试 3: DVFS 功能                                                         */
/* ========================================================================= */

static void test_dvfs(void) {
  printf("\n=== Test 3: DVFS Functionality ===\n");

  /* 初始化 DVFS */
  DVFSConfig_t config = {.enable = true,
                         .auto_scale = false, /* 手动模式 */
                         .initial_mode = DVFS_MODE_ACTIVE};

  int ret = DVFS_Init(&config);
  if (ret != 0) {
    log_test("DVFS_Init", false, "Initialization failed");
    return;
  }
  log_test("DVFS_Init", true, "Initialized successfully");

  /* 测试模式切换 */
  const struct {
    DVFSMode_t mode;
    const char *name;
    float expected_power_mw;
  } modes[] = {
      {DVFS_MODE_ACTIVE, "Active", 50.0f},
      {DVFS_MODE_STANDBY, "Standby", 0.5f},
      {DVFS_MODE_DEEPSLEEP, "DeepSleep", 0.0001f},
  };

  for (int i = 0; i < 3; i++) {
    ret = DVFS_SetMode(modes[i].mode);

    if (ret != 0) {
      char msg[128];
      snprintf(msg, sizeof(msg), "Failed to set %s mode", modes[i].name);
      log_test(modes[i].name, false, msg);
      continue;
    }

    /* 等待转换完成 */
    DVFS_WaitReady(100);

    DVFSMode_t current = DVFS_GetMode();
    float power = DVFS_GetEstimatedPower();

    char msg[128];
    snprintf(msg, sizeof(msg), "Mode: %s, Power: %.4f mW", modes[i].name,
             power);
    log_test(modes[i].name, (current == modes[i].mode), msg);
  }

  /* 恢复 Active 模式 */
  DVFS_SetMode(DVFS_MODE_ACTIVE);
}

/* ========================================================================= */
/* 测试 4: DVFS 自动缩放                                                     */
/* ========================================================================= */

static void test_dvfs_autoscale(void) {
  printf("\n=== Test 4: DVFS Auto-Scaling ===\n");

  /* 启用自动缩放 */
  DVFS_EnableAutoScale(true, 50, 200);
  DVFS_SetIdleTimeout(500);

  /* 模拟低利用率 */
  DVFS_ReportUtilization(30);
  DelayMs(100);

  DVFSMode_t mode_low = DVFS_GetMode();
  char msg[128];
  snprintf(msg, sizeof(msg), "Low util (30%%): Mode = %d", mode_low);
  log_test("DVFS_LowUtil", (mode_low == DVFS_MODE_STANDBY), msg);

  /* 模拟高利用率 */
  DVFS_ReportUtilization(220);
  DelayMs(100);

  DVFSMode_t mode_high = DVFS_GetMode();
  snprintf(msg, sizeof(msg), "High util (220%%): Mode = %d", mode_high);
  log_test("DVFS_HighUtil", (mode_high == DVFS_MODE_ACTIVE), msg);

  DVFS_EnableAutoScale(false, 0, 0);
}

/* ========================================================================= */
/* 测试 5: TinyML 自适应控制                                                 */
/* ========================================================================= */

static void test_tinyml_adaptive(void) {
  printf("\n=== Test 5: TinyML Adaptive Control ===\n");

  /* 初始化 TinyML */
  int ret = TinyML_InitDefault();
  if (ret != 0) {
    log_test("TinyML_Init", false, "Initialization failed");
    return;
  }
  log_test("TinyML_Init", true, "Initialized with default model");

  /* 正常负载测试 */
  SensorFeedback_t normal = {.torque = 2.0f,
                             .velocity = 1.0f,
                             .position_error = 0.05f,
                             .external_force = 0.0f};

  AdaptiveState_t state_normal = TinyML_ComputeAdaptive(&normal);

  char msg[128];
  snprintf(msg, sizeof(msg), "PID: %.2f, Neural: %.2f, Compliance: %.2f",
           state_normal.pid_weight, state_normal.neural_weight,
           state_normal.compliance);
  log_test("TinyML_Normal",
           (state_normal.pid_weight > 0.3f && state_normal.pid_weight < 0.7f),
           msg);

  /* 高负载测试 */
  SensorFeedback_t high_load = {.torque = 9.0f,
                                .velocity = 0.5f,
                                .position_error = 0.2f,
                                .external_force = 5.0f};

  AdaptiveState_t state_high = TinyML_ComputeAdaptive(&high_load);

  snprintf(msg, sizeof(msg), "High load - PID: %.2f, Neural: %.2f",
           state_high.pid_weight, state_high.neural_weight);
  log_test("TinyML_HighLoad", (state_high.pid_weight > 0.7f), msg);

  /* 检查高负载模式 */
  log_test("TinyML_HighLoadMode", state_high.high_load_mode,
           state_high.high_load_mode ? "High load detected"
                                     : "High load NOT detected");
}

/* ========================================================================= */
/* 测试 6: 性能基准                                                          */
/* ========================================================================= */

static void test_performance_benchmark(void) {
  printf("\n=== Test 6: Performance Benchmark ===\n");

  int8_t input[TEST_MAC_COUNT];
  int8_t weights[TEST_MAC_COUNT];
  int32_t result;

  /* 初始化数据 */
  for (int i = 0; i < TEST_MAC_COUNT; i++) {
    input[i] = (i % 127) + 1;
    weights[i] = ((i * 3) % 127) + 1;
  }

  CIM_LoadWeights(weights, TEST_MAC_COUNT);

  /* 密集模式性能 */
  CIM_EnableSparse(false, 0);

  uint32_t start = get_time_us();
  for (int i = 0; i < TEST_ITERATIONS; i++) {
    CIM_Compute(input, TEST_MAC_COUNT, &result);
  }
  uint32_t elapsed_dense = get_time_us() - start;

  float dense_latency = (float)elapsed_dense / TEST_ITERATIONS;

  /* 稀疏模式性能 */
  CIM_EnableSparse(true, SPARSE_THRESHOLD);

  start = get_time_us();
  for (int i = 0; i < TEST_ITERATIONS; i++) {
    CIM_Compute(input, TEST_MAC_COUNT, &result);
  }
  uint32_t elapsed_sparse = get_time_us() - start;

  float sparse_latency = (float)elapsed_sparse / TEST_ITERATIONS;

  char msg[128];
  snprintf(msg, sizeof(msg), "Dense: %.2f us, Sparse: %.2f us, %d iterations",
           dense_latency, sparse_latency, TEST_ITERATIONS);
  log_test("Benchmark_Latency", (dense_latency < 10.0f), msg);

  /* 吞吐量 */
  float throughput_mops =
      (float)TEST_MAC_COUNT * TEST_ITERATIONS / elapsed_dense;
  snprintf(msg, sizeof(msg), "Throughput: %.2f MOPS (million ops/sec)",
           throughput_mops);
  log_test("Benchmark_Throughput", (throughput_mops > 10.0f), msg);
}

/* ========================================================================= */
/* 主测试函数                                                                */
/* ========================================================================= */

int run_fpga_tests(void) {
  printf("\n");
  printf("##############################################\n");
  printf("#  Hive-Reflex 2.1 FPGA Hardware Tests       #\n");
  printf("##############################################\n");
  printf("\n");

  /* 系统初始化 */
  printf("Initializing hardware...\n");
  // SystemInit();

  /* 执行测试 */
  test_cim_basic();
  test_sparse_compute();
  test_dvfs();
  test_dvfs_autoscale();
  test_tinyml_adaptive();
  test_performance_benchmark();

  /* 测试汇总 */
  printf("\n");
  printf("##############################################\n");
  printf("#  Test Summary                              #\n");
  printf("##############################################\n");
  printf("\n");

  int pass_count = 0;
  int fail_count = 0;

  for (int i = 0; i < g_test_count; i++) {
    TestResult_t *r = &g_test_results[i];
    printf("  [%s] %s\n", r->passed ? "PASS" : "FAIL", r->name);
    if (r->passed)
      pass_count++;
    else
      fail_count++;
  }

  printf("\n");
  printf("Total: %d passed, %d failed\n", pass_count, fail_count);
  printf("\n");

  if (fail_count == 0) {
    printf("*** ALL TESTS PASSED ***\n");
    return 0;
  } else {
    printf("*** SOME TESTS FAILED ***\n");
    return 1;
  }
}

/* 入口点 */
int main(void) { return run_fpga_tests(); }
