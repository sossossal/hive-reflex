/*
 * IMC-22 CAN 驱动测试
 * 验证 CAN 初始化、发送和接收功能
 */

#include "imc22_sdk/imc22.h"
#include "imc22_sdk/imc22_can.h"
#include <stdio.h>

// 测试结果计数
static int tests_passed = 0;
static int tests_failed = 0;

// 简单断言宏
#define TEST_ASSERT(condition, message)                                        \
  if (condition) {                                                             \
    printf("[PASS] %s\n", message);                                            \
    tests_passed++;                                                            \
  } else {                                                                     \
    printf("[FAIL] %s\n", message);                                            \
    tests_failed++;                                                            \
  }

// 测试 CAN 初始化
void test_can_init() {
  printf("\n=== Testing CAN Init ===\n");

  CAN_Config_t config = {.baudrate = 1000000, // 1 Mbps
                         .mode = CAN_MODE_NORMAL,
                         .loopback = 0};

  int result = CAN_Init(CAN1, &config);
  TEST_ASSERT(result == 0, "CAN_Init should return 0");
}

// 测试 CAN 消息发送
void test_can_send() {
  printf("\n=== Testing CAN Send ===\n");

  CAN_Message_t msg = {.id = 0x123, .dlc = 3, .data = {0x01, 0x02, 0x03}};

  int result = CAN_Send(CAN1, &msg);
  TEST_ASSERT(result == 0, "CAN_Send should return 0");
}

// 测试 CAN 消息接收
void test_can_receive() {
  printf("\n=== Testing CAN Receive ===\n");

  CAN_Message_t msg;

  // 在仿真环境中，这应该超时
  int result = CAN_Receive(CAN1, &msg, 100);

  // 由于没有实际消息，应该返回超时
  TEST_ASSERT(result == -1, "CAN_Receive should timeout");
}

// 测试 CAN 过滤器
void test_can_filter() {
  printf("\n=== Testing CAN Filter ===\n");

  CAN_Filter_t filter = {.id = 0x100, .mask = 0x700, .mode = CAN_FILTER_MASK};

  int result = CAN_SetFilter(CAN1, 0, &filter);
  TEST_ASSERT(result == 0, "CAN_SetFilter should return 0");
}

// 主函数
int main(void) {
  printf("\n");
  printf("========================================\n");
  printf("IMC-22 CAN Driver Test Suite\n");
  printf("========================================\n");

  // 运行所有测试
  test_can_init();
  test_can_send();
  test_can_receive();
  test_can_filter();

  // 打印测试结果
  printf("\n========================================\n");
  printf("Test Results:\n");
  printf("  Passed: %d\n", tests_passed);
  printf("  Failed: %d\n", tests_failed);
  printf("========================================\n");

  if (tests_failed == 0) {
    printf("\n✓ All tests PASSED!\n\n");
    return 0;
  } else {
    printf("\n✗ Some tests FAILED!\n\n");
    return 1;
  }
}
