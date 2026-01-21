/**
 * @file minimal_demo.c
 * @brief IMC-22 最小化演示程序
 *
 * 演示: 初始化 + 简单逻辑 + 循环
 */

#include <stdint.h>

// IMC-22 基础定义
#define IMC22_SYSCLK_HZ 100000000 // 100 MHz

// 简单的延迟函数
void delay_cycles(uint32_t cycles) {
  volatile uint32_t i;
  for (i = 0; i < cycles; i++) {
    asm volatile("nop");
  }
}

// 主函数
int main(void) {
  uint32_t counter = 0;

  // 简单的主循环
  while (1) {
    counter++;

    // 延迟
    delay_cycles(1000000); // 约 0.01 秒 @ 100MHz

    // 每 10 次循环后停止（for testing）
    if (counter >= 10) {
      break;
    }
  }

  return 0;
}

// _start 函数（裸机入口）
void _start(void) {
  // 调用 main
  main();

  // 程序结束后无限循环
  while (1)
    ;
}
