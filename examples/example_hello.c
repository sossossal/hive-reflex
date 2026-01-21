/**
 * @file example_hello.c
 * @brief IMC-22 Hello World 示例
 *
 * 功能: 在 UART 上打印 "Hello from IMC-22!" 并闪烁 LED
 */

#include "imc22.h"
#include <stdio.h>

/* 简易 UART 打印函数 */
void uart_putc(char c) {
  while (!(UART->STATUS & UART_STATUS_TXE))
    ;
  UART->DATA = c;
}

void uart_puts(const char *str) {
  while (*str) {
    uart_putc(*str++);
  }
}

/* LED 引脚 (假设 GPIO bit 0) */
#define LED_PIN 0

int main(void) {
  /* 1. 初始化 UART (115200 baud) */
  // 波特率分频 = SYSCLK / (16 * baud)
  UART->BAUD = IMC22_SYSCLK_HZ / (16 * 115200);
  UART->CTRL = 0x3; // 使能 TX/RX

  /* 2. 初始化 GPIO */
  GPIO->DIR |= (1 << LED_PIN); // 设置为输出

  /* 3. 初始化定时器 (1Hz 中断) */
  TIMER->LOAD = IMC22_SYSCLK_HZ; // 1 秒
  TIMER->CTRL = TIMER_CTRL_EN | TIMER_CTRL_MODE | TIMER_CTRL_IE;
  NVIC_EnableIRQ(IRQ_TIMER);

  /* 4. 打印欢迎信息 */
  uart_puts("\r\n=================================\r\n");
  uart_puts("  IMC-22 SDK v1.0\r\n");
  uart_puts("  Hello from Hive-Reflex!\r\n");
  uart_puts("=================================\r\n\r\n");

  /* 5. 主循环 */
  uint32_t counter = 0;
  while (1) {
    char buf[64];
    snprintf(buf, sizeof(buf), "Running... count = %lu\r\n", counter++);
    uart_puts(buf);

    DelayMs(1000);
  }

  return 0;
}

/* 定时器中断: 每秒闪烁LED */
void TIMER_IRQHandler(void) {
  // 清除中断标志
  TIMER->STATUS = 0;

  // 翻转 LED
  GPIO->TOGGLE = (1 << LED_PIN);
}
