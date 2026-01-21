/**
 * @file system_imc22.c
 * @brief IMC-22 系统配置和初始化
 */

#include "imc22.h"
#include "riscv_custom.h"

/* 系统时钟配置 */
static uint32_t SystemCoreClock = IMC22_SYSCLK_HZ;

/* ========================================================================= */
/* 时钟初始化                                                                */
/* ========================================================================= */

static void SystemClock_Config(void) {
  // 配置 PLL
  // 假设: 外部晶振 25MHz -> PLL -> 100MHz

  // TODO: 根据实际硬件实现
  // 1. 使能外部晶振
  // 2. 配置 PLL 倍频/分频
  // 3. 切换系统时钟源到 PLL
  // 4. 配置外设时钟分频

  SystemCoreClock = IMC22_SYSCLK_HZ;
}

/* ========================================================================= */
/* 总线配置                                                                  */
/* ========================================================================= */

static void Bus_Config(void) {
  // AHB 总线配置
  // 设置总线仲裁优先级

  // 优先级设置（降序）:
  // 1. CIM（最高）- 需要高带宽
  // 2. DMA - 批量数据传输
  // 3. RISC-V CPU - 正常优先级
  // 4. 外设 - 最低优先级

  // TODO: 配置总线仲裁器寄存器
}

/* ========================================================================= */
/* DMA 初始化                                                                */
/* ========================================================================= */

static void DMA_Init(void) {
  // 使能 DMA 时钟
  // 配置 DMA 通道

  for (int i = 0; i < 8; i++) {
    DMA->CH[i].CTRL = 0; // 复位所有通道
  }
}

/* ========================================================================= */
/* GPIO 初始化                                                               */
/* ========================================================================= */

static void GPIO_Init(void) {
  // 配置调试 LED
  GPIO->DIR |= (1 << 0) | (1 << 1);     // PA0, PA1 输出
  GPIO->DATA &= ~((1 << 0) | (1 << 1)); // 初始熄灭
}

/* ========================================================================= */
/* UART 初始化（用于调试日志）                                              */
/* ========================================================================= */

static void UART_Init(void) {
  // 配置波特率: 115200
  // 波特率分频 = PCLK / (16 * 115200)
  uint32_t baudrate_div = IMC22_PCLK_HZ / (16 * 115200);

  UART->BAUDRATE = baudrate_div;
  UART->CTRL = (1 << 0) | (1 << 1); // 使能 TX, RX
}

/* ========================================================================= */
/* 定时器初始化（系统时钟）                                                 */
/* ========================================================================= */

static void Timer_Init(void) {
  // 配置 1ms 定时器（用于 Millis()）

  // 定时器频率 = PCLK / (reload + 1)
  // 1kHz = PCLK / reload
  uint32_t reload = (IMC22_PCLK_HZ / 1000) - 1;

  TIMER->RELOAD = reload;
  TIMER->IRQ_EN = 1; // 使能中断
  TIMER->CTRL = 1;   // 启动定时器

  // 使能定时器中断
  riscv_csr_set(0x304, 1 << 7); // MIE[MTIE]
}

/* ========================================================================= */
/* 系统初始化主函数                                                          */
/* ========================================================================= */

void System_Init(void) {
  // 1. 时钟配置
  SystemClock_Config();

  // 2. 总线配置
  Bus_Config();

  // 3. DMA 初始化
  DMA_Init();

  // 4. GPIO 初始化
  GPIO_Init();

  // 5. UART 初始化
  UART_Init();

  // 6. 定时器初始化
  Timer_Init();

  // 7. 使能全局中断
  __enable_irq();
}

/* ========================================================================= */
/* 系统信息                                                                  */
/* ========================================================================= */

uint32_t SystemCoreClockGet(void) { return SystemCoreClock; }

void SystemInfo_Print(void) {
  printf("\n");
  printf("╔═══════════════════════════════════════════╗\n");
  printf("║       IMC-22 系统信息                     ║\n");
  printf("╚═══════════════════════════════════════════╝\n");
  printf("  CPU:        RISC-V RV32IMAC @ %lu MHz\n",
         SystemCoreClock / 1000000);
  printf("  SRAM:       %lu KB\n", SRAM_SIZE / 1024);
  printf("  CIM SRAM:   %lu KB\n", CIM_SIZE / 1024);
  printf("  FLASH:      %lu KB\n", FLASH_SIZE / 1024);
  printf("  外设时钟:   %lu MHz\n", IMC22_PCLK_HZ / 1000000);
  printf("\n");
}

/* ========================================================================= */
/* 总线工具函数                                                              */
/* ========================================================================= */

/**
 * @brief DMA 传输
 */
int DMA_Transfer(uint8_t channel, void *src, void *dst, uint32_t size) {
  if (channel >= 8) {
    return -1;
  }

  DMA_Channel_TypeDef *ch = &DMA->CH[channel];

  // 等待通道空闲
  while (ch->STATUS & DMA_CTRL_ENABLE)
    ;

  // 配置 DMA
  ch->SRC_ADDR = (uint32_t)src;
  ch->DST_ADDR = (uint32_t)dst;
  ch->COUNT = size;
  ch->CTRL = DMA_CTRL_ENABLE | DMA_CTRL_MEM2MEM;

  return 0;
}

/**
 * @brief 等待 DMA 完成
 */
int DMA_Wait(uint8_t channel, uint32_t timeout_ms) {
  if (channel >= 8) {
    return -1;
  }

  uint32_t start = Millis();
  DMA_Channel_TypeDef *ch = &DMA->CH[channel];

  while (ch->STATUS & DMA_CTRL_ENABLE) {
    if (Millis() - start > timeout_ms) {
      return -1; // 超时
    }
  }

  return 0;
}

/* ========================================================================= */
/* 简单的 printf 实现（通过 UART）                                          */
/* ========================================================================= */

#include <stdarg.h>

static void uart_putc(char c) {
  while (!(UART->STATUS & UART_STATUS_TXE))
    ;
  UART->DATA = c;
}

static void uart_puts(const char *str) {
  while (*str) {
    if (*str == '\n') {
      uart_putc('\r');
    }
    uart_putc(*str++);
  }
}

// 简化的 printf (仅支持 %s, %d, %lu, %x)
int printf(const char *format, ...) {
  char buffer[256];
  va_list args;
  va_start(args, format);

  // TODO: 实现完整的格式化输出
  // 这里使用简化版本

  uart_puts(format);

  va_end(args);
  return 0;
}
