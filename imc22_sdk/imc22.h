/**
 * @file imc22.h
 * @brief IMC-22 主头文件 (更新版)
 * 包含所有硬件定义和系统配置
 * @version 2.0
 */

#ifndef IMC22_H
#define IMC22_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 系统配置                                                                  */
/* ========================================================================= */

#ifndef IMC22_SYSCLK_HZ
#define IMC22_SYSCLK_HZ 100000000 // 100 MHz 系统时钟
#endif

#define IMC22_PCLK_HZ (IMC22_SYSCLK_HZ / 2) // 50 MHz 外设时钟

/* ========================================================================= */
/* 内存映射                                                                  */
/* ========================================================================= */

// FLASH (XIP)
#define FLASH_BASE 0x08000000
#define FLASH_SIZE (2 * 1024 * 1024) // 2MB

// SRAM
#define SRAM_BASE 0x20000000
#define SRAM_SIZE (512 * 1024) // 512KB

// CIM SRAM
#define CIM_BASE 0x50000000
#define CIM_SIZE (512 * 1024) // 512KB

// 外设基地址
#define PERIPH_BASE 0x40000000

// 各外设偏移
#define RBB_OFFSET 0x0001000
#define FLASH_CTRL_OFFSET 0x00020000
#define GPIO_OFFSET 0x00030000
#define UART_OFFSET 0x00040000
#define SPI_OFFSET 0x00050000
#define I2C_OFFSET 0x00060000
#define PWM_OFFSET 0x00070000
#define ADC_OFFSET 0x00080000
#define CAN_OFFSET 0x00090000
#define DMA_OFFSET 0x000A0000
#define TIMER_OFFSET 0x000B0000

/* ========================================================================= */
/* 外设基地址                                                                */
/* ========================================================================= */

#define RBB_BASE (PERIPH_BASE + RBB_OFFSET)
#define FLASH_CTRL_BASE (PERIPH_BASE + FLASH_CTRL_OFFSET)
#define GPIO_BASE (PERIPH_BASE + GPIO_OFFSET)
#define UART_BASE (PERIPH_BASE + UART_OFFSET)
#define SPI_BASE (PERIPH_BASE + SPI_OFFSET)
#define I2C_BASE (PERIPH_BASE + I2C_OFFSET)
#define PWM_BASE (PERIPH_BASE + PWM_OFFSET)
#define ADC_BASE (PERIPH_BASE + ADC_OFFSET)
#define CAN_BASE (PERIPH_BASE + CAN_OFFSET)
#define DMA_BASE (PERIPH_BASE + DMA_OFFSET)
#define TIMER_BASE (PERIPH_BASE + TIMER_OFFSET)

/* ========================================================================= */
/* GPIO 定义                                                                 */
/* ========================================================================= */

typedef struct {
  volatile uint32_t DATA;       // 数据寄存器
  volatile uint32_t DIR;        // 方向寄存器
  volatile uint32_t SET;        // 置位寄存器
  volatile uint32_t CLR;        // 清零寄存器
  volatile uint32_t TOGGLE;     // 翻转寄存器
  volatile uint32_t PUPD;       // 上拉/下拉配置
  volatile uint32_t IRQ_EN;     // 中断使能
  volatile uint32_t IRQ_STATUS; // 中断状态
} GPIO_TypeDef;

#define GPIO ((GPIO_TypeDef *)GPIO_BASE)

/* ========================================================================= */
/* UART 定义                                                                 */
/* ========================================================================= */

typedef struct {
  volatile uint32_t DATA;     // 数据寄存器
  volatile uint32_t STATUS;   // 状态寄存器
  volatile uint32_t CTRL;     // 控制寄存器
  volatile uint32_t BAUDRATE; // 波特率分频
  volatile uint32_t IRQ_EN;   // 中断使能
} UART_TypeDef;

#define UART ((UART_TypeDef *)UART_BASE)

/* UART 状态位 */
#define UART_STATUS_TXE (1U << 0)  // 发送空
#define UART_STATUS_RXNE (1U << 1) // 接收非空
#define UART_STATUS_TC (1U << 2)   // 传输完成

/* ========================================================================= */
/* DMA 控制器                                                                */
/* ========================================================================= */

typedef struct {
  volatile uint32_t SRC_ADDR; // 源地址
  volatile uint32_t DST_ADDR; // 目标地址
  volatile uint32_t COUNT;    // 传输计数
  volatile uint32_t CTRL;     // 控制寄存器
  volatile uint32_t STATUS;   // 状态寄存器
} DMA_Channel_TypeDef;

typedef struct {
  DMA_Channel_TypeDef CH[8]; // 8 个 DMA 通道
  volatile uint32_t GLOBAL_STATUS;
  volatile uint32_t GLOBAL_CTRL;
} DMA_TypeDef;

#define DMA ((DMA_TypeDef *)DMA_BASE)

/* DMA 控制位 */
#define DMA_CTRL_ENABLE (1U << 0)
#define DMA_CTRL_MEM2MEM (1U << 1)
#define DMA_CTRL_MEM2PERIPH (1U << 2)
#define DMA_CTRL_PERIPH2MEM (1U << 3)
#define DMA_CTRL_CIRCULAR (1U << 4)
#define DMA_CTRL_IRQ_EN (1U << 5)

/* ========================================================================= */
/* 定时器                                                                    */
/* ========================================================================= */

typedef struct {
  volatile uint32_t CTRL;   // 控制寄存器
  volatile uint32_t COUNT;  // 计数值
  volatile uint32_t RELOAD; // 重载值
  volatile uint32_t STATUS; // 状态寄存器
  volatile uint32_t IRQ_EN; // 中断使能
} TIMER_TypeDef;

#define TIMER ((TIMER_TypeDef *)TIMER_BASE)

/* ========================================================================= */
/* 中断号定义                                                                */
/* ========================================================================= */

typedef enum {
  IRQ_SOFTWARE = 3,  // 软件中断
  IRQ_TIMER = 7,     // 定时器中断
  IRQ_EXTERNAL = 11, // 外部中断
  IRQ_CAN = 16,      // CAN 中断
  IRQ_CIM = 17,      // CIM 中断
  IRQ_DMA = 18,      // DMA 中断
  IRQ_UART = 19,     // UART 中断
  IRQ_SPI = 20,      // SPI 中断
  IRQ_GPIO = 21,     // GPIO 中断
} IRQn_Type;

/* ========================================================================= */
/* 系统函数声明                                                              */
/* ========================================================================= */

/**
 * @brief 系统初始化
 */
void System_Init(void);

/**
 * @brief 获取系统时钟（毫秒）
 */
uint32_t Millis(void);

/**
 * @brief 延迟（毫秒）
 */
void Delay_ms(uint32_t ms);

/**
 * @brief 延迟（微秒）
 */
void Delay_us(uint32_t us);

/**
 * @brief 获取 CPU 周期计数
 */
uint32_t GetCycleCount(void);

/**
 * @brief 使能全局中断
 */
static inline void __enable_irq(void) {
  __asm__ volatile("csrsi mstatus, 0x8");
}

/**
 * @brief 禁用全局中断
 */
static inline void __disable_irq(void) {
  __asm__ volatile("csrci mstatus, 0x8");
}

/**
 * @brief 内存屏障
 */
static inline void __dmb(void) { __asm__ volatile("fence"); }

/**
 * @brief 数据同步屏障
 */
static inline void __dsb(void) { __asm__ volatile("fence"); }

/**
 * @brief 指令同步屏障
 */
static inline void __isb(void) { __asm__ volatile("fence.i"); }

/**
 * @brief 空操作
 */
static inline void __nop(void) { __asm__ volatile("nop"); }

/**
 * @brief 等待中断
 */
static inline void __wfi(void) { __asm__ volatile("wfi"); }

#ifdef __cplusplus
}
#endif

#endif /* IMC22_H */
