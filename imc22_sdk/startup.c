/**
 * @file startup.c
 * @brief RISC-V 启动代码和中断向量表
 * @version 2.0 (更新)
 */

#include "imc22.h"
#include "riscv_custom.h"
#include <stdint.h>

/* 外部符号 (来自链接脚本) */
extern uint32_t _stext, _etext;
extern uint32_t _sdata, _edata, _sidata;
extern uint32_t _sbss, _ebss;
extern uint32_t _stack_top;

/* 主函数 */
extern int main(void);

/* ========================================================================= */
/* 启动代码                                                                  */
/* ========================================================================= */

void _start(void) __attribute__((naked, section(".text.startup")));

void _start(void) {
  // 1. 设置栈指针
  __asm__ volatile("la sp, _stack_top\n");

  // 2. 初始化 .data 段 (从 FLASH 复制到 SRAM)
  uint32_t *src = &_sidata;
  uint32_t *dst = &_sdata;
  while (dst < &_edata) {
    *dst++ = *src++;
  }

  // 3. 清零 .bss 段
  dst = &_sbss;
  while (dst < &_ebss) {
    *dst++ = 0;
  }

// 4. 初始化 FPU (如果有)
#ifdef __riscv_flen
  uint32_t mstatus = riscv_csr_read(CSR_MSTATUS);
  mstatus |= (1 << 13); // FS = Initial
  riscv_csr_write(CSR_MSTATUS, mstatus);
#endif

  // 5. 设置中断向量表
  extern void trap_vector(void);
  riscv_csr_write(CSR_MTVEC, (uint32_t)trap_vector);

  // 6. 使能全局中断
  riscv_csr_set(CSR_MSTATUS, MSTATUS_MIE);

// 7. 调用全局构造函数 (C++)
#ifdef __cplusplus
  extern void (*__init_array_start[])(void);
  extern void (*__init_array_end[])(void);
  for (void (**p)(void) = __init_array_start; p < __init_array_end; p++) {
    (*p)();
  }
#endif

  // 8. 跳转到 main
  main();

  // 9. 如果 main 返回，进入死循环
  while (1) {
    riscv_wfi();
  }
}

/* ========================================================================= */
/* 中断和异常处理                                                            */
/* ========================================================================= */

void Default_Handler(void) __attribute__((weak));
void Default_Handler(void) {
  while (1)
    ;
}

/* 异常处理函数 */
void InstructionMisaligned_Handler(void)
    __attribute__((weak, alias("Default_Handler")));
void InstructionFault_Handler(void)
    __attribute__((weak, alias("Default_Handler")));
void IllegalInstruction_Handler(void)
    __attribute__((weak, alias("Default_Handler")));
void Breakpoint_Handler(void) __attribute__((weak, alias("Default_Handler")));
void LoadMisaligned_Handler(void)
    __attribute__((weak, alias("Default_Handler")));
void LoadFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void StoreMisaligned_Handler(void)
    __attribute__((weak, alias("Default_Handler")));
void StoreFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void ECall_Handler(void) __attribute__((weak, alias("Default_Handler")));

/* 中断处理函数 */
void Timer_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void External_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void Software_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void CAN_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void CIM_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));

/**
 * @brief 统一的中断/异常入口
 */
void trap_vector(void) __attribute__((naked, aligned(4)));
void trap_vector(void) {
  // 保存上下文
  __asm__ volatile("addi sp, sp, -128\n"
                   "sw x1,   0(sp)\n"
                   "sw x5,   4(sp)\n"
                   "sw x6,   8(sp)\n"
                   "sw x7,  12(sp)\n"
                   "sw x10, 16(sp)\n"
                   "sw x11, 20(sp)\n"
                   "sw x12, 24(sp)\n"
                   "sw x13, 28(sp)\n"
                   "sw x14, 32(sp)\n"
                   "sw x15, 36(sp)\n"
                   "sw x16, 40(sp)\n"
                   "sw x17, 44(sp)\n"
                   "sw x28, 48(sp)\n"
                   "sw x29, 52(sp)\n"
                   "sw x30, 56(sp)\n"
                   "sw x31, 60(sp)\n");

  // 读取 mcause
  uint32_t mcause = riscv_csr_read(CSR_MCAUSE);
  uint32_t is_interrupt = mcause & 0x80000000;
  uint32_t code = mcause & 0x7FFFFFFF;

  if (is_interrupt) {
    // 中断
    switch (code) {
    case 3:
      Software_IRQHandler();
      break;
    case 7:
      Timer_IRQHandler();
      break;
    case 11:
      External_IRQHandler();
      break;
    case 16:
      CAN_IRQHandler();
      break;
    case 17:
      CIM_IRQHandler();
      break;
    default:
      Default_Handler();
      break;
    }
  } else {
    // 异常
    switch (code) {
    case 0:
      InstructionMisaligned_Handler();
      break;
    case 1:
      InstructionFault_Handler();
      break;
    case 2:
      IllegalInstruction_Handler();
      break;
    case 3:
      Breakpoint_Handler();
      break;
    case 4:
      LoadMisaligned_Handler();
      break;
    case 5:
      LoadFault_Handler();
      break;
    case 6:
      StoreMisaligned_Handler();
      break;
    case 7:
      StoreFault_Handler();
      break;
    case 11:
      ECall_Handler();
      break;
    default:
      Default_Handler();
      break;
    }
  }

  // 恢复上下文
  __asm__ volatile("lw x1,   0(sp)\n"
                   "lw x5,   4(sp)\n"
                   "lw x6,   8(sp)\n"
                   "lw x7,  12(sp)\n"
                   "lw x10, 16(sp)\n"
                   "lw x11, 20(sp)\n"
                   "lw x12, 24(sp)\n"
                   "lw x13, 28(sp)\n"
                   "lw x14, 32(sp)\n"
                   "lw x15, 36(sp)\n"
                   "lw x16, 40(sp)\n"
                   "lw x17, 44(sp)\n"
                   "lw x28, 48(sp)\n"
                   "lw x29, 52(sp)\n"
                   "lw x30, 56(sp)\n"
                   "lw x31, 60(sp)\n"
                   "addi sp, sp, 128\n"
                   "mret\n");
}

/* ========================================================================= */
/* 系统初始化                                                                */
/* ========================================================================= */

void System_Init(void) {
  // 在这里执行早期硬件初始化
  // (时钟、GPIO、UART 等)

  // 配置系统时钟 (假设外部晶振 25MHz, PLL 到 100MHz)
  // ... (具体取决于硬件)
}

/* ========================================================================= */
/* 工具函数                                                                  */
/* ========================================================================= */

static volatile uint32_t g_system_ticks = 0;

void SysTick_Handler(void) __attribute__((weak));
void SysTick_Handler(void) { g_system_ticks++; }

uint32_t Millis(void) { return g_system_ticks; }

void Delay_ms(uint32_t ms) {
  uint32_t start = g_system_ticks;
  while (g_system_ticks - start < ms)
    ;
}

void Delay_us(uint32_t us) {
  uint64_t start = riscv_read_cycle();
  uint64_t delay_cycles = ((uint64_t)us * IMC22_SYSCLK_HZ) / 1000000;

  while (riscv_read_cycle() - start < delay_cycles)
    ;
}

uint32_t GetCycleCount(void) { return (uint32_t)riscv_read_cycle(); }
