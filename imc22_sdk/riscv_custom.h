/**
 * @file riscv_custom.h
 * @brief RISC-V 自定义指令扩展 (CIM 专用)
 * @version 2.0
 * @date 2026-01-19
 *
 * 定义与 CIM 硬件加速器紧密集成的自定义 RISC-V 指令
 */

#ifndef RISCV_CUSTOM_H
#define RISCV_CUSTOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 自定义指令编码 (使用 RISC-V Custom-0 操作码空间)                          */
/* ========================================================================= */

/*
 * RISC-V 自定义指令格式:
 *
 * custom-0: opcode = 0x0B (0b0001011)
 * custom-1: opcode = 0x2B (0b0101011)
 *
 * 我们使用 custom-0，格式为:
 *
 *  31        25 24  20 19  15 14  12 11   7 6     0
 * ┌───────────┬──────┬──────┬──────┬──────┬───────┐
 * │   funct7  │  rs2 │  rs1 │funct3│  rd  │0001011│
 * └───────────┴──────┴──────┴──────┴──────┴───────┘
 */

/* CIM 矩阵乘法指令 */
#define CIM_MATMUL_FUNCT7 0x01
#define CIM_MATMUL_FUNCT3 0x00

/* CIM LSTM 指令 */
#define CIM_LSTM_FUNCT7 0x02
#define CIM_LSTM_FUNCT3 0x00

/* CIM 等待完成指令 */
#define CIM_WAIT_FUNCT7 0x03
#define CIM_WAIT_FUNCT3 0x00

/* CIM 状态读取指令 */
#define CIM_STATUS_FUNCT7 0x04
#define CIM_STATUS_FUNCT3 0x00

/* ========================================================================= */
/* 内联汇编宏定义                                                            */
/* ========================================================================= */

/**
 * @brief CIM 矩阵乘法指令
 * @param a 输入矩阵 A 地址
 * @param b 权重矩阵 B 地址
 * @param c 输出矩阵 C 地址
 *
 * 编码: .insn r 0x0B, 0, 0x01, rd, rs1, rs2
 */
static inline void cim_matmul(void *c, const void *a, const void *b) {
  __asm__ volatile(".insn r 0x0B, 0x0, 0x01, %0, %1, %2"
                   : "=r"(c)
                   : "r"(a), "r"(b)
                   : "memory");
}

/**
 * @brief CIM LSTM 单元指令
 * @param output 输出地址
 * @param input 输入地址
 * @param state LSTM 状态地址
 */
static inline void cim_lstm(void *output, const void *input, void *state) {
  __asm__ volatile(".insn r 0x0B, 0x0, 0x02, %0, %1, %2"
                   : "=r"(output)
                   : "r"(input), "r"(state)
                   : "memory");
}

/**
 * @brief CIM 等待完成指令
 * 阻塞直到 CIM 计算完成
 */
static inline void cim_wait(void) {
  register uint32_t zero = 0;
  __asm__ volatile(".insn r 0x0B, 0x0, 0x03, zero, zero, zero"
                   :
                   : "r"(zero)
                   : "memory");
}

/**
 * @brief CIM 状态读取指令
 * @return CIM 状态寄存器值
 */
static inline uint32_t cim_status(void) {
  register uint32_t status;
  __asm__ volatile(".insn r 0x0B, 0x0, 0x04, %0, zero, zero"
                   : "=r"(status)
                   :
                   : "memory");
  return status;
}

/* ========================================================================= */
/* 高级封装 API                                                              */
/* ========================================================================= */

/**
 * @brief 快速矩阵乘法 (使用自定义指令)
 * @param A 矩阵 A (M x K)
 * @param B 矩阵 B (K x N)
 * @param C 输出矩阵 C (M x N)
 * @param M 行数
 * @param N 列数
 * @param K 中间维度
 */
static inline void riscv_fast_matmul(const float *A, const float *B, float *C,
                                     uint32_t M, uint32_t N, uint32_t K) {
  // 配置 CIM 寄存器 (通过内存映射)
  volatile uint32_t *cim_ctrl = (volatile uint32_t *)0x50000000;
  cim_ctrl[1] = M; // DIM_M
  cim_ctrl[2] = N; // DIM_N
  cim_ctrl[3] = K; // DIM_K

  // 执行自定义指令
  cim_matmul(C, A, B);

  // 等待完成
  cim_wait();
}

/**
 * @brief 循环计数器读取 (标准 RISC-V 指令)
 */
static inline uint64_t riscv_read_cycle(void) {
  uint32_t lo, hi, hi2;

  __asm__ volatile("1: rdcycleh %0\n"
                   "   rdcycle %1\n"
                   "   rdcycleh %2\n"
                   "   bne %0, %2, 1b"
                   : "=r"(hi), "=r"(lo), "=r"(hi2)
                   :
                   : "memory");

  return ((uint64_t)hi << 32) | lo;
}

/**
 * @brief 时间计数器读取
 */
static inline uint64_t riscv_read_time(void) {
  uint32_t lo, hi, hi2;

  __asm__ volatile("1: rdtimeh %0\n"
                   "   rdtime %1\n"
                   "   rdtimeh %2\n"
                   "   bne %0, %2, 1b"
                   : "=r"(hi), "=r"(lo), "=r"(hi2)
                   :
                   : "memory");

  return ((uint64_t)hi << 32) | lo;
}

/**
 * @brief 内存屏障
 */
static inline void riscv_fence(void) { __asm__ volatile("fence" ::: "memory"); }

/**
 * @brief 指令屏障
 */
static inline void riscv_fence_i(void) {
  __asm__ volatile("fence.i" ::: "memory");
}

/**
 * @brief 等待中断 (低功耗模式)
 */
static inline void riscv_wfi(void) { __asm__ volatile("wfi" ::: "memory"); }

/* ========================================================================= */
/* CSR (控制和状态寄存器) 访问                                               */
/* ========================================================================= */

/**
 * @brief 读取 CSR
 */
#define riscv_csr_read(csr)                                                    \
  ({                                                                           \
    register unsigned long __v;                                                \
    __asm__ volatile("csrr %0, " #csr : "=r"(__v) : : "memory");               \
    __v;                                                                       \
  })

/**
 * @brief 写入 CSR
 */
#define riscv_csr_write(csr, val)                                              \
  ({ __asm__ volatile("csrw " #csr ", %0" : : "rK"(val) : "memory"); })

/**
 * @brief CSR 设置位
 */
#define riscv_csr_set(csr, val)                                                \
  ({ __asm__ volatile("csrs " #csr ", %0" : : "rK"(val) : "memory"); })

/**
 * @brief CSR 清除位
 */
#define riscv_csr_clear(csr, val)                                              \
  ({ __asm__ volatile("csrc " #csr ", %0" : : "rK"(val) : "memory"); })

/* 常用 CSR 定义 */
#define CSR_MSTATUS 0x300
#define CSR_MISA 0x301
#define CSR_MIE 0x304
#define CSR_MTVEC 0x305
#define CSR_MCAUSE 0x342
#define CSR_MEPC 0x341
#define CSR_MTVAL 0x343

/* MSTATUS 位定义 */
#define MSTATUS_MIE (1 << 3)  // 机器模式中断使能
#define MSTATUS_MPIE (1 << 7) // 先前中断使能

#ifdef __cplusplus
}
#endif

#endif /* RISCV_CUSTOM_H */
