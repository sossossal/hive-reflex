/**
 * @file imc22_flash.h
 * @brief NOR FLASH 控制器驱动 (QSPI 接口)
 * @version 2.0
 * @date 2026-01-19
 *
 * 支持 XIP (Execute-in-Place) 和直接读写
 */

#ifndef IMC22_FLASH_H
#define IMC22_FLASH_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 硬件配置                                                                  */
/* ========================================================================= */

#define FLASH_BASE_ADDR 0x08000000   // FLASH 映射地址 (XIP)
#define FLASH_CTRL_BASE 0x40020000   // FLASH 控制器基地址
#define FLASH_SIZE (2 * 1024 * 1024) // 2MB
#define FLASH_PAGE_SIZE 256          // 页大小
#define FLASH_SECTOR_SIZE 4096       // 扇区大小 (4KB)
#define FLASH_BLOCK_SIZE 65536       // 块大小 (64KB)

/* ========================================================================= */
/* 寄存器定义                                                                */
/* ========================================================================= */

typedef struct {
  volatile uint32_t CTRL;       // 控制寄存器
  volatile uint32_t STATUS;     // 状态寄存器
  volatile uint32_t CMD;        // 命令寄存器
  volatile uint32_t ADDR;       // 地址寄存器
  volatile uint32_t DATA;       // 数据寄存器
  volatile uint32_t TIMING;     // 时序配置
  volatile uint32_t XIP_CTRL;   // XIP 控制
  volatile uint32_t IRQ_STATUS; // 中断状态
} FLASH_CTRL_TypeDef;

#define FLASH_CTRL ((FLASH_CTRL_TypeDef *)FLASH_CTRL_BASE)

/* CTRL 寄存器位定义 */
#define FLASH_CTRL_ENABLE (1U << 0)     // 使能 FLASH 控制器
#define FLASH_CTRL_QSPI_MODE (1U << 1)  // QSPI 模式
#define FLASH_CTRL_XIP_ENABLE (1U << 2) // 使能 XIP
#define FLASH_CTRL_AUTO_POLL (1U << 3)  // 自动轮询

/* STATUS 寄存器位定义 */
#define FLASH_STATUS_BUSY (1U << 0)  // 忙碌
#define FLASH_STATUS_WEL (1U << 1)   // 写使能锁存
#define FLASH_STATUS_WIP (1U << 2)   // 写入进行中
#define FLASH_STATUS_ERROR (1U << 7) // 错误

/* FLASH 命令 */
#define FLASH_CMD_WRITE_ENABLE 0x06  // 写使能
#define FLASH_CMD_WRITE_DISABLE 0x04 // 写禁用
#define FLASH_CMD_READ_STATUS 0x05   // 读状态寄存器
#define FLASH_CMD_WRITE_STATUS 0x01  // 写状态寄存器
#define FLASH_CMD_READ_DATA 0x03     // 读数据
#define FLASH_CMD_FAST_READ 0x0B     // 快速读
#define FLASH_CMD_QUAD_READ 0xEB     // 四线快速读
#define FLASH_CMD_PAGE_PROGRAM 0x02  // 页编程
#define FLASH_CMD_SECTOR_ERASE 0x20  // 扇区擦除 (4KB)
#define FLASH_CMD_BLOCK_ERASE 0xD8   // 块擦除 (64KB)
#define FLASH_CMD_CHIP_ERASE 0xC7    // 芯片擦除
#define FLASH_CMD_RESET 0xFF         // 复位

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化 FLASH 控制器
 * @param enable_xip 是否使能 XIP 模式
 * @return 0 成功, -1 失败
 */
int FLASH_Init(bool enable_xip);

/**
 * @brief 读取数据
 * @param addr 地址 (相对于 FLASH 起始)
 * @param buf 缓冲区
 * @param size 读取大小
 * @return 实际读取字节数, -1 表示失败
 */
int FLASH_Read(uint32_t addr, void *buf, uint32_t size);

/**
 * @brief 写入数据 (自动处理页边界)
 * @param addr 地址
 * @param data 数据
 * @param size 写入大小
 * @return 实际写入字节数, -1 表示失败
 */
int FLASH_Write(uint32_t addr, const void *data, uint32_t size);

/**
 * @brief 擦除扇区 (4KB)
 * @param addr 扇区起始地址 (必须 4KB 对齐)
 * @return 0 成功, -1 失败
 */
int FLASH_EraseSector(uint32_t addr);

/**
 * @brief 擦除块 (64KB)
 * @param addr 块起始地址 (必须 64KB 对齐)
 * @return 0 成功, -1 失败
 */
int FLASH_EraseBlock(uint32_t addr);

/**
 * @brief 擦除整个芯片
 * @return 0 成功, -1 失败
 */
int FLASH_EraseChip(void);

/**
 * @brief 等待 FLASH 就绪
 * @param timeout_ms 超时时间 (ms)
 * @return 0 成功, -1 超时
 */
int FLASH_WaitReady(uint32_t timeout_ms);

/**
 * @brief 使能 XIP 模式
 */
void FLASH_EnableXIP(void);

/**
 * @brief 禁用 XIP 模式
 */
void FLASH_DisableXIP(void);

/**
 * @brief 获取 FLASH 信息
 */
typedef struct {
  uint32_t manufacturer_id; /**< 制造商 ID */
  uint32_t device_id;       /**< 器件 ID */
  uint32_t capacity;        /**< 容量 (bytes) */
  uint32_t page_size;       /**< 页大小 */
  uint32_t sector_size;     /**< 扇区大小 */
} FLASH_Info_t;

/**
 * @brief 读取 FLASH 信息
 * @param info 信息结构体指针
 * @return 0 成功, -1 失败
 */
int FLASH_GetInfo(FLASH_Info_t *info);

/* ========================================================================= */
/* 内部函数 (供 NVS 使用)                                                    */
/* ========================================================================= */

/**
 * @brief 发送命令
 * @param cmd 命令字节
 * @return 0 成功, -1 失败
 */
int _FLASH_SendCommand(uint8_t cmd);

/**
 * @brief 发送命令 + 地址
 * @param cmd 命令字节
 * @param addr 24-bit 地址
 * @return 0 成功, -1 失败
 */
int _FLASH_SendCommandAddr(uint8_t cmd, uint32_t addr);

/**
 * @brief 读取状态寄存器
 * @return 状态值, -1 表示失败
 */
int _FLASH_ReadStatus(void);

/**
 * @brief 写使能
 * @return 0 成功, -1 失败
 */
int _FLASH_WriteEnable(void);

/**
 * @brief 页编程 (内部使用, 不检查边界)
 * @param addr 地址 (必须页对齐)
 * @param data 数据
 * @param size 大小 (不超过页大小)
 * @return 0 成功, -1 失败
 */
int _FLASH_PageProgram(uint32_t addr, const void *data, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_FLASH_H */
