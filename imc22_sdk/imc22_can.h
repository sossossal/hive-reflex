/**
 * @file imc22_can.h
 * @brief IMC-22 CAN-FD 驱动接口
 */

#ifndef IMC22_CAN_H
#define IMC22_CAN_H

#include "imc22.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== CAN 寄存器结构 ========== */
typedef struct {
  vuint32_t CTRL;       // 控制寄存器
  vuint32_t STATUS;     // 状态寄存器
  vuint32_t BAUD;       // 波特率配置
  vuint32_t IE;         // 中断使能
  vuint32_t IF;         // 中断标志
  vuint32_t TX_ID;      // 发送 ID
  vuint32_t TX_DLC;     // 发送数据长度
  vuint32_t TX_DATA[2]; // 发送数据 (64 bits)
  vuint32_t RX_ID;      // 接收 ID
  vuint32_t RX_DLC;     // 接收数据长度
  vuint32_t RX_DATA[2]; // 接收数据 (64 bits)
  vuint32_t FILTER[8];  // 接收过滤器
} CAN_TypeDef;

#define CAN ((CAN_TypeDef *)CAN_BASE)

/* CAN 控制位 */
#define CAN_CTRL_EN (1 << 0)       // CAN 使能
#define CAN_CTRL_FD (1 << 1)       // CAN-FD 模式
#define CAN_CTRL_LOOPBACK (1 << 2) // 回环测试模式

/* CAN 状态位 */
#define CAN_STATUS_TXOK (1 << 0)   // 发送完成
#define CAN_STATUS_RXNE (1 << 1)   // 接收非空
#define CAN_STATUS_ERROR (1 << 2)  // 错误标志
#define CAN_STATUS_BUSOFF (1 << 3) // 总线关闭

/* ========== CAN 消息结构 ========== */
typedef struct {
  uint32_t id;      // CAN ID (11-bit 标准帧 或 29-bit 扩展帧)
  uint8_t dlc;      // 数据长度 (0-8)
  uint8_t data[8];  // 数据负载
  bool is_extended; // 是否为扩展帧
  bool is_fd;       // 是否为 CAN-FD
} CAN_Message_t;

/* ========== CAN 初始化配置 ========== */
typedef struct {
  uint32_t baudrate; // 波特率 (例如: 1000000 = 1Mbps)
  bool fd_mode;      // 是否启用 CAN-FD
  bool loopback;     // 回环测试模式
} CAN_Config_t;

/* ========== 函数声明 ========== */

/**
 * @brief 初始化 CAN 控制器
 * @param config 配置参数
 * @return 0=成功, -1=失败
 */
int CAN_Init(const CAN_Config_t *config);

/**
 * @brief 发送 CAN 消息 (阻塞)
 * @param msg 消息指针
 * @return 0=成功, -1=超时
 */
int CAN_Send(const CAN_Message_t *msg);

/**
 * @brief 接收 CAN 消息 (非阻塞)
 * @param msg 消息缓冲区
 * @return 0=成功, -1=无数据
 */
int CAN_Receive(CAN_Message_t *msg);

/**
 * @brief 设置接收过滤器
 * @param filter_idx 过滤器索引 (0-7)
 * @param id 要接收的 CAN ID
 * @param mask 掩码 (0=不关心)
 */
void CAN_SetFilter(uint8_t filter_idx, uint32_t id, uint32_t mask);

/**
 * @brief 启用/禁用 CAN 接收中断
 */
void CAN_EnableRxInterrupt(bool enable);

/**
 * @brief CAN 中断处理回调 (需要用户实现)
 */
extern void CAN_RxCallback(CAN_Message_t *msg);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_CAN_H */
