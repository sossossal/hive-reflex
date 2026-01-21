/**
 * @file imc22_can.c
 * @brief IMC-22 CAN-FD 驱动实现
 */

#include "imc22_can.h"

/* ========== 内部函数 ========== */

/**
 * @brief 计算波特率分频值
 */
static uint32_t CalculateBaudDiv(uint32_t baudrate) {
  // 假设 CAN 时钟为系统时钟的 1/4
  uint32_t can_clk = IMC22_SYSCLK_HZ / 4;

  // CAN 位时间 = 1 + TSEG1 + TSEG2 (通常为 16 个时钟周期)
  uint32_t tq_per_bit = 16;

  return (can_clk / baudrate / tq_per_bit) - 1;
}

/* ========== 公共函数 ========== */

int CAN_Init(const CAN_Config_t *config) {
  // 禁用 CAN
  CAN->CTRL = 0;

  // 配置波特率
  CAN->BAUD = CalculateBaudDiv(config->baudrate);

  // 配置控制寄存器
  uint32_t ctrl = CAN_CTRL_EN;
  if (config->fd_mode) {
    ctrl |= CAN_CTRL_FD;
  }
  if (config->loopback) {
    ctrl |= CAN_CTRL_LOOPBACK;
  }
  CAN->CTRL = ctrl;

  // 清除所有中断标志
  CAN->IF = 0xFFFFFFFF;

  // 等待 CAN 初始化完成 (假设需要等待状态位)
  uint32_t timeout = 10000;
  while ((CAN->STATUS & CAN_STATUS_BUSOFF) && timeout--) {
    DelayUs(10);
  }

  return (timeout > 0) ? 0 : -1;
}

int CAN_Send(const CAN_Message_t *msg) {
  // 等待发送缓冲区空闲
  uint32_t timeout = 10000;
  while (!(CAN->STATUS & CAN_STATUS_TXOK) && timeout--) {
    DelayUs(1);
  }

  if (timeout == 0) {
    return -1; // 超时
  }

  // 写入 ID (标准帧 11-bit, 扩展帧 29-bit)
  CAN->TX_ID = msg->id | (msg->is_extended ? (1 << 31) : 0);

  // 写入数据长度
  CAN->TX_DLC = msg->dlc;

  // 写入数据 (分两个 32-bit 寄存器)
  CAN->TX_DATA[0] =
      ((uint32_t)msg->data[0] << 0) | ((uint32_t)msg->data[1] << 8) |
      ((uint32_t)msg->data[2] << 16) | ((uint32_t)msg->data[3] << 24);

  CAN->TX_DATA[1] =
      ((uint32_t)msg->data[4] << 0) | ((uint32_t)msg->data[5] << 8) |
      ((uint32_t)msg->data[6] << 16) | ((uint32_t)msg->data[7] << 24);

  // 触发发送 (写入控制寄存器的发送请求位)
  CAN->CTRL |= (1 << 8); // 假设 bit 8 是发送请求

  return 0;
}

int CAN_Receive(CAN_Message_t *msg) {
  // 检查是否有数据
  if (!(CAN->STATUS & CAN_STATUS_RXNE)) {
    return -1; // 无数据
  }

  // 读取 ID
  uint32_t rx_id = CAN->RX_ID;
  msg->is_extended = (rx_id & (1 << 31)) ? true : false;
  msg->id = rx_id & (msg->is_extended ? 0x1FFFFFFF : 0x7FF);

  // 读取数据长度
  msg->dlc = CAN->RX_DLC & 0xF;

  // 读取数据
  uint32_t data_low = CAN->RX_DATA[0];
  uint32_t data_high = CAN->RX_DATA[1];

  msg->data[0] = (data_low >> 0) & 0xFF;
  msg->data[1] = (data_low >> 8) & 0xFF;
  msg->data[2] = (data_low >> 16) & 0xFF;
  msg->data[3] = (data_low >> 24) & 0xFF;
  msg->data[4] = (data_high >> 0) & 0xFF;
  msg->data[5] = (data_high >> 8) & 0xFF;
  msg->data[6] = (data_high >> 16) & 0xFF;
  msg->data[7] = (data_high >> 24) & 0xFF;

  // 清除接收标志
  CAN->IF |= CAN_STATUS_RXNE;

  return 0;
}

void CAN_SetFilter(uint8_t filter_idx, uint32_t id, uint32_t mask) {
  if (filter_idx < 8) {
    // 高 16 位为掩码，低 16 位为 ID
    CAN->FILTER[filter_idx] = (mask << 16) | (id & 0xFFFF);
  }
}

void CAN_EnableRxInterrupt(bool enable) {
  if (enable) {
    CAN->IE |= CAN_STATUS_RXNE;
    NVIC_EnableIRQ(IRQ_CAN);
  } else {
    CAN->IE &= ~CAN_STATUS_RXNE;
    NVIC_DisableIRQ(IRQ_CAN);
  }
}

/* ========== 中断处理 ========== */

/**
 * @brief CAN 中断服务程序
 */
void __attribute__((interrupt)) CAN_IRQHandler(void) {
  if (CAN->IF & CAN_STATUS_RXNE) {
    CAN_Message_t msg;
    if (CAN_Receive(&msg) == 0) {
      // 调用用户回调
      extern void CAN_RxCallback(CAN_Message_t * msg);
      CAN_RxCallback(&msg);
    }
  }
}
