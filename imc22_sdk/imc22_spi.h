/**
 * @file imc22_spi.h
 * @brief IMC-22 SPI 驱动接口
 */

#ifndef IMC22_SPI_H
#define IMC22_SPI_H

#include "imc22.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== SPI 寄存器结构 ========== */
typedef struct {
  vuint32_t CTRL;   // 控制寄存器
  vuint32_t STATUS; // 状态寄存器
  vuint32_t DATA;   // 数据寄存器
  vuint32_t BAUD;   // 波特率分频
  vuint32_t CS;     // 片选控制
} SPI_TypeDef;

#define SPI0 ((SPI_TypeDef *)SPI0_BASE)
#define SPI1 ((SPI_TypeDef *)SPI1_BASE)

/* SPI 控制位 */
#define SPI_CTRL_EN (1 << 0)     // SPI 使能
#define SPI_CTRL_MASTER (1 << 1) // 主机模式
#define SPI_CTRL_CPOL (1 << 2)   // 时钟极性
#define SPI_CTRL_CPHA (1 << 3)   // 时钟相位
#define SPI_CTRL_DMA_EN (1 << 4) // DMA 使能

/* SPI 状态位 */
#define SPI_STATUS_TXNE (1 << 0) // 发送非空
#define SPI_STATUS_RXNE (1 << 1) // 接收非空
#define SPI_STATUS_BUSY (1 << 2) // 忙碌

/* ========== SPI 配置 ========== */
typedef struct {
  uint32_t baudrate; // 波特率 (Hz)
  uint8_t cpol;      // 时钟极性 (0/1)
  uint8_t cpha;      // 时钟相位 (0/1)
  bool use_dma;      // 是否使用 DMA
} SPI_Config_t;

/* ========== 函数声明 ========== */

int SPI_Init(SPI_TypeDef *spi, const SPI_Config_t *config);
uint8_t SPI_TransferByte(SPI_TypeDef *spi, uint8_t data);
int SPI_Transfer(SPI_TypeDef *spi, const uint8_t *tx_buf, uint8_t *rx_buf,
                 uint32_t len);
void SPI_SetCS(SPI_TypeDef *spi, bool active);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_SPI_H */
