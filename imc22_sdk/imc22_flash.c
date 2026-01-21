/**
 * @file imc22_flash.c
 * @brief NOR FLASH 控制器驱动实现
 */

#include "imc22_flash.h"
#include "imc22.h"
#include <string.h>

/* 全局状态 */
static bool g_flash_initialized = false;
static bool g_xip_enabled = false;

/* ========================================================================= */
/* 内部辅助函数                                                              */
/* ========================================================================= */

static void _FLASH_Delay_us(uint32_t us) {
  // 简单的延迟函数
  volatile uint32_t count = us * (IMC22_SYSCLK_HZ / 1000000 / 4);
  while (count--)
    ;
}

/* ========================================================================= */
/* 公共 API 实现                                                             */
/* ========================================================================= */

int FLASH_Init(bool enable_xip) {
  if (g_flash_initialized) {
    return 0; // 已初始化
  }

  // 使能 FLASH 控制器
  FLASH_CTRL->CTRL = FLASH_CTRL_ENABLE | FLASH_CTRL_QSPI_MODE;

  // 配置时序 (假设 100MHz 系统时钟, FLASH 支持 104MHz)
  FLASH_CTRL->TIMING = 0x00000001; // 简化配置

  // 复位 FLASH
  _FLASH_SendCommand(FLASH_CMD_RESET);
  _FLASH_Delay_us(100);

  // 等待就绪
  if (FLASH_WaitReady(100) != 0) {
    return -1;
  }

  // 使能 XIP (如果需要)
  if (enable_xip) {
    FLASH_EnableXIP();
  }

  g_flash_initialized = true;
  return 0;
}

int FLASH_Read(uint32_t addr, void *buf, uint32_t size) {
  if (!g_flash_initialized || !buf || size == 0) {
    return -1;
  }

  // 地址检查
  if (addr + size > FLASH_SIZE) {
    return -1;
  }

  // 如果 XIP 使能, 直接从映射地址读取
  if (g_xip_enabled) {
    memcpy(buf, (void *)(FLASH_BASE_ADDR + addr), size);
    return size;
  }

  // 否则使用 QSPI 读取
  uint8_t *ptr = (uint8_t *)buf;
  uint32_t remaining = size;
  uint32_t current_addr = addr;

  while (remaining > 0) {
    // 发送读命令 + 地址
    _FLASH_SendCommandAddr(FLASH_CMD_FAST_READ, current_addr);

    // 读取数据 (一次最多 256 字节)
    uint32_t chunk = (remaining > 256) ? 256 : remaining;

    for (uint32_t i = 0; i < chunk; i++) {
      while (!(FLASH_CTRL->STATUS & FLASH_STATUS_BUSY))
        ;
      *ptr++ = (uint8_t)FLASH_CTRL->DATA;
    }

    current_addr += chunk;
    remaining -= chunk;
  }

  return size;
}

int FLASH_Write(uint32_t addr, const void *data, uint32_t size) {
  if (!g_flash_initialized || !data || size == 0) {
    return -1;
  }

  // 地址检查
  if (addr + size > FLASH_SIZE) {
    return -1;
  }

  const uint8_t *ptr = (const uint8_t *)data;
  uint32_t remaining = size;
  uint32_t current_addr = addr;

  // 处理页边界
  while (remaining > 0) {
    // 计算当前页可写入的字节数
    uint32_t page_offset = current_addr % FLASH_PAGE_SIZE;
    uint32_t page_remaining = FLASH_PAGE_SIZE - page_offset;
    uint32_t chunk = (remaining > page_remaining) ? page_remaining : remaining;

    // 页编程
    if (_FLASH_PageProgram(current_addr, ptr, chunk) != 0) {
      return -1;
    }

    ptr += chunk;
    current_addr += chunk;
    remaining -= chunk;
  }

  return size;
}

int FLASH_EraseSector(uint32_t addr) {
  if (!g_flash_initialized) {
    return -1;
  }

  // 检查对齐
  if (addr % FLASH_SECTOR_SIZE != 0) {
    return -1;
  }

  // 写使能
  if (_FLASH_WriteEnable() != 0) {
    return -1;
  }

  // 发送擦除命令
  _FLASH_SendCommandAddr(FLASH_CMD_SECTOR_ERASE, addr);

  // 等待完成 (扇区擦除约需 50ms)
  return FLASH_WaitReady(100);
}

int FLASH_EraseBlock(uint32_t addr) {
  if (!g_flash_initialized) {
    return -1;
  }

  // 检查对齐
  if (addr % FLASH_BLOCK_SIZE != 0) {
    return -1;
  }

  // 写使能
  if (_FLASH_WriteEnable() != 0) {
    return -1;
  }

  // 发送擦除命令
  _FLASH_SendCommandAddr(FLASH_CMD_BLOCK_ERASE, addr);

  // 等待完成 (块擦除约需 300ms)
  return FLASH_WaitReady(500);
}

int FLASH_EraseChip(void) {
  if (!g_flash_initialized) {
    return -1;
  }

  // 写使能
  if (_FLASH_WriteEnable() != 0) {
    return -1;
  }

  // 发送芯片擦除命令
  _FLASH_SendCommand(FLASH_CMD_CHIP_ERASE);

  // 等待完成 (芯片擦除可能需要数十秒)
  return FLASH_WaitReady(60000); // 60 秒超时
}

int FLASH_WaitReady(uint32_t timeout_ms) {
  uint32_t start = Millis();

  while (1) {
    int status = _FLASH_ReadStatus();
    if (status < 0) {
      return -1;
    }

    // 检查 WIP (写入进行中) 位
    if (!(status & FLASH_STATUS_WIP)) {
      return 0; // 就绪
    }

    // 超时检查
    if (Millis() - start > timeout_ms) {
      return -1; // 超时
    }

    _FLASH_Delay_us(10);
  }
}

void FLASH_EnableXIP(void) {
  // 配置 XIP 模式
  FLASH_CTRL->XIP_CTRL = 0x000000EB; // 使用 Quad Read 命令
  FLASH_CTRL->CTRL |= FLASH_CTRL_XIP_ENABLE;
  g_xip_enabled = true;
}

void FLASH_DisableXIP(void) {
  FLASH_CTRL->CTRL &= ~FLASH_CTRL_XIP_ENABLE;
  g_xip_enabled = false;
}

int FLASH_GetInfo(FLASH_Info_t *info) {
  if (!info) {
    return -1;
  }

  // 读取 JEDEC ID (命令 0x9F)
  _FLASH_SendCommand(0x9F);

  uint32_t id = FLASH_CTRL->DATA;

  info->manufacturer_id = (id >> 16) & 0xFF;
  info->device_id = id & 0xFFFF;
  info->capacity = FLASH_SIZE;
  info->page_size = FLASH_PAGE_SIZE;
  info->sector_size = FLASH_SECTOR_SIZE;

  return 0;
}

/* ========================================================================= */
/* 内部函数实现                                                              */
/* ========================================================================= */

int _FLASH_SendCommand(uint8_t cmd) {
  FLASH_CTRL->CMD = cmd;
  FLASH_CTRL->CTRL |= (1U << 4); // 启动传输

  // 等待完成
  while (FLASH_CTRL->STATUS & FLASH_STATUS_BUSY)
    ;

  return 0;
}

int _FLASH_SendCommandAddr(uint8_t cmd, uint32_t addr) {
  FLASH_CTRL->CMD = cmd;
  FLASH_CTRL->ADDR = addr & 0x00FFFFFF; // 24-bit 地址
  FLASH_CTRL->CTRL |= (1U << 5);        // 启动命令+地址传输

  // 等待完成
  while (FLASH_CTRL->STATUS & FLASH_STATUS_BUSY)
    ;

  return 0;
}

int _FLASH_ReadStatus(void) {
  _FLASH_SendCommand(FLASH_CMD_READ_STATUS);
  return (int)(FLASH_CTRL->DATA & 0xFF);
}

int _FLASH_WriteEnable(void) {
  _FLASH_SendCommand(FLASH_CMD_WRITE_ENABLE);

  // 验证 WEL 位
  int status = _FLASH_ReadStatus();
  if (status < 0 || !(status & FLASH_STATUS_WEL)) {
    return -1;
  }

  return 0;
}

int _FLASH_PageProgram(uint32_t addr, const void *data, uint32_t size) {
  if (size > FLASH_PAGE_SIZE) {
    return -1;
  }

  // 写使能
  if (_FLASH_WriteEnable() != 0) {
    return -1;
  }

  // 发送页编程命令 + 地址
  _FLASH_SendCommandAddr(FLASH_CMD_PAGE_PROGRAM, addr);

  // 写入数据
  const uint8_t *ptr = (const uint8_t *)data;
  for (uint32_t i = 0; i < size; i++) {
    FLASH_CTRL->DATA = *ptr++;
  }

  // 等待完成 (页编程约需 0.7ms)
  return FLASH_WaitReady(5);
}
