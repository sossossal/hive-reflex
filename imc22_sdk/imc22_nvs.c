/**
 * @file imc22_nvs.c
 * @brief NVS (Non-Volatile Storage) 实现
 * 基于 FLASH 的键值对存储系统
 */

#include "imc22_nvs.h"
#include "imc22_flash.h"
#include <stdio.h>
#include <string.h>


/* ========================================================================= */
/* 内部数据结构                                                              */
/* ========================================================================= */

typedef struct {
  char key[NVS_MAX_KEY_LEN];
  NVS_Type_t type;
  uint16_t size;
  uint16_t flags;
  uint32_t crc32;
  uint8_t data[]; // 柔性数组
} __attribute__((packed)) NVS_Item_t;

#define NVS_ITEM_FLAG_VALID (1U << 0)
#define NVS_ITEM_FLAG_DELETED (1U << 1)

/* 分区管理 */
static bool g_nvs_initialized = false;
static uint32_t g_nvs_base_addr = NVS_FLASH_BASE;
static uint32_t g_nvs_next_addr = NVS_FLASH_BASE;
static uint32_t g_erase_count = 0;

/* ========================================================================= */
/* CRC32 实现                                                                */
/* ========================================================================= */

uint32_t NVS_CalcCRC32(const void *data, uint32_t size) {
  const uint8_t *ptr = (const uint8_t *)data;
  uint32_t crc = 0xFFFFFFFF;

  for (uint32_t i = 0; i < size; i++) {
    crc ^= ptr[i];
    for (int j = 0; j < 8; j++) {
      crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
  }

  return ~crc;
}

/* ========================================================================= */
/* 内部辅助函数                                                              */
/* ========================================================================= */

static NVS_Item_t *_NVS_FindItem(const char *key) {
  uint32_t addr = g_nvs_base_addr;
  uint32_t end_addr = g_nvs_base_addr + NVS_FLASH_SIZE;

  while (addr < end_addr) {
    NVS_Item_t item;

    // 读取条目头
    if (NVS_FlashRead(addr, &item, sizeof(NVS_Item_t)) != 0) {
      break;
    }

    // 检查是否到达未使用区域
    if (item.key[0] == 0xFF) {
      break;
    }

    // 检查键是否匹配
    if ((item.flags & NVS_ITEM_FLAG_VALID) &&
        !(item.flags & NVS_ITEM_FLAG_DELETED) && strcmp(item.key, key) == 0) {

      // 验证 CRC
      uint8_t *data = malloc(item.size);
      if (data) {
        NVS_FlashRead(addr + sizeof(NVS_Item_t), data, item.size);
        uint32_t crc = NVS_CalcCRC32(data, item.size);
        free(data);

        if (crc == item.crc32) {
          return (NVS_Item_t *)addr; // 返回 FLASH 地址
        }
      }
    }

    // 移动到下一个条目
    addr += sizeof(NVS_Item_t) + item.size;

    // 对齐到 4 字节
    addr = (addr + 3) & ~3;
  }

  return NULL;
}

static int _NVS_GarbageCollection(void) {
  // 简化版本: 擦除整个分区并重建
  printf("NVS: 执行垃圾回收...\n");

  // TODO: 实际应该先备份有效数据

  // 擦除第一个扇区
  if (FLASH_EraseSector(g_nvs_base_addr) != 0) {
    return -1;
  }

  g_nvs_next_addr = g_nvs_base_addr;
  g_erase_count++;

  return 0;
}

/* ========================================================================= */
/* 公共 API 实现                                                             */
/* ========================================================================= */

int NVS_Init(void) {
  if (g_nvs_initialized) {
    return 0;
  }

  // 初始化 FLASH
  if (FLASH_Init(true) != 0) {
    return -1;
  }

  // 扫描分区，找到下一个可用地址
  g_nvs_next_addr = g_nvs_base_addr;

  uint32_t addr = g_nvs_base_addr;
  uint32_t end_addr = g_nvs_base_addr + NVS_FLASH_SIZE;

  while (addr < end_addr) {
    NVS_Item_t item;

    if (NVS_FlashRead(addr, &item, sizeof(NVS_Item_t)) != 0) {
      break;
    }

    // 检查是否到达未使用区域
    if (item.key[0] == 0xFF) {
      break;
    }

    // 移动到下一个条目
    addr += sizeof(NVS_Item_t) + item.size;
    addr = (addr + 3) & ~3;

    g_nvs_next_addr = addr;
  }

  g_nvs_initialized = true;

  printf("NVS 初始化完成, 下一个地址: 0x%08lX\n", g_nvs_next_addr);

  return 0;
}

int NVS_Format(void) {
  // 擦除整个 NVS 分区
  for (uint32_t addr = g_nvs_base_addr; addr < g_nvs_base_addr + NVS_FLASH_SIZE;
       addr += FLASH_SECTOR_SIZE) {

    if (FLASH_EraseSector(addr) != 0) {
      return -1;
    }
  }

  g_nvs_next_addr = g_nvs_base_addr;
  g_erase_count = 0;

  return 0;
}

int NVS_Write(const char *key, const void *data, uint32_t size,
              NVS_Type_t type) {
  if (!g_nvs_initialized || !key || !data || size == 0 ||
      size > NVS_MAX_VALUE_LEN) {
    return -1;
  }

  // 检查空间
  uint32_t required = sizeof(NVS_Item_t) + size;
  required = (required + 3) & ~3; // 4 字节对齐

  if (g_nvs_next_addr + required > g_nvs_base_addr + NVS_FLASH_SIZE) {
    // 空间不足, 执行垃圾回收
    if (_NVS_GarbageCollection() != 0) {
      return -1;
    }
  }

  // 构建条目
  NVS_Item_t item;
  memset(&item, 0, sizeof(NVS_Item_t));

  strncpy(item.key, key, NVS_MAX_KEY_LEN - 1);
  item.type = type;
  item.size = size;
  item.flags = NVS_ITEM_FLAG_VALID;
  item.crc32 = NVS_CalcCRC32(data, size);

  // 写入条目头
  if (NVS_FlashWrite(g_nvs_next_addr, &item, sizeof(NVS_Item_t)) != 0) {
    return -1;
  }

  // 写入数据
  if (NVS_FlashWrite(g_nvs_next_addr + sizeof(NVS_Item_t), data, size) != 0) {
    return -1;
  }

  // 更新下一个地址
  g_nvs_next_addr += required;

  return 0;
}

int NVS_Read(const char *key, void *data, uint32_t max_size) {
  if (!g_nvs_initialized || !key || !data) {
    return -1;
  }

  // 查找条目
  NVS_Item_t *item_ptr = _NVS_FindItem(key);
  if (!item_ptr) {
    return -1; // 未找到
  }

  // 读取条目头
  NVS_Item_t item;
  uint32_t item_addr = (uint32_t)item_ptr;

  if (NVS_FlashRead(item_addr, &item, sizeof(NVS_Item_t)) != 0) {
    return -1;
  }

  // 检查大小
  if (item.size > max_size) {
    return -1;
  }

  // 读取数据
  if (NVS_FlashRead(item_addr + sizeof(NVS_Item_t), data, item.size) != 0) {
    return -1;
  }

  return item.size;
}

int NVS_Erase(const char *key) {
  // 标记为已删除 (不实际擦除)
  NVS_Item_t *item_ptr = _NVS_FindItem(key);
  if (!item_ptr) {
    return -1;
  }

  // 读取条目
  NVS_Item_t item;
  uint32_t item_addr = (uint32_t)item_ptr;

  if (NVS_FlashRead(item_addr, &item, sizeof(NVS_Item_t)) != 0) {
    return -1;
  }

  // 标记为已删除
  item.flags |= NVS_ITEM_FLAG_DELETED;

  // 写回 (仅修改 flags 字段)
  if (NVS_FlashWrite(item_addr + offsetof(NVS_Item_t, flags), &item.flags,
                     sizeof(item.flags)) != 0) {
    return -1;
  }

  return 0;
}

bool NVS_Exists(const char *key) { return _NVS_FindItem(key) != NULL; }

void NVS_GetStats(NVS_Stats_t *stats) {
  if (!stats) {
    return;
  }

  memset(stats, 0, sizeof(NVS_Stats_t));

  // 遍历分区统计
  uint32_t addr = g_nvs_base_addr;
  uint32_t end_addr = g_nvs_base_addr + NVS_FLASH_SIZE;

  while (addr < end_addr) {
    NVS_Item_t item;

    if (NVS_FlashRead(addr, &item, sizeof(NVS_Item_t)) != 0) {
      break;
    }

    if (item.key[0] == 0xFF) {
      break;
    }

    if (item.flags & NVS_ITEM_FLAG_VALID) {
      stats->total_entries++;
      stats->used_bytes += sizeof(NVS_Item_t) + item.size;
    }

    addr += sizeof(NVS_Item_t) + item.size;
    addr = (addr + 3) & ~3;
  }

  stats->free_bytes = NVS_FLASH_SIZE - stats->used_bytes;
  stats->erase_count = g_erase_count;
}

int NVS_Commit(void) {
  // 当前实现立即写入, 无需提交
  return 0;
}

void NVS_Iterate(bool (*callback)(const NVS_Entry_t *entry, void *user_data),
                 void *user_data) {
  // TODO: 实现遍历
}

void NVS_WearLeveling(void) {
  // TODO: 实现磨损均衡
}

/* ========================================================================= */
/* FLASH 后端接口                                                            */
/* ========================================================================= */

int NVS_FlashRead(uint32_t addr, void *buf, uint32_t size) {
  return FLASH_Read(addr, buf, size);
}

int NVS_FlashWrite(uint32_t addr, const void *data, uint32_t size) {
  return FLASH_Write(addr, data, size);
}

int NVS_FlashEraseSector(uint32_t addr) { return FLASH_EraseSector(addr); }
