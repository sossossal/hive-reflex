/**
 * @file compression.c
 * @brief 实时解压缩库实现
 */

#include "compression.h"
#include <stdio.h>
#include <string.h>


#define COMPRESSION_MAGIC 0x484C5A43 // "HRZC"

// 全局统计
static DecompressionStats_t g_stats = {0};

// 性能计数器
static inline uint32_t get_cycle_count(void) {
  uint32_t count;
  asm volatile("rdcycle %0" : "=r"(count));
  return count;
}

uint32_t Decompress_RLE(const uint8_t *src, uint8_t *dst, uint32_t src_size,
                        uint32_t dst_size) {
  const uint8_t *src_end = src + src_size;
  uint8_t *dst_start = dst;
  uint8_t *dst_end = dst + dst_size;

  uint32_t start_cycles = get_cycle_count();

  while (src < src_end && dst < dst_end) {
    uint8_t value = *src++;
    uint8_t count = *src++;

    // 优化：零值使用memset
    if (value == 0) {
      uint32_t bytes_to_write =
          (dst + count <= dst_end) ? count : (dst_end - dst);
      memset(dst, 0, bytes_to_write);
      dst += bytes_to_write;
    } else {
      // 非零值逐个写入
      for (uint8_t i = 0; i < count && dst < dst_end; i++) {
        *dst++ = value;
      }
    }
  }

  uint32_t cycles = get_cycle_count() - start_cycles;
  uint32_t bytes_out = dst - dst_start;

  // 更新统计
  g_stats.total_bytes_in += src_size;
  g_stats.total_bytes_out += bytes_out;
  g_stats.total_time_cycles += cycles;

  return bytes_out;
}

uint32_t Decompress_LZ4(const uint8_t *src, uint8_t *dst, uint32_t src_size) {
  // LZ4简化实现（实际应使用优化的LZ4库）
  const uint8_t *src_end = src + src_size;
  uint8_t *dst_start = dst;

  uint32_t start_cycles = get_cycle_count();

  while (src < src_end) {
    uint8_t token = *src++;
    uint32_t literal_length = token >> 4;
    uint32_t match_length = (token & 0x0F) + 4;

    // 扩展字面量长度
    if (literal_length == 15) {
      uint8_t len;
      do {
        len = *src++;
        literal_length += len;
      } while (len == 255);
    }

    // 复制字面量
    memcpy(dst, src, literal_length);
    src += literal_length;
    dst += literal_length;

    if (src >= src_end)
      break;

    // 读取偏移
    uint16_t offset = *(uint16_t *)src;
    src += 2;

    // 扩展匹配长度
    if ((token & 0x0F) == 15) {
      uint8_t len;
      do {
        len = *src++;
        match_length += len;
      } while (len == 255);
    }

    // 复制匹配
    uint8_t *match_src = dst - offset;
    for (uint32_t i = 0; i < match_length; i++) {
      *dst++ = *match_src++;
    }
  }

  uint32_t cycles = get_cycle_count() - start_cycles;
  uint32_t bytes_out = dst - dst_start;

  g_stats.total_bytes_in += src_size;
  g_stats.total_bytes_out += bytes_out;
  g_stats.total_time_cycles += cycles;

  return bytes_out;
}

uint32_t Decompress_Delta(const uint8_t *src, uint8_t *dst, uint32_t src_size) {
  if (src_size == 0)
    return 0;

  uint32_t start_cycles = get_cycle_count();

  // 第一个值
  int8_t current = (int8_t)*src++;
  *dst++ = (uint8_t)current;

  // 后续差分值
  for (uint32_t i = 1; i < src_size; i++) {
    int8_t delta = (int8_t)*src++;
    current += delta;
    *dst++ = (uint8_t)current;
  }

  uint32_t cycles = get_cycle_count() - start_cycles;

  g_stats.total_bytes_in += src_size;
  g_stats.total_bytes_out += src_size;
  g_stats.total_time_cycles += cycles;

  return src_size;
}

uint32_t Decompress_Auto(const uint8_t *src, uint8_t *dst,
                         uint32_t max_dst_size) {
  // 读取头信息
  CompressionHeader_t *header = (CompressionHeader_t *)src;

  // 验证魔数
  if (header->magic != COMPRESSION_MAGIC) {
    printf("[Decompress] Invalid magic: 0x%08lX\n", header->magic);
    return 0;
  }

  // 检查空间
  if (header->decompressed_size > max_dst_size) {
    printf("[Decompress] Buffer too small: need %lu, have %lu\n",
           header->decompressed_size, max_dst_size);
    return 0;
  }

  const uint8_t *compressed_data = src + sizeof(CompressionHeader_t);
  uint32_t bytes_decompressed = 0;

  printf("[Decompress] Type: %d, Size: %lu -> %lu\n", header->type,
         header->compressed_size, header->decompressed_size);

  // 根据类型选择解压算法
  switch (header->type) {
  case COMPRESS_NONE:
    memcpy(dst, compressed_data, header->decompressed_size);
    bytes_decompressed = header->decompressed_size;
    break;

  case COMPRESS_RLE:
    bytes_decompressed =
        Decompress_RLE(compressed_data, dst, header->compressed_size,
                       header->decompressed_size);
    break;

  case COMPRESS_LZ4:
    bytes_decompressed =
        Decompress_LZ4(compressed_data, dst, header->compressed_size);
    break;

  case COMPRESS_DELTA:
    bytes_decompressed =
        Decompress_Delta(compressed_data, dst, header->compressed_size);
    break;

  default:
    printf("[Decompress] Unknown type: %d\n", header->type);
    return 0;
  }

  // 验证大小
  if (bytes_decompressed != header->decompressed_size) {
    printf("[Decompress] Size mismatch: expected %lu, got %lu\n",
           header->decompressed_size, bytes_decompressed);
  }

  return bytes_decompressed;
}

uint32_t Decompress_Chunked(const uint8_t *src, uint8_t *dst, uint32_t src_size,
                            uint32_t block_size,
                            void (*callback)(uint8_t *data, uint32_t size)) {
  uint32_t total_decompressed = 0;
  uint32_t remaining = src_size;

  while (remaining > 0) {
    uint32_t chunk_size = (remaining < block_size) ? remaining : block_size;

    uint32_t decompressed = Decompress_Auto(src, dst, chunk_size);

    if (callback) {
      callback(dst, decompressed);
    }

    src += chunk_size;
    dst += decompressed;
    remaining -= chunk_size;
    total_decompressed += decompressed;
  }

  return total_decompressed;
}

uint32_t Decompress_WithTimeout(const uint8_t *src, uint8_t *dst,
                                uint32_t src_size, uint32_t max_cycles) {
  uint32_t start = get_cycle_count();

  // 简化：使用Auto解压，然后检查时间
  uint32_t result = Decompress_Auto(src, dst, 0xFFFFFFFF);

  uint32_t elapsed = get_cycle_count() - start;

  if (elapsed > max_cycles) {
    printf("[Decompress] Timeout: %lu > %lu cycles\n", elapsed, max_cycles);
    return 0;
  }

  return result;
}

bool Decompress_Validate(const uint8_t *src, uint32_t size) {
  if (size < sizeof(CompressionHeader_t)) {
    return false;
  }

  CompressionHeader_t *header = (CompressionHeader_t *)src;

  // 检查魔数
  if (header->magic != COMPRESSION_MAGIC) {
    return false;
  }

  // 检查类型
  if (header->type >= 4) { // 超过已知类型
    return false;
  }

  // 检查大小
  if (header->compressed_size == 0 || header->decompressed_size == 0) {
    return false;
  }

  // TODO: 验证CRC32

  return true;
}

void Decompress_GetStats(DecompressionStats_t *stats) {
  memcpy(stats, &g_stats, sizeof(DecompressionStats_t));

  // 计算平均速度
  if (g_stats.total_time_cycles > 0) {
    // 假设CPU频率100MHz
    float time_seconds = g_stats.total_time_cycles / 100000000.0f;
    float mb_out = g_stats.total_bytes_out / (1024.0f * 1024.0f);
    stats->avg_speed_mbps = mb_out / time_seconds;
  }
}

void Decompress_ResetStats(void) {
  memset(&g_stats, 0, sizeof(DecompressionStats_t));
}
