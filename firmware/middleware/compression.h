/**
 * @file compression.h
 * @brief 实时解压缩库 - Flash IO优化策略2
 *
 * 支持多种压缩算法：RLE, LZ4, Huffman
 * 目标：2-5倍压缩率，500MB/s+解压速度
 */

#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <stdbool.h>
#include <stdint.h>


// 压缩类型
typedef enum {
  COMPRESS_NONE = 0,
  COMPRESS_RLE,     // 游程编码（适合稀疏权重）
  COMPRESS_LZ4,     // LZ4快速解压（适合密集权重）
  COMPRESS_HUFFMAN, // Huffman编码（高压缩率）
  COMPRESS_DELTA    // 差分编码（适合量化权重）
} CompressionType_t;

// 压缩头信息
typedef struct {
  uint32_t magic;             // 魔数 0x484C5A43 ("HRZC")
  CompressionType_t type;     // 压缩类型
  uint32_t compressed_size;   // 压缩后大小
  uint32_t decompressed_size; // 原始大小
  uint32_t checksum;          // CRC32校验和
} CompressionHeader_t;

// 解压统计
typedef struct {
  uint32_t total_bytes_in;
  uint32_t total_bytes_out;
  uint32_t total_time_cycles;
  float avg_speed_mbps; // 平均解压速度 (MB/s)
} DecompressionStats_t;

/**
 * @brief RLE解压（针对稀疏神经网络优化）
 *
 * 格式：[value, count] 对
 * 示例：0,255, 1,10 表示 255个0，10个1
 *
 * @param src 压缩数据
 * @param dst 解压缓冲区
 * @param src_size 压缩数据大小
 * @param dst_size 解压后大小
 * @return 解压的字节数，失败返回0
 */
uint32_t Decompress_RLE(const uint8_t *src, uint8_t *dst, uint32_t src_size,
                        uint32_t dst_size);

/**
 * @brief LZ4解压（快速通用解压）
 *
 * 标准LZ4格式，解压速度可达GB/s级
 *
 * @param src 压缩数据
 * @param dst 解压缓冲区
 * @param src_size 压缩数据大小
 * @return 解压的字节数
 */
uint32_t Decompress_LZ4(const uint8_t *src, uint8_t *dst, uint32_t src_size);

/**
 * @brief Huffman解压（高压缩率）
 *
 * @param src 压缩数据（含码表）
 * @param dst 解压缓冲区
 * @param src_size 压缩数据大小
 * @return 解压的字节数
 */
uint32_t Decompress_Huffman(const uint8_t *src, uint8_t *dst,
                            uint32_t src_size);

/**
 * @brief 差分解压（适合量化权重）
 *
 * 格式：首值 + 后续差分
 *
 * @param src 压缩数据
 * @param dst 解压缓冲区
 * @param src_size 压缩数据大小
 * @return 解压的字节数
 */
uint32_t Decompress_Delta(const uint8_t *src, uint8_t *dst, uint32_t src_size);

/**
 * @brief 自动检测并解压
 *
 * 读取头信息，自动选择解压算法
 *
 * @param src 压缩数据（含头）
 * @param dst 解压缓冲区
 * @param max_dst_size 最大输出大小
 * @return 解压的字节数
 */
uint32_t Decompress_Auto(const uint8_t *src, uint8_t *dst,
                         uint32_t max_dst_size);

/**
 * @brief 分块解压（适合大数据）
 *
 * 数据分成多个块，可以边解压边使用
 *
 * @param src 压缩数据
 * @param dst 解压缓冲区
 * @param src_size 压缩数据大小
 * @param block_size 每块大小
 * @param callback 每块解压完成后的回调
 * @return 总解压字节数
 */
uint32_t Decompress_Chunked(const uint8_t *src, uint8_t *dst, uint32_t src_size,
                            uint32_t block_size,
                            void (*callback)(uint8_t *data, uint32_t size));

/**
 * @brief 带超时的解压
 *
 * 防止解压时间过长阻塞系统
 *
 * @param src 压缩数据
 * @param dst 解压缓冲区
 * @param src_size 压缩数据大小
 * @param max_cycles 最大允许周期数
 * @return 解压的字节数，超时返回0
 */
uint32_t Decompress_WithTimeout(const uint8_t *src, uint8_t *dst,
                                uint32_t src_size, uint32_t max_cycles);

/**
 * @brief 获取解压统计信息
 */
void Decompress_GetStats(DecompressionStats_t *stats);

/**
 * @brief 重置统计信息
 */
void Decompress_ResetStats(void);

/**
 * @brief 验证压缩数据完整性
 *
 * @param src 压缩数据
 * @param size 数据大小
 * @return true if valid
 */
bool Decompress_Validate(const uint8_t *src, uint32_t size);

// 压缩工具函数（用于离线压缩，不在设备上运行）
#ifdef COMPRESSION_TOOLS

/**
 * @brief RLE压缩
 */
uint32_t Compress_RLE(const uint8_t *src, uint8_t *dst, uint32_t src_size);

/**
 * @brief 计算最佳压缩算法
 *
 * 对比不同算法，选择压缩率最高的
 */
CompressionType_t Compress_FindBest(const uint8_t *data, uint32_t size,
                                    float *compression_ratio);

#endif // COMPRESSION_TOOLS

#endif // COMPRESSION_H
