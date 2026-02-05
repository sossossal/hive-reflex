/**
 * @file flash_io_optimizer.h
 * @brief Flash IO 优化集成框架
 *
 * 整合三大策略：
 * - Strategy 1: 软件流水线（Pipeline）
 * - Strategy 2: 实时解压缩（Compression）
 * - Strategy 3: 条件加载（Cascade）
 *
 * 提供统一的推理接口，自动选择最优策略组合
 */

#ifndef FLASH_IO_OPTIMIZER_H
#define FLASH_IO_OPTIMIZER_H

#include "../hal/pipeline_controller.h"
#include "cascade_model.h"
#include "compression.h"


#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 优化配置
// ============================================================================

/**
 * @brief 优化策略标志
 */
typedef enum {
  OPT_NONE = 0x00,     /**< 无优化（基线） */
  OPT_PIPELINE = 0x01, /**< 启用软件流水线 */
  OPT_COMPRESS = 0x02, /**< 启用实时解压缩 */
  OPT_CASCADE = 0x04,  /**< 启用条件加载 */
  OPT_ALL = 0x07       /**< 启用所有策略 */
} OptimizationFlags_t;

/**
 * @brief 优化器配置
 */
typedef struct {
  OptimizationFlags_t flags; /**< 启用的策略 */

  // Pipeline 配置
  PipelineController_t *pipeline_ctrl; /**< 流水线控制器 */

  // Compression 配置
  CompressionType_t compression_type; /**< 压缩类型 */
  uint32_t decompress_buffer_size;    /**< 解压缩缓冲区大小 */

  // Cascade 配置
  CascadeModel_t *cascade_model; /**< 级联模型 */

  // 性能统计
  uint64_t total_inference_time_us; /**< 累计推理时间（微秒） */
  uint64_t total_flash_read_bytes;  /**< 累计 Flash 读取量 */
  uint32_t inference_count;         /**< 推理次数 */

} FlashIOOptimizer_t;

// ============================================================================
// 初始化
// ============================================================================

/**
 * @brief 初始化 Flash IO 优化器
 *
 * @param opt 优化器结构指针
 * @param flags 启用的优化策略
 * @return true 成功, false 失败
 */
bool FlashOpt_Init(FlashIOOptimizer_t *opt, OptimizationFlags_t flags);

/**
 * @brief 配置 Pipeline 策略
 *
 * @param opt 优化器结构指针
 * @param pipeline_ctrl 流水线控制器
 * @return true 成功, false 失败
 */
bool FlashOpt_ConfigPipeline(FlashIOOptimizer_t *opt,
                             PipelineController_t *pipeline_ctrl);

/**
 * @brief 配置 Compression 策略
 *
 * @param opt 优化器结构指针
 * @param type 压缩类型
 * @param buffer_size 解压缓冲区大小
 * @return true 成功, false 失败
 */
bool FlashOpt_ConfigCompression(FlashIOOptimizer_t *opt, CompressionType_t type,
                                uint32_t buffer_size);

/**
 * @brief 配置 Cascade 策略
 *
 * @param opt 优化器结构指针
 * @param cascade_model 级联模型
 * @return true 成功, false 失败
 */
bool FlashOpt_ConfigCascade(FlashIOOptimizer_t *opt,
                            CascadeModel_t *cascade_model);

// ============================================================================
// 推理接口
// ============================================================================

/**
 * @brief 优化的层推理
 *
 * 根据启用的策略自动选择最优执行路径：
 * - 如果启用 Pipeline: 异步加载下一层
 * - 如果启用 Compression: 自动解压权重
 * - 如果启用 Cascade: 检查早退出条件
 *
 * @param opt 优化器结构指针
 * @param layer_index 当前层索引
 * @param flash_addr 层权重在 Flash 的地址
 * @param flash_size 权重大小（压缩后）
 * @param input 输入数据
 * @param output 输出数据
 * @param[out] should_exit 是否早退出（仅 Cascade 启用时有效）
 * @return true 成功, false 失败
 */
bool FlashOpt_InferLayer(FlashIOOptimizer_t *opt, uint8_t layer_index,
                         uint32_t flash_addr, uint32_t flash_size,
                         const float *input, float *output, bool *should_exit);

/**
 * @brief 完整模型推理（集成所有策略）
 *
 * @param opt 优化器结构指针
 * @param num_layers 模型层数
 * @param layer_flash_addrs 每层 Flash 地址数组
 * @param layer_flash_sizes 每层大小数组
 * @param input 模型输入
 * @param output 模型输出
 * @return true 成功, false 失败
 */
bool FlashOpt_RunInference(FlashIOOptimizer_t *opt, uint8_t num_layers,
                           const uint32_t *layer_flash_addrs,
                           const uint32_t *layer_flash_sizes,
                           const float *input, float *output);

// ============================================================================
// 性能分析
// ============================================================================

/**
 * @brief 打印性能统计
 *
 * @param opt 优化器结构指针
 */
void FlashOpt_PrintStats(const FlashIOOptimizer_t *opt);

/**
 * @brief 获取平均推理时间
 *
 * @param opt 优化器结构指针
 * @return 平均推理时间（微秒）
 */
uint32_t FlashOpt_GetAverageInferenceTime(const FlashIOOptimizer_t *opt);

/**
 * @brief 获取 Flash 带宽利用率
 *
 * @param opt 优化器结构指针
 * @return Flash 读取带宽 (bytes/s)
 */
uint64_t FlashOpt_GetFlashBandwidth(const FlashIOOptimizer_t *opt);

/**
 * @brief 重置统计数据
 *
 * @param opt 优化器结构指针
 */
void FlashOpt_ResetStats(FlashIOOptimizer_t *opt);

#ifdef __cplusplus
}
#endif

#endif // FLASH_IO_OPTIMIZER_H
