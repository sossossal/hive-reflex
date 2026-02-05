/**
 * @file cascade_model.h
 * @brief Strategy 3: 条件加载 - 级联模型支持
 *
 * 通过在早期层计算置信度，跳过后续不必要的层推理，减少 Flash IO
 *
 * 应用场景：
 * - 安防摄像头：70% 帧被判定为"正常"可跳过完整分析
 * - 工业质检：仅异常产品需要深度推理
 *
 * 预期收益：
 * - Flash IO 减少 > 70% (在大多数场景为"安全"的情况下)
 * - 实时响应性提升 3-5x
 */

#ifndef CASCADE_MODEL_H
#define CASCADE_MODEL_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 配置宏
// ============================================================================

#define CASCADE_MAX_EXIT_POINTS 4 /**< 最多支持 4 个早退出点 */
#define CASCADE_HISTORY_SIZE 8    /**< 用于自适应阈值的历史窗口 */

// ============================================================================
// 早退出点定义
// ============================================================================

/**
 * @brief 早退出点配置
 */
typedef struct {
  uint8_t layer_index;        /**< 退出点所在层索引 */
  float confidence_threshold; /**< 置信度阈值 (0.0-1.0) */
  uint32_t feature_dim;       /**< 特征维度 */

  // 轻量级分类器权重（需提前加载到 SRAM）
  const float *classifier_weights; /**< 分类器权重指针 */
  const float *classifier_bias;    /**< 偏置指针 */
  uint32_t classifier_output_dim;  /**< 输出类别数 */
} ExitPoint_t;

/**
 * @brief 级联模型结构
 */
typedef struct {
  // 退出点配置
  ExitPoint_t exit_points[CASCADE_MAX_EXIT_POINTS];
  uint8_t num_exit_points; /**< 实际启用的退出点数量 */

  // 全模型配置
  uint8_t total_layers;              /**< 模型总层数 */
  const uint32_t *layer_flash_sizes; /**< 每层在 Flash 的大小(字节) */

  // 自适应阈值调整
  bool enable_adaptive_threshold; /**< 是否启用自适应阈值 */
  float threshold_adjust_factor;  /**< 阈值调整系数 (0.9-1.1) */

  // 运行时状态
  uint32_t early_exit_count;        /**< 早退出次数统计 */
  uint32_t full_inference_count;    /**< 完整推理次数统计 */
  uint64_t total_flash_saved_bytes; /**< 累计节省的 Flash 读取量 */

} CascadeModel_t;

// ============================================================================
// 核心 API
// ============================================================================

/**
 * @brief 初始化级联模型
 *
 * @param model 级联模型结构指针
 * @param num_exits 退出点数量
 * @param total_layers 模型总层数
 * @return true 成功, false 失败
 */
bool Cascade_Init(CascadeModel_t *model, uint8_t num_exits,
                  uint8_t total_layers);

/**
 * @brief 配置单个退出点
 *
 * @param model 级联模型结构指针
 * @param exit_index 退出点索引 (0 ~ num_exits-1)
 * @param layer_index 该退出点位于的层索引
 * @param threshold 置信度阈值 (0.7 - 0.95 推荐)
 * @param feature_dim 特征维度
 * @param weights 分类器权重
 * @param bias 偏置
 * @param output_dim 输出类别数
 * @return true 成功, false 失败
 */
bool Cascade_ConfigureExit(CascadeModel_t *model, uint8_t exit_index,
                           uint8_t layer_index, float threshold,
                           uint32_t feature_dim, const float *weights,
                           const float *bias, uint32_t output_dim);

/**
 * @brief 设置每层的 Flash 大小（用于计算节省量）
 *
 * @param model 级联模型结构指针
 * @param layer_sizes 每层大小数组（长度 = total_layers）
 */
void Cascade_SetLayerSizes(CascadeModel_t *model, const uint32_t *layer_sizes);

/**
 * @brief 启用自适应阈值调整
 *
 * 根据历史推理结果自动调整置信度阈值，平衡准确率和性能
 *
 * @param model 级联模型结构指针
 * @param enable 是否启用
 */
void Cascade_EnableAdaptiveThreshold(CascadeModel_t *model, bool enable);

// ============================================================================
// 推理接口
// ============================================================================

/**
 * @brief 检查当前层是否可以早退出
 *
 * @param model 级联模型结构指针
 * @param current_layer 当前层索引
 * @param features 当前层输出特征
 * @param feature_dim 特征维度
 * @param[out] predicted_class 预测的类别 ID
 * @param[out] confidence 置信度 (0.0-1.0)
 * @return true 可以退出, false 需要继续
 */
bool Cascade_ShouldExit(CascadeModel_t *model, uint8_t current_layer,
                        const float *features, uint32_t feature_dim,
                        uint8_t *predicted_class, float *confidence);

/**
 * @brief 计算节省的 Flash 读取量
 *
 * 当早退出时，计算剩余层的 Flash 大小总和
 *
 * @param model 级联模型结构指针
 * @param exit_layer 退出的层索引
 * @return 节省的字节数
 */
uint32_t Cascade_CalculateSavedBytes(CascadeModel_t *model, uint8_t exit_layer);

/**
 * @brief 更新统计信息
 *
 * @param model 级联模型结构指针
 * @param did_early_exit 是否发生早退出
 * @param saved_bytes 节省的字节数
 */
void Cascade_UpdateStats(CascadeModel_t *model, bool did_early_exit,
                         uint32_t saved_bytes);

// ============================================================================
// 统计与调试
// ============================================================================

/**
 * @brief 获取早退出率
 *
 * @param model 级联模型结构指针
 * @return 早退出率 (0.0-1.0)
 */
float Cascade_GetEarlyExitRatio(const CascadeModel_t *model);

/**
 * @brief 获取平均节省字节数
 *
 * @param model 级联模型结构指针
 * @return 平均每次推理节省的字节数
 */
uint32_t Cascade_GetAverageSavedBytes(const CascadeModel_t *model);

/**
 * @brief 打印统计信息
 *
 * @param model 级联模型结构指针
 */
void Cascade_PrintStats(const CascadeModel_t *model);

/**
 * @brief 重置统计信息
 *
 * @param model 级联模型结构指针
 */
void Cascade_ResetStats(CascadeModel_t *model);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 简单的 Softmax + 置信度计算
 *
 * @param logits 分类器输出 logits
 * @param num_classes 类别数
 * @param[out] predicted_class 预测的类别
 * @param[out] confidence 最大概率值（作为置信度）
 */
void Cascade_ComputeConfidence(const float *logits, uint32_t num_classes,
                               uint8_t *predicted_class, float *confidence);

#ifdef __cplusplus
}
#endif

#endif // CASCADE_MODEL_H
