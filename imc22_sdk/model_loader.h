/**
 * @file model_loader.h
 * @brief 神经网络模型加载器
 * @version 2.0
 * @date 2026-01-19
 *
 * 从 FLASH 加载模型权重并初始化 CIM 硬件
 */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "imc22_cim.h"
#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 模型元数据结构                                                            */
/* ========================================================================= */

#define MODEL_MAGIC 0x43494D32 // "CIM2"
#define MODEL_VERSION 0x0200   // v2.0

typedef struct {
  uint32_t magic;         /**< 魔数 (MODEL_MAGIC) */
  uint16_t version;       /**< 版本号 */
  uint16_t reserved;      /**< 保留 */
  uint32_t model_size;    /**< 模型总大小 (bytes) */
  uint32_t weight_offset; /**< 权重数据偏移 */
  uint32_t weight_size;   /**< 权重数据大小 */
  uint32_t config_offset; /**< 配置数据偏移 */
  uint32_t config_size;   /**< 配置数据大小 */
  uint32_t crc32;         /**< CRC32 校验 */
  char model_name[32];    /**< 模型名称 */
  char model_hash[64];    /**< 模型哈希 (SHA-256) */
} ModelHeader_t;

typedef struct {
  uint32_t input_size;  /**< 输入维度 */
  uint32_t output_size; /**< 输出维度 */
  uint32_t hidden_size; /**< 隐藏层维度 */
  uint32_t num_layers;  /**< 层数 */
  CIM_DataType_t dtype; /**< 数据类型 */
  bool has_lstm;        /**< 是否有 LSTM */
  float quant_scale;    /**< 量化缩放因子 */
  int32_t quant_zero;   /**< 量化零点 */
} ModelConfig_t;

typedef struct {
  ModelHeader_t header; /**< 模型头 */
  ModelConfig_t config; /**< 模型配置 */
  const void *weights;  /**< 权重数据指针 */
  bool loaded;          /**< 是否已加载 */
  uint8_t bank_id;      /**< 加载到的 CIM Bank */
} Model_t;

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 从 FLASH 加载模型
 * @param flash_addr FLASH 地址（模型起始位置）
 * @param model 模型结构体指针
 * @return 0 成功, -1 失败
 */
int Model_LoadFromFlash(uint32_t flash_addr, Model_t *model);

/**
 * @brief 从内存加载模型
 * @param data 模型数据指针
 * @param model 模型结构体指针
 * @return 0 成功, -1 失败
 */
int Model_LoadFromMemory(const void *data, Model_t *model);

/**
 * @brief 卸载模型
 * @param model 模型结构体指针
 */
void Model_Unload(Model_t *model);

/**
 * @brief 验证模型完整性
 * @param model 模型结构体指针
 * @return true 有效, false 无效
 */
bool Model_Validate(const Model_t *model);

/**
 * @brief 将模型权重加载到 CIM SRAM
 * @param model 模型结构体指针
 * @param bank_id 目标 CIM Bank (0-3)
 * @return 0 成功, -1 失败
 */
int Model_LoadToCIM(Model_t *model, uint8_t bank_id);

/**
 * @brief 获取模型信息
 * @param model 模型结构体指针
 */
void Model_PrintInfo(const Model_t *model);

/* ========================================================================= */
/* 推理 API                                                                  */
/* ========================================================================= */

/**
 * @brief 推理上下文
 */
typedef struct {
  const Model_t *model;     /**< 模型指针 */
  float *lstm_h;            /**< LSTM 隐藏状态 */
  float *lstm_c;            /**< LSTM 细胞状态 */
  float *temp_buffer;       /**< 临时缓冲区 */
  uint32_t inference_count; /**< 推理次数 */
  uint32_t total_time_us;   /**< 总推理时间 (μs) */
} InferenceContext_t;

/**
 * @brief 创建推理上下文
 * @param model 模型指针
 * @return 上下文指针, NULL 表示失败
 */
InferenceContext_t *Inference_CreateContext(const Model_t *model);

/**
 * @brief 销毁推理上下文
 * @param ctx 上下文指针
 */
void Inference_DestroyContext(InferenceContext_t *ctx);

/**
 * @brief 执行推理
 * @param ctx 推理上下文
 * @param input 输入数据
 * @param output 输出数据
 * @return 0 成功, -1 失败
 */
int Inference_Run(InferenceContext_t *ctx, const float *input, float *output);

/**
 * @brief 重置 LSTM 状态
 * @param ctx 推理上下文
 */
void Inference_ResetState(InferenceContext_t *ctx);

/**
 * @brief 获取推理统计信息
 * @param ctx 推理上下文
 * @param avg_time_us 平均推理时间 (μs)
 * @param fps 帧率 (推理/秒)
 */
void Inference_GetStats(const InferenceContext_t *ctx, uint32_t *avg_time_us,
                        float *fps);

/* ========================================================================= */
/* 预定义模型路径                                                            */
/* ========================================================================= */

// FLASH 分区地址
#define MODEL_FLASH_BASE 0x08090000    // 神经网络权重分区起始地址
#define MODEL_FLASH_SIZE (1024 * 1024) // 1MB

// 预定义模型
#define MODEL_REFLEX_V1 (MODEL_FLASH_BASE + 0x00000) // ReflexNet V1
#define MODEL_REFLEX_V2 (MODEL_FLASH_BASE + 0x40000) // ReflexNet V2
#define MODEL_CUSTOM (MODEL_FLASH_BASE + 0x80000)    // 自定义模型

#ifdef __cplusplus
}
#endif

#endif /* MODEL_LOADER_H */
