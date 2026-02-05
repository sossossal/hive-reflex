/**
 * @file pipeline_controller.h
 * @brief 软件流水线控制器 - Flash IO优化策略1
 *
 * 通过双缓冲和DMA异步加载，掩盖Flash读取延迟
 * 理论加速比：1.5-2倍
 */

#ifndef PIPELINE_CONTROLLER_H
#define PIPELINE_CONTROLLER_H

#include "cim_hal.h"
#include <stdbool.h>
#include <stdint.h>


// 流水线状态
typedef enum {
  PIPELINE_IDLE,
  PIPELINE_LOADING,   // DMA正在加载
  PIPELINE_COMPUTING, // CIM正在计算
  PIPELINE_SYNCING    // 等待DMA和CIM同步
} PipelineState_t;

// 流水线控制器
typedef struct {
  uint8_t current_bank;     // CIM当前使用的Bank (0/1)
  uint8_t loading_bank;     // DMA正在加载的Bank
  bool cim_busy;            // CIM是否繁忙
  bool dma_busy;            // DMA是否繁忙
  uint32_t next_layer_addr; // 下一层在Flash中的地址
  PipelineState_t state;

  // 性能统计
  uint32_t total_layers;
  uint32_t stall_cycles; // 等待DMA的周期数
  uint32_t idle_cycles;  // CIM空闲周期数
} PipelineController_t;

// Bank地址映射
#define BANK_0_ADDR 0x20000000UL
#define BANK_1_ADDR 0x20008000UL // 假设每个Bank 32KB
#define NUM_BANKS 2

/**
 * @brief 初始化流水线控制器
 */
void Pipeline_Init(PipelineController_t *ctrl);

/**
 * @brief 使用流水线方式运行模型推理
 *
 * @param ctrl 流水线控制器
 * @param model 模型
 * @param input 输入数据
 * @param output 输出数据
 * @return CIM_Status_t
 */
CIM_Status_t Pipeline_RunInference(PipelineController_t *ctrl,
                                   CIM_Model_t *model, const float *input,
                                   float *output);

/**
 * @brief 异步加载下一层到指定Bank
 *
 * @param bank_id Bank ID (0 or 1)
 * @param flash_addr Flash地址
 * @param size 数据大小
 */
void Pipeline_LoadLayerAsync(uint8_t bank_id, uint32_t flash_addr,
                             uint32_t size);

/**
 * @brief 等待DMA加载完成
 *
 * @return 等待的周期数
 */
uint32_t Pipeline_WaitDMA(void);

/**
 * @brief 等待CIM计算完成
 *
 * @return 等待的周期数
 */
uint32_t Pipeline_WaitCIM(void);

/**
 * @brief 同步等待DMA和CIM
 *
 * 智能判断：如果DMA已完成，只等CIM；反之亦然
 */
void Pipeline_Sync(PipelineController_t *ctrl);

/**
 * @brief 获取性能统计
 */
void Pipeline_GetStats(PipelineController_t *ctrl,
                       float *efficiency,      // 流水线效率 (0-1)
                       uint32_t *stall_cycles, // 停顿周期
                       uint32_t *idle_cycles); // 空闲周期

/**
 * @brief 重置性能统计
 */
void Pipeline_ResetStats(PipelineController_t *ctrl);

/**
 * @brief 检查DMA是否完成
 */
bool Pipeline_IsDMAComplete(void);

/**
 * @brief 检查CIM是否完成
 */
bool Pipeline_IsCIMComplete(void);

/**
 * @brief 获取Bank的SRAM地址
 */
static inline uint32_t Pipeline_GetBankAddr(uint8_t bank_id) {
  return (bank_id == 0) ? BANK_0_ADDR : BANK_1_ADDR;
}

#endif // PIPELINE_CONTROLLER_H
