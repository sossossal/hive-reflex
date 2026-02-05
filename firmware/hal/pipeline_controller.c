/**
 * @file pipeline_controller.c
 * @brief 软件流水线控制器实现
 */

#include "pipeline_controller.h"
#include <stdio.h>
#include <string.h>


// DMA寄存器（简化版，实际需要根据硬件调整）
#define DMA_BASE 0x40020000UL
#define DMA_SRC_REG (*(volatile uint32_t *)(DMA_BASE + 0x00))
#define DMA_DST_REG (*(volatile uint32_t *)(DMA_BASE + 0x04))
#define DMA_SIZE_REG (*(volatile uint32_t *)(DMA_BASE + 0x08))
#define DMA_CTRL_REG (*(volatile uint32_t *)(DMA_BASE + 0x0C))
#define DMA_STATUS_REG (*(volatile uint32_t *)(DMA_BASE + 0x10))

#define DMA_CTRL_START (1 << 0)
#define DMA_CTRL_ASYNC (1 << 1) // 异步模式
#define DMA_STATUS_BUSY (1 << 0)
#define DMA_STATUS_DONE (1 << 1)

// 性能计数器
static inline uint32_t get_cycle_count(void) {
  uint32_t count;
  asm volatile("rdcycle %0" : "=r"(count));
  return count;
}

void Pipeline_Init(PipelineController_t *ctrl) {
  memset(ctrl, 0, sizeof(PipelineController_t));
  ctrl->current_bank = 0;
  ctrl->loading_bank = 1;
  ctrl->state = PIPELINE_IDLE;

  printf("[Pipeline] Initialized (dual-buffer mode)\n");
}

void Pipeline_LoadLayerAsync(uint8_t bank_id, uint32_t flash_addr,
                             uint32_t size) {
  uint32_t bank_addr = Pipeline_GetBankAddr(bank_id);

  // 配置DMA
  DMA_SRC_REG = flash_addr;
  DMA_DST_REG = bank_addr;
  DMA_SIZE_REG = size;

  // 启动异步传输
  DMA_CTRL_REG = DMA_CTRL_START | DMA_CTRL_ASYNC;

#ifdef DEBUG_PIPELINE
  printf("[Pipeline] DMA start: Flash 0x%08lX -> Bank%d 0x%08lX (%lu bytes)\n",
         flash_addr, bank_id, bank_addr, size);
#endif
}

bool Pipeline_IsDMAComplete(void) {
  return (DMA_STATUS_REG & DMA_STATUS_DONE) != 0;
}

bool Pipeline_IsCIMComplete(void) {
  // 调用CIM HAL的繁忙检查
  extern bool CIM_IsBusy(CIM_Model_t * model);
  return !CIM_IsBusy(NULL); // TODO: 传入实际model
}

uint32_t Pipeline_WaitDMA(void) {
  uint32_t start = get_cycle_count();

  while (!Pipeline_IsDMAComplete()) {
    // 自旋等待
    asm volatile("nop");
  }

  uint32_t cycles = get_cycle_count() - start;

#ifdef DEBUG_PIPELINE
  printf("[Pipeline] DMA wait: %lu cycles\n", cycles);
#endif

  return cycles;
}

uint32_t Pipeline_WaitCIM(void) {
  uint32_t start = get_cycle_count();

  while (!Pipeline_IsCIMComplete()) {
    asm volatile("nop");
  }

  uint32_t cycles = get_cycle_count() - start;

#ifdef DEBUG_PIPELINE
  printf("[Pipeline] CIM wait: %lu cycles\n", cycles);
#endif

  return cycles;
}

void Pipeline_Sync(PipelineController_t *ctrl) {
  uint32_t stall = 0;
  uint32_t idle = 0;

  // 智能同步：优先等待慢的那个
  if (ctrl->dma_busy && !Pipeline_IsDMAComplete()) {
    stall = Pipeline_WaitDMA();
    ctrl->stall_cycles += stall;
  }

  if (ctrl->cim_busy && !Pipeline_IsCIMComplete()) {
    idle = Pipeline_WaitCIM();
    ctrl->idle_cycles += idle;
  }

  ctrl->dma_busy = false;
  ctrl->cim_busy = false;
  ctrl->state = PIPELINE_IDLE;
}

CIM_Status_t Pipeline_RunInference(PipelineController_t *ctrl,
                                   CIM_Model_t *model, const float *input,
                                   float *output) {
  printf("\n[Pipeline] Starting pipelined inference (%d layers)\n",
         model->num_layers);

  ctrl->total_layers = model->num_layers;
  uint32_t total_start = get_cycle_count();

  // 预加载第一层到Bank 0
  printf("[Pipeline] Preloading layer 0 to Bank 0...\n");
  Pipeline_LoadLayerAsync(0, model->layer_addrs[0], model->layer_sizes[0]);
  Pipeline_WaitDMA();

  float *layer_input = (float *)input;
  float *layer_output = NULL;

  for (int i = 0; i < model->num_layers; i++) {
    printf("[Pipeline] Layer %d/%d\n", i + 1, model->num_layers);

    // 1. 启动CIM计算当前层（使用current_bank）
    uint32_t current_bank_addr = Pipeline_GetBankAddr(ctrl->current_bank);

    printf("  [CIM] Computing on Bank %d...\n", ctrl->current_bank);

    // 分配输出缓冲
    static float intermediate_buffer[1024]; // 假设最大1024个神经元
    layer_output = (i == model->num_layers - 1) ? output : intermediate_buffer;

    // 启动CIM（异步）
    CIM_Status_t status =
        CIM_RunInferenceAsync(model, layer_input, layer_output);

    if (status != CIM_OK) {
      printf("  [ERROR] CIM failed: %d\n", status);
      return status;
    }

    ctrl->cim_busy = true;
    ctrl->state = PIPELINE_COMPUTING;

    // 2. 立即启动DMA加载下一层（如果还有）
    if (i + 1 < model->num_layers) {
      ctrl->loading_bank = 1 - ctrl->current_bank; // 切换Bank

      printf("  [DMA] Loading layer %d to Bank %d (async)...\n", i + 1,
             ctrl->loading_bank);

      Pipeline_LoadLayerAsync(ctrl->loading_bank, model->layer_addrs[i + 1],
                              model->layer_sizes[i + 1]);

      ctrl->dma_busy = true;
    }

    // 3. 同步：等待CIM计算完成 + DMA加载完成
    printf("  [SYNC] Waiting for CIM and DMA...\n");
    Pipeline_Sync(ctrl);

    // 4. 切换Bank
    ctrl->current_bank = 1 - ctrl->current_bank;

    // 5. 下一层的输入是这一层的输出
    layer_input = layer_output;
  }

  uint32_t total_cycles = get_cycle_count() - total_start;

  printf("[Pipeline] Inference complete!\n");
  printf("  Total cycles: %lu\n", total_cycles);
  printf("  Stall cycles: %lu (%.1f%%)\n", ctrl->stall_cycles,
         (ctrl->stall_cycles * 100.0f) / total_cycles);
  printf("  Idle cycles:  %lu (%.1f%%)\n", ctrl->idle_cycles,
         (ctrl->idle_cycles * 100.0f) / total_cycles);

  return CIM_OK;
}

void Pipeline_GetStats(PipelineController_t *ctrl, float *efficiency,
                       uint32_t *stall_cycles, uint32_t *idle_cycles) {
  uint32_t total = ctrl->stall_cycles + ctrl->idle_cycles;

  if (efficiency) {
    // 效率 = 1 - (停顿时间 / 总时间)
    *efficiency = total > 0 ? 1.0f - ((float)ctrl->stall_cycles / total) : 1.0f;
  }

  if (stall_cycles)
    *stall_cycles = ctrl->stall_cycles;
  if (idle_cycles)
    *idle_cycles = ctrl->idle_cycles;
}

void Pipeline_ResetStats(PipelineController_t *ctrl) {
  ctrl->stall_cycles = 0;
  ctrl->idle_cycles = 0;
  ctrl->total_layers = 0;
}
