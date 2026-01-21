/**
 * @file imc22_npu.c
 * @brief IMC-22 神经加速器驱动实现
 */

#include "imc22_npu.h"
#include <string.h>

/* NPU SRAM 配置 */
#define NPU_SRAM_BASE (IMC22_NPU_BASE + 0x100000)
#define NPU_SRAM_SIZE (128 * 1024) // 128 KB

static uint32_t npu_sram_offset = 0; // 当前 SRAM 分配偏移

/* ========== 公共函数 ========== */

int NPU_Init(void) {
  // 复位 NPU
  NPU->CTRL = 0;
  DelayUs(10);

  // 使能 NPU
  NPU->CTRL = NPU_CTRL_EN;

  // 清除状态
  NPU->STATUS = 0xFFFFFFFF;

  // 复位 SRAM 分配器
  npu_sram_offset = 0;

  return 0;
}

int NPU_LoadModel(NPU_Model_t *model, const void *weight_data) {
  // 检查 SRAM 空间
  if (npu_sram_offset + model->weight_size > NPU_SRAM_SIZE) {
    return -1; // SRAM 不足
  }

  // 分配 SRAM 地址
  model->weight_addr = NPU_SRAM_BASE + npu_sram_offset;
  npu_sram_offset += model->weight_size;

  // 拷贝权重到 NPU SRAM (使用 DMA 或 memcpy)
  memcpy((void *)model->weight_addr, weight_data, model->weight_size);

  return 0;
}

int NPU_Inference(NPU_Context_t *ctx, const void *input, void *output) {
  const NPU_Model_t *model = ctx->model;

  // 配置数据类型
  uint32_t ctrl = NPU_CTRL_EN;
  if (model->dtype == NPU_DTYPE_INT8) {
    ctrl |= NPU_CTRL_INT8;
  } else if (model->dtype == NPU_DTYPE_FP16) {
    ctrl |= NPU_CTRL_FP16;
  }
  NPU->CTRL = ctrl;

  // 配置地址
  NPU->INPUT_ADDR = (uint32_t)input;
  NPU->OUTPUT_ADDR = (uint32_t)output;
  NPU->WEIGHT_ADDR = model->weight_addr;

  // 如果有 LSTM，需要配置隐藏状态地址
  // (这里简化处理，实际需要在寄存器中配置)

  // 配置层参数 (编码输入/输出维度)
  NPU->LAYER_CFG = (model->input_dims[3] << 0) |  // 输入特征数
                   (model->output_dims[3] << 16); // 输出特征数

  // 启动推理
  NPU->CTRL |= NPU_CTRL_START;

  // 等待完成
  return NPU_WaitDone(100000); // 100ms 超时
}

void NPU_StartInference(NPU_Context_t *ctx, const void *input) {
  const NPU_Model_t *model = ctx->model;

  NPU->INPUT_ADDR = (uint32_t)input;
  NPU->WEIGHT_ADDR = model->weight_addr;

  // 启动推理
  NPU->CTRL |= NPU_CTRL_START;
}

int NPU_WaitDone(uint32_t timeout_us) {
  if (timeout_us == 0) {
    // 无限等待
    while (NPU_IsBusy())
      ;
    return 0;
  }

  uint32_t start = GetCycleCount();
  uint32_t timeout_cycles = (IMC22_SYSCLK_HZ / 1000000) * timeout_us;

  while (NPU_IsBusy()) {
    if ((GetCycleCount() - start) > timeout_cycles) {
      return -1; // 超时
    }
  }

  // 检查错误
  if (NPU->STATUS & NPU_STATUS_ERROR) {
    return -1;
  }

  return 0;
}

/* ========== 中断处理 ========== */

void __attribute__((interrupt)) NPU_IRQHandler(void) {
  if (NPU->STATUS & NPU_STATUS_DONE) {
    // 清除中断标志
    NPU->STATUS = NPU_STATUS_DONE;

    // 调用用户回调
    extern void NPU_DoneCallback(void);
    NPU_DoneCallback();
  }
}
