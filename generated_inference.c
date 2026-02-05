/**
 * 自动生成的 CIM 推理代码
 * 由 MLIR 编译器生成
 */

#include "imc22_cim.h"
#include "model_loader.h"
#include <string.h>

// 权重数据 (在 FLASH 中)
extern const uint8_t model_weights[];
extern const uint32_t model_weights_size;

/**
 * @brief 模型推理函数 (Optimized Memory Layout)
 * Required Scratchpad Size: 4 bytes
 */
int model_inference_optimized(const float *input, float *output, void *context) {
    InferenceContext_t *ctx = (InferenceContext_t*)context;
    uint8_t *heap = (uint8_t*)ctx->temp_buffer;
    
    // Layer 0: 全连接 (unknown)
    {
        const float *weights = (const float*)(model_weights + weight_offset_0);
        const float *bias = weights + 0;
        
        // 使用 CIM 加速
        CIM_FullyConnected(
            input, output,
            weights, bias,
            0, 0,
            0  // 激活函数
        );
    }
    
    return 0;
}

// 权重偏移量 (自动计算)
const uint32_t weight_offset_0 = 0;
// ... (其他层的偏移)
