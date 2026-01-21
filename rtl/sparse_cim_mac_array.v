/**
 * 稀疏感知 CIM MAC 阵列
 * 支持动态跳过零值/近零值乘法，减少无效运算 20-50%
 * 
 * @file sparse_cim_mac_array.v
 * @author Hive-Reflex Team
 * @version 2.1.0
 */

`timescale 1ns / 1ps

module sparse_cim_mac_array #(
    parameter MAC_COUNT = 256,          // MAC 单元数量
    parameter DATA_WIDTH = 8,           // 数据位宽 (int8)
    parameter ACC_WIDTH = 32,           // 累加器位宽
    parameter SPARSITY_THRESHOLD = 2,   // 稀疏阈值 |value| < threshold → skip
    parameter ENABLE_CSR = 1            // 启用 CSR 压缩格式
)(
    input wire clk,
    input wire rst_n,
    
    // ========================================================================
    // 数据接口
    // ========================================================================
    
    // 输入激活值 (稠密格式)
    input wire signed [DATA_WIDTH-1:0] input_data [0:MAC_COUNT-1],
    // 输入激活值有效掩码 (可选，用于预计算稀疏)
    input wire [MAC_COUNT-1:0] input_valid_mask,
    input wire input_use_mask,  // 是否使用外部掩码
    
    // 权重 (稠密格式)
    input wire signed [DATA_WIDTH-1:0] weight_data [0:MAC_COUNT-1],
    // 权重有效掩码 (可选)
    input wire [MAC_COUNT-1:0] weight_valid_mask,
    input wire weight_use_mask,
    
    // ========================================================================
    // 控制接口
    // ========================================================================
    
    input wire start,
    input wire sparse_enable,           // 启用稀疏模式
    input wire [DATA_WIDTH-1:0] threshold_config,  // 运行时阈值配置
    output reg done,
    output reg busy,
    
    // ========================================================================
    // 输出接口
    // ========================================================================
    
    output reg signed [ACC_WIDTH-1:0] result,
    
    // 稀疏统计
    output reg [15:0] total_ops,        // 总操作数
    output reg [15:0] skipped_ops,      // 跳过的操作数
    output wire [7:0] sparsity_ratio,   // 稀疏率 (0-100%)
    output wire sparse_mode_active
);

    // ========================================================================
    // 内部信号
    // ========================================================================
    
    // 动态阈值
    wire [DATA_WIDTH-1:0] active_threshold;
    assign active_threshold = (threshold_config != 0) ? threshold_config : SPARSITY_THRESHOLD[DATA_WIDTH-1:0];
    
    // 稀疏检测掩码
    reg [MAC_COUNT-1:0] input_sparse_mask;   // 1 = 非零，0 = 跳过
    reg [MAC_COUNT-1:0] weight_sparse_mask;
    reg [MAC_COUNT-1:0] combined_mask;       // 最终有效掩码
    
    // MAC 结果
    wire signed [2*DATA_WIDTH-1:0] mac_products [0:MAC_COUNT-1];
    wire signed [2*DATA_WIDTH-1:0] masked_products [0:MAC_COUNT-1];
    
    // 累加树中间结果
    reg signed [ACC_WIDTH-1:0] sum_stage1 [0:15];  // 16 组，每组 16 个
    reg signed [ACC_WIDTH-1:0] sum_stage2 [0:3];   // 4 组，每组 4 个
    reg signed [ACC_WIDTH-1:0] sum_stage3;         // 最终结果
    
    // 状态机
    localparam IDLE = 2'b00;
    localparam DETECT = 2'b01;
    localparam COMPUTE = 2'b10;
    localparam OUTPUT = 2'b11;
    
    reg [1:0] state;
    reg [1:0] compute_stage;
    
    // 稀疏统计计数器
    reg [15:0] skip_count;
    
    // ========================================================================
    // 稀疏检测逻辑
    // ========================================================================
    
    // 输入稀疏检测
    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : input_sparse_detect
            always @(*) begin
                if (input_use_mask) begin
                    input_sparse_mask[i] = input_valid_mask[i];
                end else begin
                    // 检测是否超过阈值（绝对值比较）
                    if (input_data[i] >= 0) begin
                        input_sparse_mask[i] = (input_data[i] >= active_threshold);
                    end else begin
                        input_sparse_mask[i] = (-input_data[i] >= active_threshold);
                    end
                end
            end
        end
    endgenerate
    
    // 权重稀疏检测
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : weight_sparse_detect
            always @(*) begin
                if (weight_use_mask) begin
                    weight_sparse_mask[i] = weight_valid_mask[i];
                end else begin
                    if (weight_data[i] >= 0) begin
                        weight_sparse_mask[i] = (weight_data[i] >= active_threshold);
                    end else begin
                        weight_sparse_mask[i] = (-weight_data[i] >= active_threshold);
                    end
                end
            end
        end
    endgenerate
    
    // 组合掩码：只有两个操作数都非零才计算
    always @(*) begin
        if (sparse_enable) begin
            combined_mask = input_sparse_mask & weight_sparse_mask;
        end else begin
            combined_mask = {MAC_COUNT{1'b1}};  // 非稀疏模式：全部计算
        end
    end
    
    // ========================================================================
    // MAC 单元生成
    // ========================================================================
    
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : mac_units
            // 乘法器
            assign mac_products[i] = input_data[i] * weight_data[i];
            
            // 掩码后的结果（跳过的为 0）
            assign masked_products[i] = combined_mask[i] ? mac_products[i] : 0;
        end
    endgenerate
    
    // ========================================================================
    // 流水线累加树 (3 级)
    // ========================================================================
    
    integer j, k;
    
    // 第一级：16 组并行累加，每组 16 个
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < 16; j = j + 1) begin
                sum_stage1[j] <= 0;
            end
        end else if (state == COMPUTE && compute_stage == 0) begin
            for (j = 0; j < 16; j = j + 1) begin
                sum_stage1[j] <= 0;
                for (k = 0; k < 16; k = k + 1) begin
                    sum_stage1[j] <= sum_stage1[j] + 
                        {{(ACC_WIDTH-2*DATA_WIDTH){masked_products[j*16+k][2*DATA_WIDTH-1]}}, 
                         masked_products[j*16+k]};
                end
            end
        end
    end
    
    // 第二级：4 组累加，每组 4 个第一级结果
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < 4; j = j + 1) begin
                sum_stage2[j] <= 0;
            end
        end else if (state == COMPUTE && compute_stage == 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                sum_stage2[j] <= sum_stage1[j*4] + sum_stage1[j*4+1] + 
                                 sum_stage1[j*4+2] + sum_stage1[j*4+3];
            end
        end
    end
    
    // 第三级：最终累加
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_stage3 <= 0;
        end else if (state == COMPUTE && compute_stage == 2) begin
            sum_stage3 <= sum_stage2[0] + sum_stage2[1] + sum_stage2[2] + sum_stage2[3];
        end
    end
    
    // ========================================================================
    // 主状态机
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            compute_stage <= 0;
            done <= 0;
            busy <= 0;
            result <= 0;
            total_ops <= 0;
            skipped_ops <= 0;
            skip_count <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= DETECT;
                        busy <= 1;
                        compute_stage <= 0;
                    end
                end
                
                DETECT: begin
                    // 统计跳过的操作数
                    skip_count <= 0;
                    for (j = 0; j < MAC_COUNT; j = j + 1) begin
                        if (!combined_mask[j]) begin
                            skip_count <= skip_count + 1;
                        end
                    end
                    state <= COMPUTE;
                end
                
                COMPUTE: begin
                    if (compute_stage < 3) begin
                        compute_stage <= compute_stage + 1;
                    end else begin
                        state <= OUTPUT;
                    end
                end
                
                OUTPUT: begin
                    result <= sum_stage3;
                    total_ops <= MAC_COUNT;
                    skipped_ops <= skip_count;
                    done <= 1;
                    busy <= 0;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // ========================================================================
    // 状态输出
    // ========================================================================
    
    assign sparse_mode_active = sparse_enable;
    
    // 稀疏率计算 (百分比)
    assign sparsity_ratio = (total_ops != 0) ? 
        ((skipped_ops * 100) / total_ops) : 8'd0;

endmodule


// ============================================================================
// 稀疏索引生成器（用于 CSR 格式支持）
// ============================================================================

module sparse_index_generator #(
    parameter MAC_COUNT = 256,
    parameter INDEX_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    
    input wire [MAC_COUNT-1:0] valid_mask,
    input wire start,
    
    output reg [INDEX_WIDTH-1:0] indices [0:MAC_COUNT-1],
    output reg [INDEX_WIDTH-1:0] num_valid,
    output reg done
);

    integer i;
    reg [INDEX_WIDTH-1:0] count;
    reg processing;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            num_valid <= 0;
            done <= 0;
            count <= 0;
            processing <= 0;
            for (i = 0; i < MAC_COUNT; i = i + 1) begin
                indices[i] <= 0;
            end
        end else if (start && !processing) begin
            processing <= 1;
            count <= 0;
            done <= 0;
            // 生成压缩索引
            for (i = 0; i < MAC_COUNT; i = i + 1) begin
                if (valid_mask[i]) begin
                    indices[count] <= i[INDEX_WIDTH-1:0];
                    count <= count + 1;
                end
            end
        end else if (processing) begin
            num_valid <= count;
            done <= 1;
            processing <= 0;
        end else begin
            done <= 0;
        end
    end

endmodule
