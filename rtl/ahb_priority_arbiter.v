/**
 * @file ahb_priority_arbiter.v
 * @brief AHB 总线优先级仲裁器
 * 
 * 解决总线争抢问题,为不同 Master 分配优先级:
 *   Priority 3 (最高): CPU - 中断响应、关键控制
 *   Priority 2 (高):   CIM - 计算结果回写  
 *   Priority 1 (中):   Network - 实时通信
 *   Priority 0 (低):   DMA - 后台数据传输
 * 
 * 效果: CPU 永远不会被低优先级 DMA 阻塞
 */

`timescale 1ns / 1ps

module ahb_priority_arbiter #(
    parameter NUM_MASTERS = 4
)(
    input  wire clk,
    input  wire rst_n,
    
    // 请求信号 (每个 Master 1 bit)
    input  wire [NUM_MASTERS-1:0] req,
    
    // 优先级 (每个 Master 2 bits: 0-3)
    input  wire [NUM_MASTERS*2-1:0] priority,
    
    // 授权信号 (One-hot 编码)
    output reg  [NUM_MASTERS-1:0] grant,
    
    // 获胜的 Master 索引
    output reg  [$clog2(NUM_MASTERS)-1:0] winner
);

    // 优先级定义常量
    localparam PRIO_DMA     = 2'd0;  // 最低
    localparam PRIO_NETWORK = 2'd1;
    localparam PRIO_CIM     = 2'd2;
    localparam PRIO_CPU     = 2'd3;  // 最高

    // ========================================================================
    // 仲裁逻辑
    // ========================================================================
    
    always @(*) begin : arbiter_logic
        integer i;
        integer highest_pri;
        integer winner_idx;
        
        // 初始化
        highest_pri = -1;
        winner_idx = 0;
        grant = 0;
        
        // 找到最高优先级的请求者
        for (i = 0; i < NUM_MASTERS; i = i + 1) begin
            if (req[i]) begin
                integer current_pri;
                current_pri = priority[i*2 +: 2];
                
                if (current_pri > highest_pri) begin
                    highest_pri = current_pri;
                    winner_idx = i;
                end else if (current_pri == highest_pri) begin
                    // 同优先级: 使用轮询 (Round-Robin)
                    // 为了简化, 这里选择索引小的
                    if (i < winner_idx) begin
                        winner_idx = i;
                    end
                end
            end
        end
        
        // 输出授权信号
        if (highest_pri >= 0) begin
            grant[winner_idx] = 1'b1;
            winner = winner_idx;
        end else begin
            winner = 0;
        end
    end

    // ========================================================================
    // 仲裁统计 (可选, 用于调试)
    // ========================================================================
    
    `ifdef ENABLE_ARBITER_STATS
    reg [31:0] grant_count [0:NUM_MASTERS-1];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            integer j;
            for (j = 0; j < NUM_MASTERS; j = j + 1) begin
                grant_count[j] <= 0;
            end
        end else begin
            integer k;
            for (k = 0; k < NUM_MASTERS; k = k + 1) begin
                if (grant[k]) begin
                    grant_count[k] <= grant_count[k] + 1;
                end
            end
        end
    end
    `endif

endmodule
