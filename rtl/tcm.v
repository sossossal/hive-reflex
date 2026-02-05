/**
 * @file tcm.v
 * @brief 紧耦合内存 (Tightly Coupled Memory)
 * 
 * 32KB 单周期访问内存,绕过 AHB 总线,消除总线争抢
 * 
 * 用途:
 *   - 中断向量表 (0x0000 - 0x0100)
 *   - 热点代码 (滤波算法, IMU 驱动, 控制循环)
 *   - 关键全局变量
 * 
 * 性能: 单周期读写,无 Wait State
 */

`timescale 1ns / 1ps

module tcm #(
    parameter SIZE = 32768,        // 32KB
    parameter ADDR_WIDTH = 15,     // log2(32768) = 15
    parameter DATA_WIDTH = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,
    
    // TCM 接口 (CPU 专用)
    input  wire                    en,
    input  wire [ADDR_WIDTH-1:0]   addr,      // 字节地址
    input  wire [DATA_WIDTH-1:0]   wdata,
    input  wire                    we,
    input  wire [DATA_WIDTH/8-1:0] be,        // Byte Enable
    output reg  [DATA_WIDTH-1:0]   rdata,
    output wire                    ready
);

    // ========================================================================
    // 内存阵列
    // ========================================================================
    
    localparam MEM_DEPTH = SIZE / (DATA_WIDTH / 8);  // 8192 words
    
    reg [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];
    
    // 字地址 (忽略最低 2 位)
    wire [$clog2(MEM_DEPTH)-1:0] word_addr;
    assign word_addr = addr[ADDR_WIDTH-1:2];
    
    // ========================================================================
    // 读写逻辑
    // ========================================================================
    
    always @(posedge clk) begin
        if (en) begin
            if (we) begin
                // 写操作 (支持字节使能)
                if (be[0]) mem[word_addr][ 7: 0] <= wdata[ 7: 0];
                if (be[1]) mem[word_addr][15: 8] <= wdata[15: 8];
                if (be[2]) mem[word_addr][23:16] <= wdata[23:16];
                if (be[3]) mem[word_addr][31:24] <= wdata[31:24];
            end
            
            // 读操作 (始终执行)
            rdata <= mem[word_addr];
        end
    end
    
    // TCM 永远 ready (单周期访问)
    assign ready = 1'b1;
    
    // ========================================================================
    // 初始化 (可选 - 用于仿真)
    // ========================================================================
    
    `ifdef SIMULATION
    integer i;
    initial begin
        for (i = 0; i < MEM_DEPTH; i = i + 1) begin
            mem[i] = 32'h0;
        end
        $display("[TCM] Initialized %d words (%d KB)", MEM_DEPTH, SIZE/1024);
    end
    `endif
    
    // ========================================================================
    // 性能计数器 (调试用)
    // ========================================================================
    
    `ifdef ENABLE_TCM_STATS
    reg [31:0] read_count;
    reg [31:0] write_count;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_count <= 0;
            write_count <= 0;
        end else if (en) begin
            if (we) write_count <= write_count + 1;
            else    read_count  <= read_count + 1;
        end
    end
    `endif

endmodule
