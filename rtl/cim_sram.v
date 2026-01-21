/**
 * CIM SRAM 双端口存储器
 * 512KB 容量
 * 
 * @file cim_sram.v
 */

`timescale 1ns / 1ps

module cim_sram #(
    parameter ADDR_WIDTH = 17,      // 2^17 * 4 = 512KB
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    
    // Port A - 读写端口 (CPU/DMA 访问)
    input wire [ADDR_WIDTH-1:0] addr_a,
    input wire [DATA_WIDTH-1:0] wdata_a,
    output reg [DATA_WIDTH-1:0] rdata_a,
    input wire we_a,
    input wire en_a,
    
    // Port B - 只读端口 (CIM 访问)
    input wire [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] rdata_b,
    input wire en_b
);

    // SRAM 存储阵列
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    
    // 初始化 (仅用于仿真)
    integer i;
    initial begin
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            mem[i] = 0;
        end
    end
    
    // Port A - 读写
    always @(posedge clk) begin
        if (en_a) begin
            if (we_a) begin
                mem[addr_a] <= wdata_a;
            end
            rdata_a <= mem[addr_a];
        end
    end
    
    // Port B - 只读
    always @(posedge clk) begin
        if (en_b) begin
            rdata_b <= mem[addr_b];
        end
    end

endmodule
