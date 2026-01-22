/**
 * MAC (Multiply-Accumulate) 单元
 * 支持 INT8 乘法
 * 
 * @file mac_unit.v
 * @author Hive-Reflex Team
 * @date 2026-01-19
 */

`timescale 1ns / 1ps

module mac_unit #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // 数据输入
    input wire signed [DATA_WIDTH-1:0] a,       // 激活值
    input wire signed [DATA_WIDTH-1:0] b,       // 权重
    
    // 控制
    input wire enable,
    input wire accumulate,      // 1: 累加, 0: 清零后计算
    
    // 输出
    output reg signed [ACC_WIDTH-1:0] result,
    output reg valid
);

    // 内部寄存器
    reg signed [2*DATA_WIDTH-1:0] product;
    reg signed [ACC_WIDTH-1:0] accumulator;
    
    // 流水线级 1: 乘法
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product <= 0;
        end else if (enable) begin
            product <= a * b;
        end
    end
    
    // 流水线级 2: 累加
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            result <= 0;
            valid <= 0;
        end else if (enable) begin
            if (accumulate) begin
                // 显式符号扩展 product 到 ACC_WIDTH 位
                accumulator <= accumulator + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
            end else begin
                // 显式符号扩展 product 到 ACC_WIDTH 位
                accumulator <= {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
            end
            result <= accumulator;
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end

endmodule
