/**
 * CIM MAC 阵列 RTL 示例
 * 用于 FPGA 验证
 */

`timescale 1ns / 1ps

module cim_mac_array #(
    parameter MAC_COUNT = 256,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // 输入激活值
    input wire signed [DATA_WIDTH-1:0] input_data [0:MAC_COUNT-1],
    
    // 权重
    input wire signed [DATA_WIDTH-1:0] weight_data [0:MAC_COUNT-1],
    
    // 控制
    input wire start,
    output reg done,
    
    // 输出
    output reg signed [ACC_WIDTH-1:0] result
);

    // MAC 单元输出
    wire signed [2*DATA_WIDTH-1:0] mac_products [0:MAC_COUNT-1];
    
    // 生成 MAC 单元
    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : mac_units
            // 简单乘法器
            assign mac_products[i] = input_data[i] * weight_data[i];
        end
    endgenerate
    
    // 累加逻辑
    integer j;
    reg signed [ACC_WIDTH-1:0] sum;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
            done <= 0;
        end else if (start) begin
            // 累加所有乘积
            sum = 0;
            for (j = 0; j < MAC_COUNT; j = j + 1) begin
                sum = sum + mac_products[j];
            end
            result <= sum;
            done <= 1;
        end else begin
            done <= 0;
        end
    end

endmodule
