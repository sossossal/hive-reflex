/**
 * CIM 核心顶层模块
 * 集成 MAC 阵列、SRAM、控制器和 LSTM
 * 
 * @file cim_core_top.v
 */

`timescale 1ns / 1ps

module cim_core_top #(
    parameter MAC_COUNT = 256,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter SRAM_ADDR_WIDTH = 17,
    parameter SRAM_DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // AHB 总线接口
    input wire [31:0] haddr,
    input wire [31:0] hwdata,
    output wire [31:0] hrdata,
    input wire hwrite,
    input wire [2:0] hsize,
    input wire [1:0] htrans,
    input wire hsel,
    output wire hready,
    output wire hresp,
    
    // 中断
    output wire irq,
    
    // 硬件触发
    input wire trigger_in
);

    // 内部信号
    wire mac_start, mac_done;
    wire signed [ACC_WIDTH-1:0] mac_result;
    
    // CIM SRAM 信号
    wire [SRAM_ADDR_WIDTH-1:0] sram_addr_a, sram_addr_b;
    wire [SRAM_DATA_WIDTH-1:0] sram_wdata_a, sram_rdata_a, sram_rdata_b;
    wire sram_we_a, sram_en_a, sram_en_b;
    
    // MAC 阵列数据
    wire signed [DATA_WIDTH-1:0] input_data [0:MAC_COUNT-1];
    wire signed [DATA_WIDTH-1:0] weight_data [0:MAC_COUNT-1];
    
    // 配置信号
    wire [31:0] dim_m, dim_n, dim_k;
    wire [31:0] input_addr, weight_addr, output_addr;
    
    // ========================================================================
    // CIM 控制器
    // ========================================================================
    cim_controller controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // AHB 接口
        .haddr(haddr),
        .hwdata(hwdata),
        .hrdata(hrdata),
        .hwrite(hwrite),
        .hsize(hsize),
        .htrans(htrans),
        .hsel(hsel),
        .hready(hready),
        .hresp(hresp),
        
        // MAC 控制
        .mac_start(mac_start),
        .mac_done(mac_done),
        
        // 配置
        .dim_m(dim_m),
        .dim_n(dim_n),
        .dim_k(dim_k),
        .input_addr(input_addr),
        .weight_addr(weight_addr),
        .output_addr(output_addr),
        
        // 中断
        .irq(irq),
        
        // 触发
        .trigger_in(trigger_in)
    );
    
    // ========================================================================
    // MAC 阵列
    // ========================================================================
    cim_mac_array #(
        .MAC_COUNT(MAC_COUNT),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) mac_array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(input_data),
        .weight_data(weight_data),
        .start(mac_start),
        .done(mac_done),
        .result(mac_result)
    );
    
    // ========================================================================
    // CIM SRAM
    // ========================================================================
    cim_sram #(
        .ADDR_WIDTH(SRAM_ADDR_WIDTH),
        .DATA_WIDTH(SRAM_DATA_WIDTH)
    ) sram_inst (
        .clk(clk),
        
        // Port A (CPU/DMA)
        .addr_a(sram_addr_a),
        .wdata_a(sram_wdata_a),
        .rdata_a(sram_rdata_a),
        .we_a(sram_we_a),
        .en_a(sram_en_a),
        
        // Port B (CIM)
        .addr_b(sram_addr_b),
        .rdata_b(sram_rdata_b),
        .en_b(sram_en_b)
    );
    
    // ========================================================================
    // 数据通路逻辑
    // ========================================================================
    
    // 从 SRAM 加载数据到 MAC 阵列
    // (简化版 - 实际需要状态机控制)
    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : data_path
            // 输入数据从 SRAM Port B 读取
            assign input_data[i] = sram_rdata_b[DATA_WIDTH-1:0];
            
            // 权重数据从 SRAM 读取
            assign weight_data[i] = sram_rdata_b[DATA_WIDTH-1:0];
        end
    endgenerate
    
    // SRAM 地址控制 (简化)
    assign sram_addr_b = input_addr[SRAM_ADDR_WIDTH-1:0];
    assign sram_en_b = mac_start;
    
endmodule
