/**
 * AHB 总线互联
 * 支持多 Master 多 Slave
 * 
 * @file ahb_interconnect.v
 */

`timescale 1ns / 1ps

module ahb_interconnect #(
    parameter NUM_MASTERS = 1,
    parameter NUM_SLAVES = 3,
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // Master 接口 (数组)
    input wire [ADDR_WIDTH-1:0] haddr_m [0:NUM_MASTERS-1],
    input wire [DATA_WIDTH-1:0] hwdata_m [0:NUM_MASTERS-1],
    output reg [DATA_WIDTH-1:0] hrdata_m [0:NUM_MASTERS-1],
    input wire hwrite_m [0:NUM_MASTERS-1],
    input wire [2:0] hsize_m [0:NUM_MASTERS-1],
    input wire [1:0] htrans_m [0:NUM_MASTERS-1],
    output reg hready_m [0:NUM_MASTERS-1],
    output reg hresp_m [0:NUM_MASTERS-1],
    
    // Slave 接口 (输出到所有 slave)
    output reg [ADDR_WIDTH-1:0] haddr_s,
    output reg [DATA_WIDTH-1:0] hwdata_s,
    input wire [DATA_WIDTH-1:0] hrdata_s [0:NUM_SLAVES-1],
    output reg hwrite_s,
    output reg [2:0] hsize_s,
    output reg [1:0] htrans_s,
    output reg hsel_s [0:NUM_SLAVES-1],
    input wire hready_s [0:NUM_SLAVES-1],
    input wire hresp_s [0:NUM_SLAVES-1]
);

    // 地址映射
    localparam ADDR_CIM    = 32'h5000_0000;  // Slave 0: CIM
    localparam ADDR_SRAM   = 32'h2000_0000;  // slave 1: SRAM
    localparam ADDR_PERIPH = 32'h4000_0000;  // Slave 2: 外设
    
    // 仲裁器 - 固定优先级
    integer granted_master;
    integer selected_slave;
    
    // 地址译码
    function integer decode_address;
        input [ADDR_WIDTH-1:0] addr;
        begin
            casez (addr[31:28])
                4'h5: decode_address = 0;  // CIM
                4'h2: decode_address = 1;  // SRAM
                4'h4: decode_address = 2;  // 外设
                default: decode_address = 1;  // 默认 SRAM
            endcase
        end
    endfunction
    
    // 仲裁逻辑
    always @(*) begin
        granted_master = 0;
        
        // 简单优先级: Master 0 优先
        for (integer i = 0; i < NUM_MASTERS; i = i + 1) begin
            if (htrans_m[i] != 2'b00) begin  // IDLE
                granted_master = i;
                break;
            end
        end
    end
    
    // 地址相位
    always @(*) begin
        // 默认值
        haddr_s = 0;
        hwdata_s = 0;
        hwrite_s = 0;
        hsize_s = 0;
        htrans_s = 2'b00;
        
        for (integer i = 0; i < NUM_SLAVES; i = i + 1) begin
            hsel_s[i] = 0;
        end
        
        // 从获胜的 master 路由信号
        haddr_s = haddr_m[granted_master];
        hwdata_s = hwdata_m[granted_master];
        hwrite_s = hwrite_m[granted_master];
        hsize_s = hsize_m[granted_master];
        htrans_s = htrans_m[granted_master];
        
        // 选择 slave
        selected_slave = decode_address(haddr_s);
        hsel_s[selected_slave] = (htrans_s != 2'b00);
    end
    
    // 数据相位 - 路由返回数据
    always @(*) begin
        for (integer i = 0; i < NUM_MASTERS; i = i + 1) begin
            if (i == granted_master) begin
                hrdata_m[i] = hrdata_s[selected_slave];
                hready_m[i] = hready_s[selected_slave];
                hresp_m[i] = hresp_s[selected_slave];
            end else begin
                hrdata_m[i] = 0;
                hready_m[i] = 1;
                hresp_m[i] = 0;
            end
        end
    end

endmodule
