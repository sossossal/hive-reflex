/**
 * AHB 总线互联
 * 支持多 Master 多 Slave
 * 
 * @file ahb_interconnect.v
 */

`timescale 1ns / 1ps

module ahb_interconnect #(
    parameter NUM_MASTERS = 1,
    parameter NUM_SLAVES = 4,
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter ENABLE_PRIORITY = 1  // 启用优先级仲裁
)(
    input wire clk,
    input wire rst_n,
    
    // Master 优先级配置 (每个 Master 2 bits: 0-3)
    // 3=CPU, 2=CIM, 1=Network, 0=DMA
    input wire [NUM_MASTERS*2-1:0] master_priority,
    
    // Master 接口 (Packed Arrays)
    input wire [NUM_MASTERS*ADDR_WIDTH-1:0] haddr_m,
    input wire [NUM_MASTERS*DATA_WIDTH-1:0] hwdata_m,
    output reg [NUM_MASTERS*DATA_WIDTH-1:0] hrdata_m,
    input wire [NUM_MASTERS-1:0] hwrite_m,
    input wire [NUM_MASTERS*3-1:0] hsize_m,
    input wire [NUM_MASTERS*2-1:0] htrans_m,
    output reg [NUM_MASTERS-1:0] hready_m,
    output reg [NUM_MASTERS-1:0] hresp_m,
    
    // Slave 接口 (Output to common slave bus, Input from slaves packed)
    output reg [ADDR_WIDTH-1:0] haddr_s,
    output reg [DATA_WIDTH-1:0] hwdata_s,
    input wire [NUM_SLAVES*DATA_WIDTH-1:0] hrdata_s,
    output reg hwrite_s,
    output reg [2:0] hsize_s,
    output reg [1:0] htrans_s,
    output reg [NUM_SLAVES-1:0] hsel_s,
    input wire [NUM_SLAVES-1:0] hready_s,
    input wire [NUM_SLAVES-1:0] hresp_s
);

    // 地址映射
    localparam ADDR_CIM    = 32'h5000_0000;  // Slave 0: CIM
    localparam ADDR_SRAM   = 32'h2000_0000;  // Slave 1: SRAM
    localparam ADDR_PERIPH = 32'h4000_0000;  // Slave 2: 外设
    localparam ADDR_NET    = 32'h6000_0000;  // Slave 3: 网络控制器

    // 仲裁器
    wire [NUM_MASTERS-1:0] master_req;
    wire [NUM_MASTERS-1:0] master_grant;
    wire [$clog2(NUM_MASTERS)-1:0] granted_master_idx;
    integer granted_master;
    integer selected_slave;
    
    // 生成请求信号
    genvar i;
    generate
        for (i = 0; i < NUM_MASTERS; i = i + 1) begin : gen_req
            assign master_req[i] = (htrans_m[i*2 +: 2] != 2'b00);
        end
    endgenerate
    
    // 地址译码
    function integer decode_address;
        input [ADDR_WIDTH-1:0] addr;
        begin
            casez (addr[31:28])
                4'h5: decode_address = 0;  // CIM
                4'h2: decode_address = 1;  // SRAM
                4'h4: decode_address = 2;  // 外设
                4'h6: decode_address = 3;  // 网络控制器
                4'h7: decode_address = 4;  // AHB-DMA (Addr 0x7000_0000)
                default: decode_address = 1;  // 默认 SRAM
            endcase
        end
    endfunction
    
    // 优先级仲裁器实例化
    generate
        if (ENABLE_PRIORITY) begin : gen_priority_arbiter
            ahb_priority_arbiter #(
                .NUM_MASTERS(NUM_MASTERS)
            ) arbiter (
                .clk(clk),
                .rst_n(rst_n),
                .req(master_req),
                .priority(master_priority),
                .grant(master_grant),
                .winner(granted_master_idx)
            );
            
            assign granted_master = granted_master_idx;
        end else begin : gen_simple_arbiter
            // 简单仲裁逻辑 (向后兼容)
            reg [$clog2(NUM_MASTERS)-1:0] granted_master_reg;
            
            always @(*) begin
                granted_master_reg = 0;
                begin : find_master
                    integer found;
                    found = 0;
                    for (integer j = 0; j < NUM_MASTERS; j = j + 1) begin
                        if (found == 0 && htrans_m[j*2 +: 2] != 2'b00) begin
                            granted_master_reg = j;
                            found = 1;
                        end
                    end
                end
            end
            
            assign granted_master = granted_master_reg;
        end
    endgenerate
    
    // 地址相位
    always @(*) begin
        // 默认值
        haddr_s = 0;
        hwdata_s = 0;
        hwrite_s = 0;
        hsize_s = 0;
        htrans_s = 2'b00;
        
        hsel_s = 0;
        
        // 从获胜的 master 路由信号
        haddr_s = haddr_m[granted_master*ADDR_WIDTH +: ADDR_WIDTH];
        hwdata_s = hwdata_m[granted_master*DATA_WIDTH +: DATA_WIDTH];
        hwrite_s = hwrite_m[granted_master];
        hsize_s = hsize_m[granted_master*3 +: 3];
        htrans_s = htrans_m[granted_master*2 +: 2];
        
        // 选择 slave
        selected_slave = decode_address(haddr_s);
        hsel_s[selected_slave] = (htrans_s != 2'b00);
    end
    
    // 数据相位 - 路由返回数据
    always @(*) begin
        hrdata_m = 0;
        hready_m = {NUM_MASTERS{1'b1}}; // Default ready
        hresp_m = 0;
        
        hrdata_m[granted_master*DATA_WIDTH +: DATA_WIDTH] = hrdata_s[selected_slave*DATA_WIDTH +: DATA_WIDTH];
        hready_m[granted_master] = hready_s[selected_slave];
        hresp_m[granted_master] = hresp_s[selected_slave];
    end

endmodule
