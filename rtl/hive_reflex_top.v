/**
 * Hive-Reflex 2.0 系统顶层
 * RISC-V + CIM + 外设集成
 * 
 * @file hive_reflex_top.v
 */

`timescale 1ns / 1ps

module hive_reflex_top #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    // 系统时钟和复位
    input wire sys_clk_p,
    input wire sys_clk_n,
    input wire sys_rst_n,
    
    // LED 指示灯
    output wire [3:0] led,
    
    // UART
    output wire uart_tx,
    input wire uart_rx,
    
    // 调试接口 (JTAG)
    input wire tck,
    input wire tms,
    input wire tdi,
    output wire tdo
);

    // ========================================================================
    // 时钟和复位
    // ========================================================================
    
    wire clk_100mhz;
    wire rst_n_sync;
    
    // 差分时钟转单端
    IBUFDS clk_ibufds (
        .I(sys_clk_p),
        .IB(sys_clk_n),
        .O(clk_100mhz)
    );
    
    // 复位同步
    reg [3:0] rst_sync_r;
    always @(posedge clk_100mhz or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            rst_sync_r <= 4'b0000;
        end else begin
            rst_sync_r <= {rst_sync_r[2:0], 1'b1};
        end
    end
    assign rst_n_sync = rst_sync_r[3];
    
    // ========================================================================
    // AHB 总线信号
    // ========================================================================
    
    // Master 0: RISC-V CPU
    wire [ADDR_WIDTH-1:0] haddr_m0;
    wire [DATA_WIDTH-1:0] hwdata_m0, hrdata_m0;
    wire hwrite_m0;
    wire [2:0] hsize_m0;
    wire [1:0] htrans_m0;
    wire hready_m0, hresp_m0;
    
    // Slave 0: CIM 核心
    wire [ADDR_WIDTH-1:0] haddr_s0;
    wire [DATA_WIDTH-1:0] hwdata_s0, hrdata_s0;
    wire hwrite_s0;
    wire [2:0] hsize_s0;
    wire [1:0] htrans_s0;
    wire hsel_s0, hready_s0, hresp_s0;
    
    // Slave 1: SRAM
    wire hsel_s1, hready_s1, hresp_s1;
    wire [DATA_WIDTH-1:0] hrdata_s1;
    
    // Slave 2: 外设
    wire hsel_s2, hready_s2, hresp_s2;
    wire [DATA_WIDTH-1:0] hrdata_s2;
    
    // ========================================================================
    // RISC-V 核心 (Rocket Chip 接口)
    // ========================================================================
    
    // TODO: 实例化 Rocket Chip
    // 这里使用简化的占位符
    riscv_cpu_wrapper cpu_inst (
        .clk(clk_100mhz),
        .rst_n(rst_n_sync),
        
        // AHB Master 接口
        .haddr(haddr_m0),
        .hwdata(hwdata_m0),
        .hrdata(hrdata_m0),
        .hwrite(hwrite_m0),
        .hsize(hsize_m0),
        .htrans(htrans_m0),
        .hready(hready_m0),
        .hresp(hresp_m0),
        
        // 调试接口
        .jtag_tck(tck),
        .jtag_tms(tms),
        .jtag_tdi(tdi),
        .jtag_tdo(tdo)
    );
    
    // ========================================================================
    // AHB 总线互联
    // ========================================================================
    
    ahb_interconnect #(
        .NUM_MASTERS(1),
        .NUM_SLAVES(3),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) bus_interconnect (
        .clk(clk_100mhz),
        .rst_n(rst_n_sync),
        
        // Master 0
        .haddr_m({haddr_m0}),
        .hwdata_m({hwdata_m0}),
        .hrdata_m({hrdata_m0}),
        .hwrite_m({hwrite_m0}),
        .hsize_m({hsize_m0}),
        .htrans_m({htrans_m0}),
        .hready_m({hready_m0}),
        .hresp_m({hresp_m0}),
        
        // Slaves
        .haddr_s(haddr_s0),
        .hwdata_s(hwdata_s0),
        .hrdata_s({hrdata_s2, hrdata_s1, hrdata_s0}),
        .hwrite_s(hwrite_s0),
        .hsize_s(hsize_s0),
        .htrans_s(htrans_s0),
        .hsel_s({hsel_s2, hsel_s1, hsel_s0}),
        .hready_s({hready_s2, hready_s1, hready_s0}),
        .hresp_s({hresp_s2, hresp_s1, hresp_s0})
    );
    
    // ========================================================================
    // CIM 核心
    // ========================================================================
    
    wire cim_irq;
    
    cim_core_top cim_inst (
        .clk(clk_100mhz),
        .rst_n(rst_n_sync),
        
        // AHB Slave 接口
        .haddr(haddr_s0),
        .hwdata(hwdata_s0),
        .hrdata(hrdata_s0),
        .hwrite(hwrite_s0),
        .hsize(hsize_s0),
        .htrans(htrans_s0),
        .hsel(hsel_s0),
        .hready(hready_s0),
        .hresp(hresp_s0),
        
        // 中断
        .irq(cim_irq)
    );
    
    // ========================================================================
    // SRAM (512KB 系统 SRAM)
    // ========================================================================
    
    ahb_sram #(
        .ADDR_WIDTH(19),  // 512KB
        .DATA_WIDTH(32)
    ) system_sram (
        .clk(clk_100mhz),
        .rst_n(rst_n_sync),
        
        // AHB Slave 接口
        .haddr(haddr_s0),
        .hwdata(hwdata_s0),
        .hrdata(hrdata_s1),
        .hwrite(hwrite_s0),
        .hsel(hsel_s1),
        .hready(hready_s1),
        .hresp(hresp_s1)
    );
    
    // ========================================================================
    // 外设子系统
    // ========================================================================
    
    peripheral_subsystem peripherals (
        .clk(clk_100mhz),
        .rst_n(rst_n_sync),
        
        // AHB Slave 接口
        .haddr(haddr_s0),
        .hwdata(hwdata_s0),
        .hrdata(hrdata_s2),
        .hwrite(hwrite_s0),
        .hsel(hsel_s2),
        .hready(hready_s2),
        .hresp(hresp_s2),
        
        // 外部接口
        .led(led),
        .uart_tx(uart_tx),
        .uart_rx(uart_rx)
    );

endmodule
