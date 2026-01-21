/**
 * CIM 控制器
 * 管理 CIM 计算流程，AHB 总线接口
 * 
 * @file cim_controller.v
 */

`timescale 1ns / 1ps

module cim_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // AHB 总线接口 (Slave)
    input wire [ADDR_WIDTH-1:0] haddr,
    input wire [DATA_WIDTH-1:0] hwdata,
    output reg [DATA_WIDTH-1:0] hrdata,
    input wire hwrite,
    input wire [2:0] hsize,
    input wire [1:0] htrans,
    input wire hsel,
    output reg hready,
    output reg hresp,
    
    // CIM MAC 阵列控制
    output reg mac_start,
    input wire mac_done,
    
    // 配置和状态
    output reg [31:0] dim_m,
    output reg [31:0] dim_n,
    output reg [31:0] dim_k,
    output reg [31:0] input_addr,
    output reg [31:0] weight_addr,
    output reg [31:0] output_addr,
    
    // 中断
    output reg irq,
    
    // 稀疏控制接口 (v2.1 新增)
    output reg sparse_enable,           // 启用稀疏模式
    output reg [7:0] sparse_threshold,  // 稀疏阈值
    input wire [15:0] sparse_total_ops, // 总操作数
    input wire [15:0] sparse_skipped_ops, // 跳过操作数
    input wire [7:0] sparse_ratio       // 稀疏率 0-100%
);

    // 寄存器地址定义
    localparam CTRL_REG     = 8'h00;
    localparam STATUS_REG   = 8'h04;
    localparam DIM_M_REG    = 8'h08;
    localparam DIM_N_REG    = 8'h0C;
    localparam DIM_K_REG    = 8'h10;
    localparam INPUT_ADDR_REG  = 8'h14;
    localparam WEIGHT_ADDR_REG = 8'h18;
    localparam OUTPUT_ADDR_REG = 8'h1C;
    
    // 稀疏控制寄存器 (v2.1 新增)
    localparam SPARSE_CTRL_REG = 8'h30;  // 稀疏控制
    localparam SPARSE_STAT_REG = 8'h34;  // 稀疏统计
    localparam SPARSE_THRESH_REG = 8'h38; // 稀疏阈值
    
    // 控制寄存器位定义
    localparam CTRL_START   = 0;
    localparam CTRL_RESET   = 1;
    localparam CTRL_IRQ_EN  = 2;
    
    // 状态寄存器位定义
    localparam STATUS_BUSY  = 0;
    localparam STATUS_DONE  = 1;
    localparam STATUS_ERROR = 2;
    
    // 内部寄存器
    reg [31:0] ctrl_reg;
    reg [31:0] status_reg;
    reg [31:0] sparse_ctrl_reg;  // 稀疏控制寄存器
    
    // 状态机
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        LOAD_DATA   = 3'b001,
        COMPUTE     = 3'b010,
        WRITE_BACK  = 3'b011,
        DONE        = 3'b100,
        ERROR       = 3'b101
    } state_t;
    
    state_t state, next_state;
    
    // AHB 事务处理
    reg [ADDR_WIDTH-1:0] haddr_reg;
    reg hwrite_reg;
    reg hsel_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            haddr_reg <= 0;
            hwrite_reg <= 0;
            hsel_reg <= 0;
        end else if (htrans[1]) begin  // NONSEQ or SEQ
            haddr_reg <= haddr;
            hwrite_reg <= hwrite;
            hsel_reg <= hsel;
        end
    end
    
    // 寄存器读写
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg <= 0;
            sparse_ctrl_reg <= 0;
            sparse_enable <= 0;
            sparse_threshold <= 8'd2;  // 默认阈值 2
            dim_m <= 0;
            dim_n <= 0;
            dim_k <= 0;
            input_addr <= 0;
            weight_addr <= 0;
            output_addr <= 0;
            hrdata <= 0;
            hready <= 1;
            hresp <= 0;
        end else begin
            if (hsel_reg && hready) begin
                if (hwrite_reg) begin
                    // 写操作
                    case (haddr_reg[7:0])
                        CTRL_REG:        ctrl_reg <= hwdata;
                        DIM_M_REG:       dim_m <= hwdata;
                        DIM_N_REG:       dim_n <= hwdata;
                        DIM_K_REG:       dim_k <= hwdata;
                        INPUT_ADDR_REG:  input_addr <= hwdata;
                        WEIGHT_ADDR_REG: weight_addr <= hwdata;
                        OUTPUT_ADDR_REG: output_addr <= hwdata;
                        SPARSE_CTRL_REG: begin
                            sparse_ctrl_reg <= hwdata;
                            sparse_enable <= hwdata[0];  // bit 0: 启用稀疏
                        end
                        SPARSE_THRESH_REG: sparse_threshold <= hwdata[7:0];
                    endcase
                end else begin
                    // 读操作
                    case (haddr_reg[7:0])
                        CTRL_REG:        hrdata <= ctrl_reg;
                        STATUS_REG:      hrdata <= status_reg;
                        DIM_M_REG:       hrdata <= dim_m;
                        DIM_N_REG:       hrdata <= dim_n;
                        DIM_K_REG:       hrdata <= dim_k;
                        INPUT_ADDR_REG:  hrdata <= input_addr;
                        WEIGHT_ADDR_REG: hrdata <= weight_addr;
                        OUTPUT_ADDR_REG: hrdata <= output_addr;
                        SPARSE_CTRL_REG: hrdata <= sparse_ctrl_reg;
                        SPARSE_STAT_REG: hrdata <= {sparse_ratio, sparse_skipped_ops[7:0], sparse_total_ops[7:0]};
                        SPARSE_THRESH_REG: hrdata <= {24'b0, sparse_threshold};
                        default:         hrdata <= 32'h0;
                    endcase
                end
            end
        end
    end
    
    // 状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // 状态机逻辑
    always @(*) begin
        next_state = state;
        mac_start = 0;
        status_reg = 0;
        irq = 0;
        
        case (state)
            IDLE: begin
                if (ctrl_reg[CTRL_START]) begin
                    next_state = LOAD_DATA;
                end
            end
            
            LOAD_DATA: begin
                status_reg[STATUS_BUSY] = 1;
                next_state = COMPUTE;
            end
            
            COMPUTE: begin
                status_reg[STATUS_BUSY] = 1;
                mac_start = 1;
                if (mac_done) begin
                    next_state = WRITE_BACK;
                end
            end
            
            WRITE_BACK: begin
                status_reg[STATUS_BUSY] = 1;
                next_state = DONE;
            end
            
            DONE: begin
                status_reg[STATUS_DONE] = 1;
                if (ctrl_reg[CTRL_IRQ_EN]) begin
                    irq = 1;
                end
                next_state = IDLE;
            end
            
            ERROR: begin
                status_reg[STATUS_ERROR] = 1;
                next_state = IDLE;
            end
        endcase
    end

endmodule
