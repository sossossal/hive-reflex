/**
 * @file clock_gate.v
 * @brief 时钟门控单元 - 用于 DeepSleep 功耗优化
 * 
 * 实现集成时钟门控 (ICG) 以在空闲时关闭时钟，
 * 从而将动态功耗降至接近零
 * 
 * 时钟门控策略:
 * - DeepSleep: 完全关闭时钟 (仅保留唤醒逻辑)
 * - Standby: 仅保留关键路径时钟
 * - Active: 全时钟运行
 * 
 * @version 2.1.0
 */

module clock_gate #(
    parameter ENABLE_LATCH = 1  // 1 = 使用锁存器防止毛刺, 0 = 直接门控
)(
    input wire clk_in,          // 输入时钟
    input wire rst_n,           // 复位
    input wire enable,          // 门控使能
    input wire scan_mode,       // 扫描模式 (测试时绕过门控)
    output wire clk_out         // 输出时钟
);

    // =========================================================================
    // 锁存器式时钟门控 (推荐，无毛刺)
    // =========================================================================
    
    generate
        if (ENABLE_LATCH) begin : gen_latch_icg
            
            reg enable_latch;
            
            // 负沿锁存使能信号
            always @(clk_in or enable or rst_n) begin
                if (!rst_n) begin
                    enable_latch <= 1'b0;
                end else if (!clk_in) begin
                    enable_latch <= enable;
                end
            end
            
            // 与门生成门控时钟
            assign clk_out = (enable_latch | scan_mode) & clk_in;
            
        end else begin : gen_direct_icg
            
            // 直接门控 (简单但可能有毛刺)
            assign clk_out = (enable | scan_mode) & clk_in;
            
        end
    endgenerate

endmodule


/**
 * @brief 多级时钟门控控制器
 * 
 * 根据电源模式控制各模块时钟
 */
module clock_gate_controller #(
    parameter NUM_DOMAINS = 4   // 时钟域数量
)(
    input wire clk_master,      // 主时钟
    input wire rst_n,           // 复位
    
    // 电源模式 (00=DeepSleep, 01=Standby, 10=Active)
    input wire [1:0] power_mode,
    
    // 模块活动信号
    input wire cim_active,      // CIM 计算中
    input wire dvfs_busy,       // DVFS 转换中
    input wire uart_active,     // UART 通信中
    input wire wakeup_pending,  // 唤醒待处理
    
    // 扫描模式
    input wire scan_mode,
    
    // 门控时钟输出
    output wire clk_cim,        // CIM 时钟
    output wire clk_dvfs,       // DVFS 时钟
    output wire clk_uart,       // UART 时钟
    output wire clk_always      // 始终运行时钟 (唤醒逻辑)
);

    // =========================================================================
    // 电源模式常量
    // =========================================================================
    
    localparam MODE_DEEPSLEEP = 2'b00;
    localparam MODE_STANDBY   = 2'b01;
    localparam MODE_ACTIVE    = 2'b10;
    
    // =========================================================================
    // 门控使能逻辑
    // =========================================================================
    
    reg cim_clk_en;
    reg dvfs_clk_en;
    reg uart_clk_en;
    
    always @(*) begin
        case (power_mode)
            MODE_DEEPSLEEP: begin
                // DeepSleep: 几乎所有时钟关闭
                cim_clk_en  = 1'b0;
                dvfs_clk_en = wakeup_pending;  // 仅唤醒时开启
                uart_clk_en = 1'b0;
            end
            
            MODE_STANDBY: begin
                // Standby: 仅活动模块时钟
                cim_clk_en  = cim_active;
                dvfs_clk_en = dvfs_busy;
                uart_clk_en = uart_active;
            end
            
            default: begin  // MODE_ACTIVE
                // Active: 所有时钟运行
                cim_clk_en  = 1'b1;
                dvfs_clk_en = 1'b1;
                uart_clk_en = 1'b1;
            end
        endcase
    end
    
    // =========================================================================
    // 时钟门控实例
    // =========================================================================
    
    clock_gate #(.ENABLE_LATCH(1)) u_cg_cim (
        .clk_in(clk_master),
        .rst_n(rst_n),
        .enable(cim_clk_en),
        .scan_mode(scan_mode),
        .clk_out(clk_cim)
    );
    
    clock_gate #(.ENABLE_LATCH(1)) u_cg_dvfs (
        .clk_in(clk_master),
        .rst_n(rst_n),
        .enable(dvfs_clk_en),
        .scan_mode(scan_mode),
        .clk_out(clk_dvfs)
    );
    
    clock_gate #(.ENABLE_LATCH(1)) u_cg_uart (
        .clk_in(clk_master),
        .rst_n(rst_n),
        .enable(uart_clk_en),
        .scan_mode(scan_mode),
        .clk_out(clk_uart)
    );
    
    // 始终运行时钟 (仅用于唤醒逻辑，极低功耗)
    // 在 DeepSleep 时使用极低速时钟 (1MHz)
    assign clk_always = clk_master;  // 实际应用中应连接到 1MHz 时钟

endmodule


/**
 * @brief 超低功耗唤醒控制器
 * 
 * 在 DeepSleep 模式下运行，监测唤醒事件
 * 使用极低时钟和最小逻辑
 */
module wakeup_controller (
    input wire clk_slow,        // 慢速时钟 (1MHz 或更低)
    input wire rst_n,           // 复位
    
    // 唤醒源
    input wire wakeup_gpio,     // GPIO 唤醒
    input wire wakeup_uart_rx,  // UART RX 活动
    input wire wakeup_timer,    // 定时器唤醒
    input wire wakeup_can,      // CAN 总线活动
    
    // 唤醒配置
    input wire [3:0] wakeup_mask,   // 唤醒源使能掩码
    input wire [15:0] debounce_cnt, // 去抖计数
    
    // 输出
    output reg wakeup_request,      // 唤醒请求
    output reg [3:0] wakeup_source  // 唤醒源标识
);

    // =========================================================================
    // 唤醒源去抖
    // =========================================================================
    
    wire [3:0] raw_wakeup = {wakeup_can, wakeup_timer, wakeup_uart_rx, wakeup_gpio};
    wire [3:0] masked_wakeup = raw_wakeup & wakeup_mask;
    
    reg [3:0] wakeup_sync [0:1];
    reg [15:0] debounce_counter;
    reg wakeup_detected;
    
    // 同步器
    always @(posedge clk_slow or negedge rst_n) begin
        if (!rst_n) begin
            wakeup_sync[0] <= 4'b0;
            wakeup_sync[1] <= 4'b0;
        end else begin
            wakeup_sync[0] <= masked_wakeup;
            wakeup_sync[1] <= wakeup_sync[0];
        end
    end
    
    // 边沿检测和去抖
    wire [3:0] wakeup_edge = wakeup_sync[0] & ~wakeup_sync[1];
    wire any_wakeup = |wakeup_edge;
    
    always @(posedge clk_slow or negedge rst_n) begin
        if (!rst_n) begin
            debounce_counter <= 16'd0;
            wakeup_detected <= 1'b0;
            wakeup_source <= 4'b0;
        end else begin
            if (any_wakeup) begin
                debounce_counter <= debounce_cnt;
                wakeup_source <= wakeup_edge;
            end else if (debounce_counter > 0) begin
                debounce_counter <= debounce_counter - 1;
                if (debounce_counter == 1) begin
                    wakeup_detected <= 1'b1;
                end
            end
            
            if (wakeup_request) begin
                wakeup_detected <= 1'b0;
            end
        end
    end
    
    always @(posedge clk_slow or negedge rst_n) begin
        if (!rst_n) begin
            wakeup_request <= 1'b0;
        end else begin
            wakeup_request <= wakeup_detected;
        end
    end

endmodule
