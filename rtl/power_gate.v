/**
 * @file power_gate.v
 * @brief 电源门控 (Power Gating) 模块
 * 
 * 通过控制电源开关实现 nW 级 DeepSleep 功耗
 * 
 * 电源门控策略:
 * - 使用 MTCMOS (Multi-Threshold CMOS) 开关
 * - Virtual VDD/VSS 隔离
 * - 状态保持单元 (Retention Cells)
 * 
 * @version 2.1.0
 */

module power_switch #(
    parameter SWITCH_WIDTH = 8      // 开关单元数量
)(
    input wire vdd_main,            // 主电源
    input wire vss,                 // 地
    input wire sleep_n,             // 睡眠控制 (低有效)
    output wire vdd_virtual         // 虚拟电源输出
);

    // =========================================================================
    // MTCMOS 电源开关建模
    // =========================================================================
    // 实际 FPGA 中需要外部 PMU 支持，这里建模逻辑行为
    
    // 开关状态
    reg power_good;
    
    always @(*) begin
        power_good = sleep_n;  // 简化模型
    end
    
    // 虚拟电源 (实际硬件中是模拟开关)
    assign vdd_virtual = power_good ? vdd_main : 1'b0;

endmodule


/**
 * @brief 电源域定义
 */
`define PD_CIM      0   // CIM 计算核心
`define PD_CPU      1   // RISC-V CPU
`define PD_PERIPH   2   // 外设
`define PD_ALWAYS   3   // 始终上电域

`define NUM_POWER_DOMAINS 4


/**
 * @brief 电源域控制器
 * 
 * 管理多个电源域的开关控制
 */
module power_domain_controller #(
    parameter NUM_DOMAINS = `NUM_POWER_DOMAINS
)(
    input wire clk,                 // 始终上电时钟
    input wire rst_n,               // 复位
    
    // 电源模式请求
    input wire [1:0] power_mode,    // 00=DeepSleep, 01=Standby, 10=Active
    
    // 电源域状态
    output reg [NUM_DOMAINS-1:0] domain_power_en,   // 电源使能
    output reg [NUM_DOMAINS-1:0] domain_isolated,   // 隔离状态
    output reg [NUM_DOMAINS-1:0] domain_retained,   // 状态保持
    output reg [NUM_DOMAINS-1:0] domain_clk_en,     // 时钟使能
    
    // PMU 接口
    output reg pmu_request,         // PMU 请求
    output reg [7:0] pmu_voltage,   // 目标电压 (0.4V-1.0V 映射到 0-255)
    input wire pmu_ready,           // PMU 就绪
    
    // 状态
    output reg [1:0] current_mode,
    output reg transition_done
);

    // =========================================================================
    // 状态机
    // =========================================================================
    
    localparam ST_ACTIVE    = 3'd0;
    localparam ST_PRE_SLEEP = 3'd1;
    localparam ST_ISOLATE   = 3'd2;
    localparam ST_POWER_OFF = 3'd3;
    localparam ST_SLEEPING  = 3'd4;
    localparam ST_POWER_ON  = 3'd5;
    localparam ST_DEISOLATE = 3'd6;
    localparam ST_RESTORE   = 3'd7;
    
    reg [2:0] state, next_state;
    reg [15:0] delay_counter;
    
    // 目标电源域配置
    reg [NUM_DOMAINS-1:0] target_power_en;
    reg [NUM_DOMAINS-1:0] target_clk_en;
    
    // 模式配置
    always @(*) begin
        case (power_mode)
            2'b00: begin  // DeepSleep
                target_power_en = 4'b1000;  // 仅始终上电域
                target_clk_en   = 4'b1000;
            end
            2'b01: begin  // Standby
                target_power_en = 4'b1011;  // CPU + Always
                target_clk_en   = 4'b1010;  // 低速时钟
            end
            default: begin  // Active
                target_power_en = 4'b1111;  // 全部
                target_clk_en   = 4'b1111;
            end
        endcase
    end
    
    // =========================================================================
    // 电源域转换序列
    // =========================================================================
    
    // 关闭序列: 保存状态 -> 关闭时钟 -> 隔离 -> 断电
    // 开启序列: 上电 -> 等待稳定 -> 解除隔离 -> 恢复时钟 -> 恢复状态
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_ACTIVE;
            domain_power_en <= {NUM_DOMAINS{1'b1}};
            domain_isolated <= {NUM_DOMAINS{1'b0}};
            domain_retained <= {NUM_DOMAINS{1'b0}};
            domain_clk_en <= {NUM_DOMAINS{1'b1}};
            delay_counter <= 16'd0;
            current_mode <= 2'b10;
            transition_done <= 1'b1;
            pmu_request <= 1'b0;
            pmu_voltage <= 8'd255;  // 1.0V
        end else begin
            state <= next_state;
            
            case (state)
                ST_ACTIVE: begin
                    if (power_mode != current_mode) begin
                        transition_done <= 1'b0;
                    end
                end
                
                ST_PRE_SLEEP: begin
                    // 启用状态保持
                    domain_retained <= ~target_power_en;
                    delay_counter <= 16'd100;
                end
                
                ST_ISOLATE: begin
                    // 关闭时钟并隔离
                    domain_clk_en <= target_clk_en;
                    domain_isolated <= ~target_power_en;
                    delay_counter <= 16'd50;
                end
                
                ST_POWER_OFF: begin
                    // 请求 PMU 降压
                    pmu_request <= 1'b1;
                    case (power_mode)
                        2'b00: pmu_voltage <= 8'd102;   // 0.4V
                        2'b01: pmu_voltage <= 8'd153;   // 0.6V
                        default: pmu_voltage <= 8'd255; // 1.0V
                    endcase
                    
                    if (pmu_ready) begin
                        domain_power_en <= target_power_en;
                        pmu_request <= 1'b0;
                    end
                end
                
                ST_SLEEPING: begin
                    current_mode <= power_mode;
                    transition_done <= 1'b1;
                    
                    // 检查是否需要唤醒
                    if (power_mode != current_mode) begin
                        transition_done <= 1'b0;
                    end
                end
                
                ST_POWER_ON: begin
                    // 请求 PMU 升压
                    pmu_request <= 1'b1;
                    case (power_mode)
                        2'b00: pmu_voltage <= 8'd102;
                        2'b01: pmu_voltage <= 8'd153;
                        default: pmu_voltage <= 8'd255;
                    endcase
                    
                    if (pmu_ready) begin
                        domain_power_en <= target_power_en;
                        pmu_request <= 1'b0;
                        delay_counter <= 16'd200;  // 电压稳定时间
                    end
                end
                
                ST_DEISOLATE: begin
                    // 解除隔离
                    domain_isolated <= {NUM_DOMAINS{1'b0}};
                    delay_counter <= 16'd50;
                end
                
                ST_RESTORE: begin
                    // 恢复时钟和状态
                    domain_clk_en <= target_clk_en;
                    domain_retained <= {NUM_DOMAINS{1'b0}};
                    delay_counter <= 16'd100;
                end
            endcase
            
            // 延迟计数
            if (delay_counter > 0) begin
                delay_counter <= delay_counter - 1;
            end
        end
    end
    
    // 下一状态逻辑
    always @(*) begin
        next_state = state;
        
        case (state)
            ST_ACTIVE: begin
                if (power_mode < current_mode) begin
                    next_state = ST_PRE_SLEEP;
                end else if (power_mode > current_mode) begin
                    next_state = ST_POWER_ON;
                end
            end
            
            ST_PRE_SLEEP: begin
                if (delay_counter == 0) next_state = ST_ISOLATE;
            end
            
            ST_ISOLATE: begin
                if (delay_counter == 0) next_state = ST_POWER_OFF;
            end
            
            ST_POWER_OFF: begin
                if (!pmu_request && domain_power_en == target_power_en) begin
                    next_state = ST_SLEEPING;
                end
            end
            
            ST_SLEEPING: begin
                if (power_mode > current_mode) begin
                    next_state = ST_POWER_ON;
                end else if (power_mode < current_mode) begin
                    next_state = ST_PRE_SLEEP;
                end
            end
            
            ST_POWER_ON: begin
                if (delay_counter == 0 && !pmu_request) begin
                    next_state = ST_DEISOLATE;
                end
            end
            
            ST_DEISOLATE: begin
                if (delay_counter == 0) next_state = ST_RESTORE;
            end
            
            ST_RESTORE: begin
                if (delay_counter == 0) next_state = ST_ACTIVE;
            end
        endcase
    end

endmodule


/**
 * @brief 状态保持寄存器
 * 
 * 在电源门控期间保持关键状态
 */
module retention_register #(
    parameter WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire retain,              // 保持使能
    input wire [WIDTH-1:0] d,       // 数据输入
    output reg [WIDTH-1:0] q        // 数据输出
);

    reg [WIDTH-1:0] shadow;         // 影子寄存器
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q <= {WIDTH{1'b0}};
            shadow <= {WIDTH{1'b0}};
        end else if (retain) begin
            // 保持模式: 使用影子寄存器
            q <= shadow;
        end else begin
            // 正常模式
            q <= d;
            shadow <= d;
        end
    end

endmodule


/**
 * @brief 超低功耗电源控制器顶层
 */
module ultra_low_power_controller (
    input wire clk_always,          // 始终上电时钟 (1MHz)
    input wire rst_n,
    
    // 电源模式
    input wire [1:0] power_mode_request,
    
    // 唤醒源
    input wire [3:0] wakeup_sources,
    input wire [3:0] wakeup_mask,
    
    // PMU 接口
    output wire pmu_voltage_request,
    output wire [7:0] pmu_voltage_target,
    input wire pmu_voltage_stable,
    
    // 电源域输出
    output wire [`NUM_POWER_DOMAINS-1:0] power_domain_en,
    output wire [`NUM_POWER_DOMAINS-1:0] power_domain_isolated,
    output wire [`NUM_POWER_DOMAINS-1:0] power_domain_clk_en,
    
    // 状态
    output wire [1:0] current_power_mode,
    output wire power_transition_done,
    output wire wakeup_pending
);

    // =========================================================================
    // 唤醒检测
    // =========================================================================
    
    wire [3:0] masked_wakeup = wakeup_sources & wakeup_mask;
    assign wakeup_pending = |masked_wakeup;
    
    // 唤醒时自动请求 Active 模式
    wire [1:0] effective_power_mode;
    assign effective_power_mode = wakeup_pending ? 2'b10 : power_mode_request;
    
    // =========================================================================
    // 电源域控制器
    // =========================================================================
    
    wire [3:0] domain_retained;
    
    power_domain_controller u_pdc (
        .clk(clk_always),
        .rst_n(rst_n),
        .power_mode(effective_power_mode),
        .domain_power_en(power_domain_en),
        .domain_isolated(power_domain_isolated),
        .domain_retained(domain_retained),
        .domain_clk_en(power_domain_clk_en),
        .pmu_request(pmu_voltage_request),
        .pmu_voltage(pmu_voltage_target),
        .pmu_ready(pmu_voltage_stable),
        .current_mode(current_power_mode),
        .transition_done(power_transition_done)
    );

endmodule
