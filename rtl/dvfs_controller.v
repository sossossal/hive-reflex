/**
 * DVFS 动态电压频率缩放控制器
 * 根据负载自动调整电压/频率，降低待机功耗至 nW 级
 * 
 * 电压域：Active (1.0V) → Standby (0.6V) → DeepSleep (0.4V)
 * 频率域：100MHz → 50MHz → 10MHz → 1MHz
 * 
 * @file dvfs_controller.v
 * @version 2.1.0
 */

`timescale 1ns / 1ps

module dvfs_controller #(
    parameter UTILIZATION_WINDOW = 16,      // 利用率滑动窗口大小
    parameter TRANSITION_DELAY_US = 10,     // 电压转换延迟 (μs)
    parameter CLK_FREQ_MHZ = 100            // 主时钟频率
)(
    input wire clk,
    input wire rst_n,
    
    // ========================================================================
    // RISC-V 控制接口 (AHB Slave)
    // ========================================================================
    
    input wire [7:0] target_power_mode,     // 目标电源模式
    input wire dvfs_enable,                 // DVFS 使能
    input wire force_mode_valid,            // 强制模式有效
    input wire [1:0] force_voltage,         // 强制电压等级
    input wire [1:0] force_freq,            // 强制频率等级
    
    // ========================================================================
    // 负载监测接口
    // ========================================================================
    
    input wire cim_active,                  // CIM 正在计算
    input wire [15:0] cim_utilization,      // CIM 利用率 (Q8.8, 0-256 = 0-100%)
    
    // 自动缩放阈值
    input wire [7:0] util_threshold_high,   // 高利用率阈值 (默认 200 = 78%)
    input wire [7:0] util_threshold_low,    // 低利用率阈值 (默认 50 = 20%)
    input wire [15:0] idle_timeout_ms,      // 空闲超时进入深度睡眠
    
    // ========================================================================
    // 电压/频率控制输出
    // ========================================================================
    
    output reg [1:0] voltage_level,         // 0=DeepSleep(0.4V), 1=Standby(0.6V), 2=Active(1.0V)
    output reg [1:0] freq_divider,          // 0=/100(1MHz), 1=/10(10MHz), 2=/2(50MHz), 3=/1(100MHz)
    output wire [3:0] freq_div_ratio,       // 实际分频比
    output reg dvfs_ready,                  // DVFS 转换完成
    output reg in_transition,               // 正在转换中
    
    // 时钟门控输出
    output reg clk_gate_enable,             // 时钟门控使能
    output wire gated_clk,                  // 门控后的时钟
    
    // 电压调节器接口 (外部 PMU)
    output reg pmu_voltage_request,         // 电压调节请求
    output reg [7:0] pmu_voltage_target,    // 目标电压 (mV / 10)
    input wire pmu_voltage_stable,          // 电压已稳定
    
    // ========================================================================
    // 状态和统计
    // ========================================================================
    
    output wire [1:0] current_power_mode,   // 当前电源模式
    output reg [31:0] time_in_active,       // Active 模式累计时间 (cycles)
    output reg [31:0] time_in_standby,      // Standby 模式累计时间
    output reg [31:0] time_in_deepsleep,    // DeepSleep 模式累计时间
    output reg [15:0] transition_count,     // 模式切换次数
    
    // 中断
    output reg irq_mode_changed             // 模式变化中断
);

    // ========================================================================
    // 常量定义
    // ========================================================================
    
    // 电源模式
    localparam MODE_DEEPSLEEP = 2'b00;
    localparam MODE_STANDBY   = 2'b01;
    localparam MODE_ACTIVE    = 2'b10;
    
    // 电压值 (mV / 10)
    localparam VOLTAGE_DEEPSLEEP = 8'd40;   // 0.4V
    localparam VOLTAGE_STANDBY   = 8'd60;   // 0.6V
    localparam VOLTAGE_ACTIVE    = 8'd100;  // 1.0V
    
    // 转换延迟周期数
    localparam TRANSITION_CYCLES = (TRANSITION_DELAY_US * CLK_FREQ_MHZ);
    
    // ========================================================================
    // 状态机
    // ========================================================================
    
    localparam S_IDLE           = 3'b000;
    localparam S_CHECK_LOAD     = 3'b001;
    localparam S_SCALE_DOWN     = 3'b010;   // 降压降频
    localparam S_SCALE_UP       = 3'b011;   // 升压升频
    localparam S_WAIT_VOLTAGE   = 3'b100;
    localparam S_WAIT_STABLE    = 3'b101;
    localparam S_DONE           = 3'b110;
    
    reg [2:0] state;
    reg [2:0] next_state;
    
    // ========================================================================
    // 内部信号
    // ========================================================================
    
    reg [1:0] current_mode;
    reg [1:0] target_mode;
    reg [1:0] current_voltage;
    reg [1:0] current_freq;
    reg [1:0] target_voltage;
    reg [1:0] target_freq;
    
    // 利用率滑动窗口
    reg [15:0] util_history [0:UTILIZATION_WINDOW-1];
    reg [19:0] util_sum;
    reg [3:0] util_index;
    wire [15:0] util_average;
    
    // 空闲计时器
    reg [31:0] idle_counter;
    wire idle_timeout;
    
    // 转换延迟计时器
    reg [15:0] transition_timer;
    
    // ========================================================================
    // 利用率滑动平均计算
    // ========================================================================
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            util_sum <= 0;
            util_index <= 0;
            for (i = 0; i < UTILIZATION_WINDOW; i = i + 1) begin
                util_history[i] <= 0;
            end
        end else if (dvfs_enable) begin
            // 更新滑动窗口
            util_sum <= util_sum - util_history[util_index] + cim_utilization;
            util_history[util_index] <= cim_utilization;
            util_index <= (util_index + 1) % UTILIZATION_WINDOW;
        end
    end
    
    assign util_average = util_sum[19:4];  // / UTILIZATION_WINDOW (16)
    
    // ========================================================================
    // 空闲检测
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idle_counter <= 0;
        end else if (cim_active) begin
            idle_counter <= 0;
        end else if (dvfs_enable && (idle_counter < 32'hFFFFFFFF)) begin
            idle_counter <= idle_counter + 1;
        end
    end
    
    // 空闲超时 (假设 100MHz 时钟)
    assign idle_timeout = (idle_counter >= (idle_timeout_ms * CLK_FREQ_MHZ * 1000));
    
    // ========================================================================
    // 目标模式决策
    // ========================================================================
    
    always @(*) begin
        if (force_mode_valid) begin
            // 强制模式
            target_mode = force_voltage;
            target_voltage = force_voltage;
            target_freq = force_freq;
        end else if (!dvfs_enable) begin
            // DVFS 禁用，保持 Active
            target_mode = MODE_ACTIVE;
            target_voltage = MODE_ACTIVE;
            target_freq = 2'b11;
        end else if (idle_timeout) begin
            // 空闲超时，进入深度睡眠
            target_mode = MODE_DEEPSLEEP;
            target_voltage = MODE_DEEPSLEEP;
            target_freq = 2'b00;
        end else if (util_average >= util_threshold_high) begin
            // 高负载，Active 模式
            target_mode = MODE_ACTIVE;
            target_voltage = MODE_ACTIVE;
            target_freq = 2'b11;
        end else if (util_average <= util_threshold_low) begin
            // 低负载，Standby 模式
            target_mode = MODE_STANDBY;
            target_voltage = MODE_STANDBY;
            target_freq = 2'b01;
        end else begin
            // 中等负载，维持当前
            target_mode = current_mode;
            target_voltage = current_voltage;
            target_freq = current_freq;
        end
    end
    
    // ========================================================================
    // 主状态机
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always @(*) begin
        next_state = state;
        
        case (state)
            S_IDLE: begin
                if (target_mode != current_mode) begin
                    next_state = S_CHECK_LOAD;
                end
            end
            
            S_CHECK_LOAD: begin
                if (target_voltage > current_voltage) begin
                    // 需要升压：先升压后升频
                    next_state = S_SCALE_UP;
                end else if (target_voltage < current_voltage) begin
                    // 需要降压：先降频后降压
                    next_state = S_SCALE_DOWN;
                end else if (target_freq != current_freq) begin
                    // 只需调频
                    next_state = S_WAIT_STABLE;
                end else begin
                    next_state = S_DONE;
                end
            end
            
            S_SCALE_DOWN: begin
                // 降频完成后等待电压稳定
                if (transition_timer == 0) begin
                    next_state = S_WAIT_VOLTAGE;
                end
            end
            
            S_SCALE_UP: begin
                // 升压完成后升频
                if (pmu_voltage_stable) begin
                    next_state = S_WAIT_STABLE;
                end
            end
            
            S_WAIT_VOLTAGE: begin
                if (pmu_voltage_stable) begin
                    next_state = S_DONE;
                end
            end
            
            S_WAIT_STABLE: begin
                if (transition_timer == 0) begin
                    next_state = S_DONE;
                end
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end
    
    // ========================================================================
    // 状态机输出控制
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_mode <= MODE_ACTIVE;
            current_voltage <= MODE_ACTIVE;
            current_freq <= 2'b11;
            voltage_level <= MODE_ACTIVE;
            freq_divider <= 2'b11;
            dvfs_ready <= 1;
            in_transition <= 0;
            pmu_voltage_request <= 0;
            pmu_voltage_target <= VOLTAGE_ACTIVE;
            transition_timer <= 0;
            clk_gate_enable <= 0;
            transition_count <= 0;
            irq_mode_changed <= 0;
        end else begin
            irq_mode_changed <= 0;
            
            case (state)
                S_IDLE: begin
                    dvfs_ready <= 1;
                    in_transition <= 0;
                end
                
                S_CHECK_LOAD: begin
                    dvfs_ready <= 0;
                    in_transition <= 1;
                    transition_timer <= TRANSITION_CYCLES[15:0];
                end
                
                S_SCALE_DOWN: begin
                    // 先降频
                    freq_divider <= target_freq;
                    current_freq <= target_freq;
                    
                    if (transition_timer > 0) begin
                        transition_timer <= transition_timer - 1;
                    end else begin
                        // 再降压
                        pmu_voltage_request <= 1;
                        case (target_voltage)
                            MODE_DEEPSLEEP: pmu_voltage_target <= VOLTAGE_DEEPSLEEP;
                            MODE_STANDBY:   pmu_voltage_target <= VOLTAGE_STANDBY;
                            default:        pmu_voltage_target <= VOLTAGE_ACTIVE;
                        endcase
                    end
                end
                
                S_SCALE_UP: begin
                    // 先升压
                    pmu_voltage_request <= 1;
                    case (target_voltage)
                        MODE_ACTIVE:    pmu_voltage_target <= VOLTAGE_ACTIVE;
                        MODE_STANDBY:   pmu_voltage_target <= VOLTAGE_STANDBY;
                        default:        pmu_voltage_target <= VOLTAGE_DEEPSLEEP;
                    endcase
                    transition_timer <= TRANSITION_CYCLES[15:0];
                end
                
                S_WAIT_VOLTAGE: begin
                    if (pmu_voltage_stable) begin
                        pmu_voltage_request <= 0;
                        voltage_level <= target_voltage;
                        current_voltage <= target_voltage;
                    end
                end
                
                S_WAIT_STABLE: begin
                    // 升频
                    freq_divider <= target_freq;
                    current_freq <= target_freq;
                    
                    if (transition_timer > 0) begin
                        transition_timer <= transition_timer - 1;
                    end
                    pmu_voltage_request <= 0;
                end
                
                S_DONE: begin
                    current_mode <= target_mode;
                    voltage_level <= target_voltage;
                    current_voltage <= target_voltage;
                    freq_divider <= target_freq;
                    current_freq <= target_freq;
                    transition_count <= transition_count + 1;
                    irq_mode_changed <= 1;
                end
            endcase
        end
    end
    
    // ========================================================================
    // 时钟门控
    // ========================================================================
    
    // 分频比映射
    assign freq_div_ratio = (freq_divider == 2'b00) ? 4'd100 :
                            (freq_divider == 2'b01) ? 4'd10  :
                            (freq_divider == 2'b10) ? 4'd2   : 4'd1;
    
    // 简化的时钟门控（实际需要用专用门控单元）
    assign gated_clk = clk_gate_enable ? 1'b0 : clk;
    
    // ========================================================================
    // 统计计数器
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            time_in_active <= 0;
            time_in_standby <= 0;
            time_in_deepsleep <= 0;
        end else begin
            case (current_mode)
                MODE_ACTIVE:    time_in_active <= time_in_active + 1;
                MODE_STANDBY:   time_in_standby <= time_in_standby + 1;
                MODE_DEEPSLEEP: time_in_deepsleep <= time_in_deepsleep + 1;
            endcase
        end
    end
    
    // ========================================================================
    // 状态输出
    // ========================================================================
    
    assign current_power_mode = current_mode;

endmodule


// ============================================================================
// 时钟分频器（配合 DVFS 使用）
// ============================================================================

module dvfs_clock_divider (
    input wire clk_in,
    input wire rst_n,
    input wire [1:0] div_select,    // 0=/100, 1=/10, 2=/2, 3=/1
    output reg clk_out
);

    reg [6:0] counter;
    reg [6:0] div_value;
    
    always @(*) begin
        case (div_select)
            2'b00: div_value = 7'd49;   // /100 (计数 0-49)
            2'b01: div_value = 7'd4;    // /10
            2'b10: div_value = 7'd0;    // /2
            2'b11: div_value = 7'd0;    // /1 (直通)
            default: div_value = 7'd0;
        endcase
    end
    
    always @(posedge clk_in or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 0;
            clk_out <= 0;
        end else if (div_select == 2'b11) begin
            // 直通模式
            clk_out <= clk_in;
        end else begin
            if (counter >= div_value) begin
                counter <= 0;
                clk_out <= ~clk_out;
            end else begin
                counter <= counter + 1;
            end
        end
    end

endmodule
