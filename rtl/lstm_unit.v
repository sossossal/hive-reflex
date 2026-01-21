/**
 * LSTM 加速单元
 * 硬件实现 LSTM 门控逻辑
 * 
 * @file lstm_unit.v
 */

`timescale 1ns / 1ps

module lstm_unit #(
    parameter DATA_WIDTH = 32,
    parameter HIDDEN_SIZE = 16
)(
    input wire clk,
    input wire rst_n,
    
    // 输入
    input wire [DATA_WIDTH-1:0] input_data,
    input wire [DATA_WIDTH-1:0] h_prev [0:HIDDEN_SIZE-1],
    input wire [DATA_WIDTH-1:0] c_prev [0:HIDDEN_SIZE-1],
    
    // 权重 (预加载)
    input wire [DATA_WIDTH-1:0] w_i [0:HIDDEN_SIZE-1],  // Input gate
    input wire [DATA_WIDTH-1:0] w_f [0:HIDDEN_SIZE-1],  // Forget gate
    input wire [DATA_WIDTH-1:0] w_c [0:HIDDEN_SIZE-1],  // Cell gate
    input wire [DATA_WIDTH-1:0] w_o [0:HIDDEN_SIZE-1],  // Output gate
    
    // 控制
    input wire start,
    output reg done,
    
    // 输出
    output reg [DATA_WIDTH-1:0] h_next [0:HIDDEN_SIZE-1],
    output reg [DATA_WIDTH-1:0] c_next [0:HIDDEN_SIZE-1]
);

    // 内部信号
    reg [DATA_WIDTH-1:0] i_gate [0:HIDDEN_SIZE-1];  // Input gate
    reg [DATA_WIDTH-1:0] f_gate [0:HIDDEN_SIZE-1];  // Forget gate
    reg [DATA_WIDTH-1:0] c_gate [0:HIDDEN_SIZE-1];  // Cell gate
    reg [DATA_WIDTH-1:0] o_gate [0:HIDDEN_SIZE-1];  // Output gate
    
    // 状态
    typedef enum logic [2:0] {
        IDLE,
        COMPUTE_GATES,
        COMPUTE_CELL,
        COMPUTE_HIDDEN,
        DONE_STATE
    } state_t;
    
    state_t state;
    
    integer i;
    
    // 激活函数 (简化版 - 查找表或分段线性)
    function [DATA_WIDTH-1:0] sigmoid;
        input [DATA_WIDTH-1:0] x;
        begin
            // 简化实现: 限幅到 [0, 1]
            if (x > 32'h3F800000) begin  // > 1.0 (FP32)
                sigmoid = 32'h3F800000;  // 1.0
            end else if (x < 0) begin
                sigmoid = 0;
            end else begin
                sigmoid = x;
            end
        end
    endfunction
    
    function [DATA_WIDTH-1:0] tanh_fn;
        input [DATA_WIDTH-1:0] x;
        begin
            // 简化实现: 限幅到 [-1, 1]
            if (x > 32'h3F800000) begin  // > 1.0
                tanh_fn = 32'h3F800000;  // 1.0
            end else if (x < 32'hBF800000) begin  // < -1.0
                tanh_fn = 32'hBF800000;  // -1.0
            end else begin
                tanh_fn = x;
            end
        end
    endfunction
    
    // 主状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                h_next[i] <= 0;
                c_next[i] <= 0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= COMPUTE_GATES;
                    end
                end
                
                COMPUTE_GATES: begin
                    // 计算门控信号
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        i_gate[i] <= sigmoid(w_i[i]);  // 简化
                        f_gate[i] <= sigmoid(w_f[i]);
                        c_gate[i] <= tanh_fn(w_c[i]);
                        o_gate[i] <= sigmoid(w_o[i]);
                    end
                    state <= COMPUTE_CELL;
                end
                
                COMPUTE_CELL: begin
                    // c_next = f * c_prev + i * c_gate
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        c_next[i] <= f_gate[i] * c_prev[i] + i_gate[i] * c_gate[i];
                    end
                    state <= COMPUTE_HIDDEN;
                end
                
                COMPUTE_HIDDEN: begin
                    // h_next = o * tanh(c_next)
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        h_next[i] <= o_gate[i] * tanh_fn(c_next[i]);
                    end
                    state <= DONE_STATE;
                end
                
                DONE_STATE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
