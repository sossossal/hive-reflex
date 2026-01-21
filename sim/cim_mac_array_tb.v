/**
 * MAC 阵列测试平台
 * 测试 256 个 MAC 的矩阵乘法功能
 * 
 * @file cim_mac_array_tb.v
 */

`timescale 1ns / 1ps

module cim_mac_array_tb;

    parameter MAC_COUNT = 256;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    
    // 信号
    reg clk;
    reg rst_n;
    reg signed [DATA_WIDTH-1:0] input_data [0:MAC_COUNT-1];
    reg signed [DATA_WIDTH-1:0] weight_data [0:MAC_COUNT-1];
    reg start;
    wire done;
    wire signed [ACC_WIDTH-1:0] result;
    
    // DUT 实例化
    cim_mac_array #(
        .MAC_COUNT(MAC_COUNT),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(input_data),
        .weight_data(weight_data),
        .start(start),
        .done(done),
        .result(result)
    );
    
    // 时钟生成
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // 测试
    integer i;
    reg signed [ACC_WIDTH-1:0] expected;
    integer errors;
    
    initial begin
        $display("========================================");
        $display("CIM MAC Array Testbench");
        $display("Testing %0d MACs", MAC_COUNT);
        $display("========================================");
        
        errors = 0;
        rst_n = 0;
        start = 0;
        
        // 初始化输入
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = 0;
            weight_data[i] = 0;
        end
        
        // 复位
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // 测试 1: 全 1 向量点积
        $display("\n[TEST 1] 全 1 向量点积");
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = 1;
            weight_data[i] = 1;
        end
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        wait(done);
        #(CLK_PERIOD);
        
        expected = MAC_COUNT;  // 256 * (1*1)
        if (result == expected) begin
            $display("  PASS: Result = %0d", result);
        end else begin
            $display("  FAIL: Result = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 测试 2: 递增序列
        $display("\n[TEST 2] 递增序列点积");
        for (i = 0; i < 16; i = i + 1) begin
            input_data[i] = i;
            weight_data[i] = i;
        end
        for (i = 16; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = 0;
            weight_data[i] = 0;
        end
        
        #(CLK_PERIOD*2);
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        wait(done);
        #(CLK_PERIOD);
        
        // 0^2 + 1^2 + 2^2 + ... + 15^2 = 1240
        expected = 1240;
        if (result == expected) begin
            $display("  PASS: Result = %0d", result);
        end else begin
            $display("  FAIL: Result = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 测试 3: 负数
        $display("\n[TEST 3] 混合正负数");
        for (i = 0; i < 8; i = i + 1) begin
            input_data[i] = 10;
            weight_data[i] = -5;
        end
        for (i = 8; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = 0;
            weight_data[i] = 0;
        end
        
        #(CLK_PERIOD*2);
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        wait(done);
        #(CLK_PERIOD);
        
        expected = -400;  // 8 * (10 * -5)
        if (result == expected) begin
            $display("  PASS: Result = %0d", result);
        end else begin
            $display("  FAIL: Result = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 总结
        $display("\n========================================");
        if (errors == 0) begin
            $display("✓ 所有测试通过!");
            $display("✓ MAC 阵列功能正确");
        end else begin
            $display("✗ %0d 个测试失败", errors);
        end
        $display("========================================");
        
        #(CLK_PERIOD*10);
        $finish;
    end
    
    // 波形
    initial begin
        $dumpfile("cim_mac_array_tb.vcd");
        $dumpvars(0, cim_mac_array_tb);
    end

endmodule
