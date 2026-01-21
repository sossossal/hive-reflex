/**
 * MAC 单元测试平台
 * 
 * @file mac_unit_tb.v
 */

`timescale 1ns / 1ps

module mac_unit_tb;

    // 参数
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz
    
    // 信号
    reg clk;
    reg rst_n;
    reg signed [DATA_WIDTH-1:0] a;
    reg signed [DATA_WIDTH-1:0] b;
    reg enable;
    reg accumulate;
    wire signed [ACC_WIDTH-1:0] result;
    wire valid;
    
    // 实例化 DUT
    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .a(a),
        .b(b),
        .enable(enable),
        .accumulate(accumulate),
        .result(result),
        .valid(valid)
    );
    
    // 时钟生成
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // 测试向量
    integer i;
    reg signed [ACC_WIDTH-1:0] expected;
    integer errors;
    
    initial begin
        $display("========================================");
        $display("MAC Unit Testbench");
        $display("========================================");
        
        // 初始化
        errors = 0;
        rst_n = 0;
        enable = 0;
        accumulate = 0;
        a = 0;
        b = 0;
        
        // 复位
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // 测试用例 1: 简单乘法
        $display("\n[TEST 1] 简单乘法测试");
        a = 8'd5;
        b = 8'd3;
        enable = 1;
        accumulate = 0;
        #(CLK_PERIOD*3);
        
        expected = 15;
        if (result == expected) begin
            $display("  PASS: 5 * 3 = %0d", result);
        end else begin
            $display("  FAIL: 5 * 3 = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 测试用例 2: 累加
        $display("\n[TEST 2] 累加测试");
        accumulate = 1;
        for (i = 0; i < 4; i = i + 1) begin
            a = i;
            b = i;
            #(CLK_PERIOD*2);
        end
        
        expected = 0 + 1 + 4 + 9;  // 0*0 + 1*1 + 2*2 + 3*3
        if (result == expected) begin
            $display("  PASS: 累加结果 = %0d", result);
        end else begin
            $display("  FAIL: 累加结果 = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 测试用例 3: 负数乘法
        $display("\n[TEST 3] 负数乘法");
        accumulate = 0;
        a = -8'd10;
        b = 8'd5;
        #(CLK_PERIOD*3);
        
        expected = -50;
        if (result == expected) begin
            $display("  PASS: -10 * 5 = %0d", result);
        end else begin
            $display("  FAIL: -10 * 5 = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 测试用例 4: 边界值
        $display("\n[TEST 4] 边界值测试");
        a = 8'd127;   // MAX_INT8
        b = 8'd127;
        #(CLK_PERIOD*3);
        
        expected = 16129;
        if (result == expected) begin
            $display("  PASS: 127 * 127 = %0d", result);
        end else begin
            $display("  FAIL: 127 * 127 = %0d (expected %0d)", result, expected);
            errors = errors + 1;
        end
        
        // 总结
        $display("\n========================================");
        if (errors == 0) begin
            $display("✓ 所有测试通过!");
        end else begin
            $display("✗ %0d 个测试失败", errors);
        end
        $display("========================================");
        
        #(CLK_PERIOD*10);
        $finish;
    end
    
    // 波形文件
    initial begin
        $dumpfile("mac_unit_tb.vcd");
        $dumpvars(0, mac_unit_tb);
    end

endmodule
