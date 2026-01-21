/**
 * 稀疏 CIM MAC 阵列测试台
 * 验证稀疏计算功能和正确性
 * 
 * @file sparse_cim_mac_array_tb.v
 */

`timescale 1ns / 1ps

module sparse_cim_mac_array_tb;

    // 参数
    parameter MAC_COUNT = 256;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter CLK_PERIOD = 10;  // 100MHz
    
    // 信号
    reg clk;
    reg rst_n;
    
    reg signed [DATA_WIDTH-1:0] input_data [0:MAC_COUNT-1];
    reg [MAC_COUNT-1:0] input_valid_mask;
    reg input_use_mask;
    
    reg signed [DATA_WIDTH-1:0] weight_data [0:MAC_COUNT-1];
    reg [MAC_COUNT-1:0] weight_valid_mask;
    reg weight_use_mask;
    
    reg start;
    reg sparse_enable;
    reg [DATA_WIDTH-1:0] threshold_config;
    
    wire done;
    wire busy;
    wire signed [ACC_WIDTH-1:0] result;
    wire [15:0] total_ops;
    wire [15:0] skipped_ops;
    wire [7:0] sparsity_ratio;
    wire sparse_mode_active;
    
    // DUT 实例化
    sparse_cim_mac_array #(
        .MAC_COUNT(MAC_COUNT),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SPARSITY_THRESHOLD(2)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(input_data),
        .input_valid_mask(input_valid_mask),
        .input_use_mask(input_use_mask),
        .weight_data(weight_data),
        .weight_valid_mask(weight_valid_mask),
        .weight_use_mask(weight_use_mask),
        .start(start),
        .sparse_enable(sparse_enable),
        .threshold_config(threshold_config),
        .done(done),
        .busy(busy),
        .result(result),
        .total_ops(total_ops),
        .skipped_ops(skipped_ops),
        .sparsity_ratio(sparsity_ratio),
        .sparse_mode_active(sparse_mode_active)
    );
    
    // 时钟生成
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // 测试计数器
    integer test_passed = 0;
    integer test_failed = 0;
    
    // 参考结果计算
    function signed [ACC_WIDTH-1:0] compute_reference;
        input integer threshold;
        input integer use_sparse;
        integer i;
        reg signed [ACC_WIDTH-1:0] sum;
        begin
            sum = 0;
            for (i = 0; i < MAC_COUNT; i = i + 1) begin
                if (use_sparse) begin
                    // 稀疏模式：跳过低于阈值的
                    if ((input_data[i] >= threshold || input_data[i] <= -threshold) &&
                        (weight_data[i] >= threshold || weight_data[i] <= -threshold)) begin
                        sum = sum + input_data[i] * weight_data[i];
                    end
                end else begin
                    sum = sum + input_data[i] * weight_data[i];
                end
            end
            compute_reference = sum;
        end
    endfunction
    
    // 初始化任务
    task reset_dut;
        begin
            rst_n = 0;
            start = 0;
            sparse_enable = 0;
            threshold_config = 0;
            input_use_mask = 0;
            weight_use_mask = 0;
            #(CLK_PERIOD * 5);
            rst_n = 1;
            #(CLK_PERIOD * 2);
        end
    endtask
    
    // 运行计算任务
    task run_compute;
        begin
            start = 1;
            #CLK_PERIOD;
            start = 0;
            // 等待完成
            while (!done) #CLK_PERIOD;
            #CLK_PERIOD;
        end
    endtask
    
    // 测试用例
    integer i;
    reg signed [ACC_WIDTH-1:0] expected_result;
    
    initial begin
        clk = 0;
        
        $display("========================================");
        $display("稀疏 CIM MAC 阵列测试");
        $display("MAC_COUNT = %d, DATA_WIDTH = %d", MAC_COUNT, DATA_WIDTH);
        $display("========================================\n");
        
        // ====================================================================
        // 测试 1: 非稀疏模式 - 全密集数据
        // ====================================================================
        $display("[测试 1] 非稀疏模式 - 全密集数据");
        reset_dut();
        
        // 初始化输入数据
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = (i % 127) + 1;  // 1-127
            weight_data[i] = ((i * 3) % 127) + 1;
        end
        
        sparse_enable = 0;
        run_compute();
        
        expected_result = compute_reference(2, 0);
        
        if (result == expected_result) begin
            $display("  ✓ 通过: 结果 = %d", result);
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 失败: 期望 %d, 实际 %d", expected_result, result);
            test_failed = test_failed + 1;
        end
        
        $display("  统计: 总操作 = %d, 跳过 = %d, 稀疏率 = %d%%\n", 
                 total_ops, skipped_ops, sparsity_ratio);
        
        // ====================================================================
        // 测试 2: 稀疏模式 - 50% 零值输入
        // ====================================================================
        $display("[测试 2] 稀疏模式 - 50%% 零值输入");
        reset_dut();
        
        // 初始化输入数据（50% 零值）
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            if (i % 2 == 0) begin
                input_data[i] = 0;  // 零值
            end else begin
                input_data[i] = (i % 63) + 5;  // 非零
            end
            weight_data[i] = ((i * 7) % 127) + 1;
        end
        
        sparse_enable = 1;
        threshold_config = 2;
        run_compute();
        
        expected_result = compute_reference(2, 1);
        
        if (result == expected_result) begin
            $display("  ✓ 通过: 结果 = %d", result);
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 失败: 期望 %d, 实际 %d", expected_result, result);
            test_failed = test_failed + 1;
        end
        
        $display("  统计: 总操作 = %d, 跳过 = %d, 稀疏率 = %d%%\n", 
                 total_ops, skipped_ops, sparsity_ratio);
        
        if (skipped_ops > 100) begin
            $display("  ✓ 稀疏跳过验证: 跳过操作 > 100");
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 稀疏跳过验证失败: 跳过操作 = %d (期望 > 100)", skipped_ops);
            test_failed = test_failed + 1;
        end
        
        // ====================================================================
        // 测试 3: 稀疏模式 - 80% 稀疏输入
        // ====================================================================
        $display("\n[测试 3] 稀疏模式 - 80%% 稀疏输入");
        reset_dut();
        
        // 初始化输入数据（80% 为低于阈值）
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            if (i % 5 == 0) begin
                input_data[i] = (i % 50) + 10;  // 非零大值
            end else begin
                input_data[i] = 1;  // 低于阈值 2
            end
            weight_data[i] = ((i * 11) % 100) + 5;
        end
        
        sparse_enable = 1;
        threshold_config = 2;
        run_compute();
        
        expected_result = compute_reference(2, 1);
        
        if (result == expected_result) begin
            $display("  ✓ 通过: 结果 = %d", result);
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 失败: 期望 %d, 实际 %d", expected_result, result);
            test_failed = test_failed + 1;
        end
        
        $display("  统计: 总操作 = %d, 跳过 = %d, 稀疏率 = %d%%\n", 
                 total_ops, skipped_ops, sparsity_ratio);
        
        // 验证高稀疏率
        if (sparsity_ratio >= 70) begin
            $display("  ✓ 稀疏率验证: %d%% >= 70%%", sparsity_ratio);
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 稀疏率验证失败: %d%% < 70%%", sparsity_ratio);
            test_failed = test_failed + 1;
        end
        
        // ====================================================================
        // 测试 4: 动态阈值配置
        // ====================================================================
        $display("\n[测试 4] 动态阈值配置");
        reset_dut();
        
        // 初始化输入数据
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = i % 10;  // 0-9
            weight_data[i] = i % 10;
        end
        
        sparse_enable = 1;
        threshold_config = 5;  // 阈值设为 5
        run_compute();
        
        expected_result = compute_reference(5, 1);
        
        if (result == expected_result) begin
            $display("  ✓ 通过 (阈值=5): 结果 = %d", result);
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 失败: 期望 %d, 实际 %d", expected_result, result);
            test_failed = test_failed + 1;
        end
        
        $display("  统计: 总操作 = %d, 跳过 = %d, 稀疏率 = %d%%\n", 
                 total_ops, skipped_ops, sparsity_ratio);
        
        // ====================================================================
        // 测试 5: 稀疏模式 vs 非稀疏模式对比
        // ====================================================================
        $display("\n[测试 5] 模式对比测试");
        reg signed [ACC_WIDTH-1:0] sparse_result, dense_result;
        reg [15:0] sparse_skip, dense_skip;
        
        reset_dut();
        
        // 50% 稀疏数据
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = (i % 2 == 0) ? 0 : 50;
            weight_data[i] = 10;
        end
        
        // 非稀疏模式
        sparse_enable = 0;
        run_compute();
        dense_result = result;
        dense_skip = skipped_ops;
        
        // 稀疏模式
        reset_dut();
        for (i = 0; i < MAC_COUNT; i = i + 1) begin
            input_data[i] = (i % 2 == 0) ? 0 : 50;
            weight_data[i] = 10;
        end
        sparse_enable = 1;
        threshold_config = 1;
        run_compute();
        sparse_result = result;
        sparse_skip = skipped_ops;
        
        $display("  非稀疏模式: 结果 = %d, 跳过 = %d", dense_result, dense_skip);
        $display("  稀疏模式:   结果 = %d, 跳过 = %d", sparse_result, sparse_skip);
        
        // 结果应该相同（零值乘积为 0）
        if (dense_result == sparse_result) begin
            $display("  ✓ 模式对比通过: 结果一致");
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 模式对比失败: 结果不一致");
            test_failed = test_failed + 1;
        end
        
        // 稀疏模式应该有更多跳过
        if (sparse_skip > dense_skip) begin
            $display("  ✓ 稀疏优化有效: 跳过增加 %d", sparse_skip - dense_skip);
            test_passed = test_passed + 1;
        end else begin
            $display("  ✗ 稀疏优化无效");
            test_failed = test_failed + 1;
        end
        
        // ====================================================================
        // 测试结果汇总
        // ====================================================================
        $display("\n========================================");
        $display("测试完成");
        $display("通过: %d, 失败: %d", test_passed, test_failed);
        $display("========================================");
        
        if (test_failed == 0) begin
            $display("\n🎉 所有测试通过!\n");
        end else begin
            $display("\n❌ 存在失败测试\n");
        end
        
        $finish;
    end
    
    // 波形生成
    initial begin
        $dumpfile("sparse_cim_mac_array_tb.vcd");
        $dumpvars(0, sparse_cim_mac_array_tb);
    end

endmodule
