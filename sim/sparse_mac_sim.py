#!/usr/bin/env python3
"""
Hive-Reflex 2.1 稀疏 MAC 行为仿真
用于验证稀疏计算逻辑正确性

使用方法:
    python sparse_mac_sim.py
"""

import numpy as np
import time


def sparse_mac_compute(input_data, weight_data, threshold=2, sparse_enable=True):
    """
    模拟稀疏 MAC 运算
    
    Args:
        input_data: 输入激活值 (int8)
        weight_data: 权重 (int8)
        threshold: 稀疏阈值
        sparse_enable: 是否启用稀疏模式
        
    Returns:
        result: 计算结果
        stats: 统计信息
    """
    total_ops = len(input_data)
    
    if sparse_enable:
        # 创建稀疏掩码
        input_mask = np.abs(input_data) >= threshold
        weight_mask = np.abs(weight_data) >= threshold
        combined_mask = input_mask & weight_mask
        
        # 只计算非零部分
        masked_input = input_data * combined_mask
        masked_weight = weight_data * combined_mask
        result = np.sum(masked_input.astype(np.int32) * masked_weight.astype(np.int32))
        
        skipped_ops = total_ops - np.sum(combined_mask)
    else:
        result = np.sum(input_data.astype(np.int32) * weight_data.astype(np.int32))
        skipped_ops = 0
    
    stats = {
        'total_ops': total_ops,
        'skipped_ops': skipped_ops,
        'sparsity_ratio': skipped_ops / total_ops if total_ops > 0 else 0
    }
    
    return result, stats


def test_dense_mode():
    """测试 1: 非稀疏模式 - 全密集数据"""
    print("[Test 1] Non-sparse mode - Dense data")
    
    MAC_COUNT = 256
    input_data = np.array([(i % 127) + 1 for i in range(MAC_COUNT)], dtype=np.int8)
    weight_data = np.array([((i * 3) % 127) + 1 for i in range(MAC_COUNT)], dtype=np.int8)
    
    result, stats = sparse_mac_compute(input_data, weight_data, sparse_enable=False)
    
    # 参考计算
    expected = np.sum(input_data.astype(np.int32) * weight_data.astype(np.int32))
    
    passed = (result == expected)
    
    print(f"  Result: {result}")
    print(f"  Expected: {expected}")
    print(f"  Sparsity: {stats['sparsity_ratio']*100:.1f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    
    return passed


def test_sparse_50():
    """测试 2: 稀疏模式 - 50% 零值输入"""
    print("[Test 2] Sparse mode - 50% zero inputs")
    
    MAC_COUNT = 256
    THRESHOLD = 2
    
    input_data = np.array([0 if i % 2 == 0 else (i % 63) + 5 for i in range(MAC_COUNT)], dtype=np.int8)
    weight_data = np.array([((i * 7) % 127) + 1 for i in range(MAC_COUNT)], dtype=np.int8)
    
    result, stats = sparse_mac_compute(input_data, weight_data, threshold=THRESHOLD, sparse_enable=True)
    
    # 验证跳过了足够多的操作
    passed = stats['skipped_ops'] > 100
    
    print(f"  Result: {result}")
    print(f"  Skipped: {stats['skipped_ops']} / {stats['total_ops']}")
    print(f"  Sparsity: {stats['sparsity_ratio']*100:.1f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'} (expect skipped > 100)")
    print()
    
    return passed


def test_sparse_80():
    """测试 3: 稀疏模式 - 80% 稀疏输入"""
    print("[Test 3] Sparse mode - 80% sparse inputs")
    
    MAC_COUNT = 256
    THRESHOLD = 2
    
    # 80% 值低于阈值
    input_data = np.array([((i % 50) + 10) if i % 5 == 0 else 1 for i in range(MAC_COUNT)], dtype=np.int8)
    weight_data = np.array([((i * 11) % 100) + 5 for i in range(MAC_COUNT)], dtype=np.int8)
    
    result, stats = sparse_mac_compute(input_data, weight_data, threshold=THRESHOLD, sparse_enable=True)
    
    # 验证稀疏率 >= 70%
    passed = stats['sparsity_ratio'] >= 0.70
    
    print(f"  Result: {result}")
    print(f"  Skipped: {stats['skipped_ops']} / {stats['total_ops']}")
    print(f"  Sparsity: {stats['sparsity_ratio']*100:.1f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'} (expect sparsity >= 70%)")
    print()
    
    return passed


def test_dynamic_threshold():
    """测试 4: 动态阈值配置"""
    print("[Test 4] Dynamic threshold configuration")
    
    MAC_COUNT = 256
    
    input_data = np.array([i % 10 for i in range(MAC_COUNT)], dtype=np.int8)
    weight_data = np.array([i % 10 for i in range(MAC_COUNT)], dtype=np.int8)
    
    # 阈值 = 5
    result, stats = sparse_mac_compute(input_data, weight_data, threshold=5, sparse_enable=True)
    
    # 验证结果正确（只计算 value >= 5 的）
    passed = stats['sparsity_ratio'] > 0.4  # 0-4 被跳过 = 50%
    
    print(f"  Result: {result}")
    print(f"  Threshold: 5")
    print(f"  Sparsity: {stats['sparsity_ratio']*100:.1f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    
    return passed


def test_mode_comparison():
    """测试 5: 模式对比"""
    print("[Test 5] Mode comparison (sparse vs dense)")
    
    MAC_COUNT = 256
    
    # 50% 零值
    input_data = np.array([0 if i % 2 == 0 else 50 for i in range(MAC_COUNT)], dtype=np.int8)
    weight_data = np.array([10 for i in range(MAC_COUNT)], dtype=np.int8)
    
    result_dense, stats_dense = sparse_mac_compute(input_data, weight_data, sparse_enable=False)
    result_sparse, stats_sparse = sparse_mac_compute(input_data, weight_data, threshold=1, sparse_enable=True)
    
    # 结果应该相同（零值乘积为 0）
    results_match = (result_dense == result_sparse)
    sparse_faster = stats_sparse['skipped_ops'] > stats_dense['skipped_ops']
    
    print(f"  Dense result: {result_dense}, skipped: {stats_dense['skipped_ops']}")
    print(f"  Sparse result: {result_sparse}, skipped: {stats_sparse['skipped_ops']}")
    print(f"  Results match: {'YES' if results_match else 'NO'}")
    print(f"  Sparse optimization: {'EFFECTIVE' if sparse_faster else 'NOT EFFECTIVE'}")
    print(f"  Status: {'PASS' if results_match and sparse_faster else 'FAIL'}")
    print()
    
    return results_match and sparse_faster


def run_performance_benchmark():
    """性能基准测试"""
    print("=" * 50)
    print("Performance Benchmark")
    print("=" * 50)
    
    MAC_COUNT = 256
    ITERATIONS = 1000
    
    # 生成测试数据 (50% 稀疏)
    input_data = np.array([0 if i % 2 == 0 else 50 for i in range(MAC_COUNT)], dtype=np.int8)
    weight_data = np.array([10 for i in range(MAC_COUNT)], dtype=np.int8)
    
    # 密集模式
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        sparse_mac_compute(input_data, weight_data, sparse_enable=False)
    dense_time = time.perf_counter() - start
    
    # 稀疏模式
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        sparse_mac_compute(input_data, weight_data, threshold=1, sparse_enable=True)
    sparse_time = time.perf_counter() - start
    
    speedup = dense_time / sparse_time if sparse_time > 0 else 1.0
    
    print(f"  Dense mode: {dense_time*1000:.2f} ms ({ITERATIONS} iterations)")
    print(f"  Sparse mode: {sparse_time*1000:.2f} ms ({ITERATIONS} iterations)")
    print(f"  Speedup: {speedup:.2f}x")
    print()


def main():
    print("=" * 50)
    print("Hive-Reflex 2.1 Sparse MAC Behavioral Simulation")
    print("=" * 50)
    print()
    
    tests = [
        ("Dense Mode", test_dense_mode),
        ("50% Sparse", test_sparse_50),
        ("80% Sparse", test_sparse_80),
        ("Dynamic Threshold", test_dynamic_threshold),
        ("Mode Comparison", test_mode_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))
    
    # 性能测试
    run_performance_benchmark()
    
    # 汇总
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "x"
        print(f"  [{symbol}] {name}: {status}")
    
    print()
    print(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print()
        print("All tests PASSED!")
        return 0
    else:
        print()
        print("Some tests FAILED!")
        return 1


if __name__ == '__main__':
    exit(main())
