#!/usr/bin/env python3
"""
Flash IO 优化性能测试工具
用于离线压缩模型权重和生成性能报告
"""

import argparse
import struct
import time
from pathlib import Path
from typing import Tuple, List
import json

# 简化的压缩算法实现（实际应使用 lz4/zlib 等库）
class ModelCompressor:
    """模型权重压缩工具"""
    
    @staticmethod
    def rle_compress(data: bytes) -> Tuple[bytes, float]:
        """RLE 压缩（适用于稀疏权重）"""
        compressed = bytearray()
        i = 0
        while i < len(data):
            count = 1
            while i + count < len(data) and data[i] == data[i + count] and count < 255:
                count += 1
            compressed.append(count)
            compressed.append(data[i])
            i += count
        
        ratio = len(data) / len(compressed) if compressed else 0
        return bytes(compressed), ratio
    
    @staticmethod
    def delta_compress(data: bytes) -> Tuple[bytes, float]:
        """Delta 编码（适用于平滑变化的权重）"""
        if len(data) == 0:
            return b'', 0
        
        compressed = bytearray([data[0]])
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) & 0xFF
            compressed.append(delta)
        
        ratio = len(data) / len(compressed) if compressed else 0
        return bytes(compressed), ratio
    
    @staticmethod
    def compress_layer(layer_data: bytes, algorithm: str) -> Tuple[bytes, float, str]:
        """
        压缩单层权重
        
        Returns:
            (compressed_data, compression_ratio, algorithm_used)
        """
        if algorithm == 'rle':
            compressed, ratio = ModelCompressor.rle_compress(layer_data)
            return compressed, ratio, 'RLE'
        elif algorithm == 'delta':
            compressed, ratio = ModelCompressor.delta_compress(layer_data)
            return compressed, ratio, 'DELTA'
        elif algorithm == 'auto':
            # 自动选择最佳算法
            rle_data, rle_ratio = ModelCompressor.rle_compress(layer_data)
            delta_data, delta_ratio = ModelCompressor.delta_compress(layer_data)
            
            if rle_ratio > delta_ratio:
                return rle_data, rle_ratio, 'RLE'
            else:
                return delta_data, delta_ratio, 'DELTA'
        else:
            # 无压缩
            return layer_data, 1.0, 'NONE'


class FlashIOBenchmark:
    """Flash IO 性能测试"""
    
    def __init__(self, num_layers: int = 8):
        self.num_layers = num_layers
        self.layer_sizes = [
            128*1024, 64*1024, 64*1024, 32*1024,
            32*1024, 16*1024, 16*1024, 8*1024
        ][:num_layers]
        
    def simulate_flash_read(self, size_bytes: int) -> float:
        """模拟 Flash 读取时间（100MB/s）"""
        bandwidth_mbps = 100
        return (size_bytes / (bandwidth_mbps * 1024 * 1024)) * 1000  # ms
    
    def simulate_cim_compute(self, layer_idx: int) -> float:
        """模拟 CIM 计算时间（假设每层 5ms）"""
        return 5.0  # ms
    
    def simulate_decompress(self, compressed_size: int, algorithm: str) -> float:
        """模拟解压时间"""
        speeds = {
            'RLE': 500,    # MB/s
            'LZ4': 300,
            'DELTA': 250,
            'HUFFMAN': 150
        }
        speed = speeds.get(algorithm, 500)
        return (compressed_size / (speed * 1024 * 1024)) * 1000  # ms
    
    def test_baseline(self) -> dict:
        """测试基线性能（无优化）"""
        total_time = 0
        total_flash = 0
        
        for i in range(self.num_layers):
            flash_time = self.simulate_flash_read(self.layer_sizes[i])
            compute_time = self.simulate_cim_compute(i)
            total_time += flash_time + compute_time
            total_flash += self.layer_sizes[i]
        
        return {
            'strategy': 'Baseline',
            'time_ms': total_time,
            'flash_bytes': total_flash,
            'speedup': 1.0,
            'flash_saved': 0
        }
    
    def test_pipeline(self) -> dict:
        """测试流水线优化"""
        total_time = 0
        total_flash = 0
        
        # Layer 0: 串行（初始加载）
        total_time += self.simulate_flash_read(self.layer_sizes[0])
        total_time += self.simulate_cim_compute(0)
        total_flash += self.layer_sizes[0]
        
        # Layer 1-N: 流水线（Flash 读取与 CIM 并行）
        for i in range(1, self.num_layers):
            flash_time = self.simulate_flash_read(self.layer_sizes[i])
            compute_time = self.simulate_cim_compute(i)
            # 仅计入较长的时间（并行执行）
            total_time += max(flash_time, compute_time)
            total_flash += self.layer_sizes[i]
        
        baseline = self.test_baseline()
        return {
            'strategy': 'Pipeline',
            'time_ms': total_time,
            'flash_bytes': total_flash,
            'speedup': baseline['time_ms'] / total_time,
            'flash_saved': 0
        }
    
    def test_compression(self, compression_ratio: float = 2.0, algorithm: str = 'LZ4') -> dict:
        """测试压缩优化"""
        total_time = 0
        total_flash = 0
        
        for i in range(self.num_layers):
            compressed_size = int(self.layer_sizes[i] / compression_ratio)
            flash_time = self.simulate_flash_read(compressed_size)
            decompress_time = self.simulate_decompress(compressed_size, algorithm)
            compute_time = self.simulate_cim_compute(i)
            
            total_time += flash_time + decompress_time + compute_time
            total_flash += compressed_size
        
        baseline = self.test_baseline()
        return {
            'strategy': f'Compression ({algorithm}, {compression_ratio}x)',
            'time_ms': total_time,
            'flash_bytes': total_flash,
            'speedup': baseline['time_ms'] / total_time,
            'flash_saved': int((1 - 1/compression_ratio) * 100)
        }
    
    def test_cascade(self, early_exit_ratio: float = 0.7, exit_layer: int = 2) -> dict:
        """测试级联模型优化"""
        # 70% 的推理在 Layer 2 就退出
        early_time = 0
        early_flash = 0
        for i in range(exit_layer + 1):
            early_time += self.simulate_flash_read(self.layer_sizes[i])
            early_time += self.simulate_cim_compute(i)
            early_flash += self.layer_sizes[i]
        
        # 30% 需要完整推理
        full_baseline = self.test_baseline()
        
        avg_time = early_exit_ratio * early_time + (1 - early_exit_ratio) * full_baseline['time_ms']
        avg_flash = early_exit_ratio * early_flash + (1 - early_exit_ratio) * full_baseline['flash_bytes']
        
        return {
            'strategy': f'Cascade ({int(early_exit_ratio*100)}% early exit)',
            'time_ms': avg_time,
            'flash_bytes': avg_flash,
            'speedup': full_baseline['time_ms'] / avg_time,
            'flash_saved': int((1 - avg_flash / full_baseline['flash_bytes']) * 100)
        }
    
    def test_all_combined(self) -> dict:
        """测试组合优化"""
        # 参数
        compression_ratio = 2.0
        early_exit_ratio = 0.7
        exit_layer = 2
        
        # 早退出场景（70%）
        early_time = 0
        early_flash = 0
        
        # Layer 0: 初始加载（压缩）
        compressed_size = int(self.layer_sizes[0] / compression_ratio)
        early_time += self.simulate_flash_read(compressed_size)
        early_time += self.simulate_decompress(compressed_size, 'LZ4')
        early_time += self.simulate_cim_compute(0)
        early_flash += compressed_size
        
        # Layer 1-2: 流水线 + 压缩
        for i in range(1, exit_layer + 1):
            compressed_size = int(self.layer_sizes[i] / compression_ratio)
            flash_time = self.simulate_flash_read(compressed_size)
            decompress_time = self.simulate_decompress(compressed_size, 'LZ4')
            compute_time = self.simulate_cim_compute(i)
            # 流水线：Flash+解压 与 CIM 并行
            early_time += max(flash_time + decompress_time, compute_time)
            early_flash += compressed_size
        
        # 完整推理场景（30%）
        full_time = 0
        full_flash = 0
        
        # Layer 0
        compressed_size = int(self.layer_sizes[0] / compression_ratio)
        full_time += self.simulate_flash_read(compressed_size)
        full_time += self.simulate_decompress(compressed_size, 'LZ4')
        full_time += self.simulate_cim_compute(0)
        full_flash += compressed_size
        
        # Layer 1-N: 流水线 + 压缩
        for i in range(1, self.num_layers):
            compressed_size = int(self.layer_sizes[i] / compression_ratio)
            flash_time = self.simulate_flash_read(compressed_size)
            decompress_time = self.simulate_decompress(compressed_size, 'LZ4')
            compute_time = self.simulate_cim_compute(i)
            full_time += max(flash_time + decompress_time, compute_time)
            full_flash += compressed_size
        
        avg_time = early_exit_ratio * early_time + (1 - early_exit_ratio) * full_time
        avg_flash = early_exit_ratio * early_flash + (1 - early_exit_ratio) * full_flash
        
        baseline = self.test_baseline()
        return {
            'strategy': 'All Combined',
            'time_ms': avg_time,
            'flash_bytes': avg_flash,
            'speedup': baseline['time_ms'] / avg_time,
            'flash_saved': int((1 - avg_flash / baseline['flash_bytes']) * 100)
        }
    
    def run_all_tests(self) -> List[dict]:
        """运行所有测试"""
        results = [
            self.test_baseline(),
            self.test_pipeline(),
            self.test_compression(2.0, 'LZ4'),
            self.test_cascade(0.7, 2),
            self.test_all_combined()
        ]
        return results
    
    def print_results(self, results: List[dict]):
        """打印测试结果"""
        print("\n" + "="*70)
        print(" Flash IO Optimization Performance Comparison")
        print("="*70)
        print(f"{'Strategy':<30} {'Time (ms)':<12} {'Speedup':<10} {'Flash Saved'}")
        print("-"*70)
        
        for r in results:
            print(f"{r['strategy']:<30} {r['time_ms']:>8.1f}    {r['speedup']:>6.2f}x    {r['flash_saved']:>6}%")
        
        print("="*70)
        print(f"\nTotal Flash Size: {sum(self.layer_sizes) / 1024:.0f} KB")
        print(f"Model Layers: {self.num_layers}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Flash IO Optimization Benchmark')
    parser.add_argument('--layers', type=int, default=8, help='Number of model layers')
    parser.add_argument('--compress', action='store_true', help='Enable compression test')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # 运行测试
    benchmark = FlashIOBenchmark(num_layers=args.layers)
    results = benchmark.run_all_tests()
    benchmark.print_results(results)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
