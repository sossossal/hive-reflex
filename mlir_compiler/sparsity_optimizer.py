#!/usr/bin/env python3
"""
稀疏优化器 - MLIR 编译器稀疏支持模块
分析和优化稀疏神经网络模型，生成稀疏 CIM 指令

@file sparsity_optimizer.py
@version 2.1.0
"""

import numpy as np
import onnx
from onnx import numpy_helper
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SparsityStats:
    """稀疏统计信息"""
    layer_name: str
    total_elements: int
    zero_elements: int
    near_zero_elements: int  # 接近零 (< threshold)
    sparsity_ratio: float
    recommended_mode: str  # 'dense', 'sparse', 'csr'


@dataclass
class SparseLayerConfig:
    """稀疏层配置"""
    layer_name: str
    use_sparse: bool
    threshold: float
    format: str  # 'dense', 'csr', 'csc', 'coo'
    pruned_weights: Optional[np.ndarray] = None


class SparsityOptimizer:
    """
    稀疏优化器
    
    功能：
    - 模型稀疏度分析
    - 权重剪枝
    - 稀疏格式转换
    - 稀疏 CIM 指令生成
    """
    
    def __init__(self, threshold: float = 0.01, prune_ratio: float = 0.0):
        """
        初始化稀疏优化器
        
        Args:
            threshold: 近零阈值（|value| < threshold 视为零）
            prune_ratio: 目标剪枝比例 (0.0 = 不剪枝, 0.5 = 剪枝 50%)
        """
        self.threshold = threshold
        self.prune_ratio = prune_ratio
        self.stats: Dict[str, SparsityStats] = {}
        self.layer_configs: Dict[str, SparseLayerConfig] = {}
    
    def analyze_sparsity(self, model_path: str) -> Dict[str, SparsityStats]:
        """
        分析 ONNX 模型的稀疏度
        
        Args:
            model_path: ONNX 模型路径
            
        Returns:
            各层的稀疏统计信息
        """
        logger.info(f"📊 分析模型稀疏度: {model_path}")
        
        model = onnx.load(model_path)
        self.stats = {}
        
        for initializer in model.graph.initializer:
            weights = numpy_helper.to_array(initializer)
            
            total = weights.size
            zeros = np.sum(weights == 0)
            near_zeros = np.sum(np.abs(weights) < self.threshold)
            sparsity = near_zeros / total if total > 0 else 0.0
            
            # 推荐模式
            if sparsity > 0.7:
                mode = 'csr'  # 高稀疏度使用 CSR
            elif sparsity > 0.3:
                mode = 'sparse'  # 中等稀疏度使用稀疏计算
            else:
                mode = 'dense'  # 低稀疏度使用密集计算
            
            stats = SparsityStats(
                layer_name=initializer.name,
                total_elements=total,
                zero_elements=zeros,
                near_zero_elements=near_zeros,
                sparsity_ratio=sparsity,
                recommended_mode=mode
            )
            
            self.stats[initializer.name] = stats
            
            logger.info(f"  {initializer.name}: "
                       f"稀疏率 {sparsity*100:.1f}%, "
                       f"推荐模式: {mode}")
        
        return self.stats
    
    def prune_weights(self, model_path: str, output_path: str,
                     strategy: str = 'magnitude') -> Dict[str, SparseLayerConfig]:
        """
        权重剪枝
        
        Args:
            model_path: 输入模型路径
            output_path: 输出模型路径
            strategy: 剪枝策略 ('magnitude', 'random', 'structured')
            
        Returns:
            各层的稀疏配置
        """
        logger.info(f"✂️  权重剪枝: 策略={strategy}, 目标比例={self.prune_ratio*100:.0f}%")
        
        model = onnx.load(model_path)
        self.layer_configs = {}
        
        for i, initializer in enumerate(model.graph.initializer):
            weights = numpy_helper.to_array(initializer)
            original_shape = weights.shape
            
            if self.prune_ratio > 0:
                # 幅度剪枝
                if strategy == 'magnitude':
                    flat = weights.flatten()
                    threshold_value = np.percentile(np.abs(flat), 
                                                    self.prune_ratio * 100)
                    mask = np.abs(weights) >= threshold_value
                    pruned = weights * mask
                    
                # 随机剪枝
                elif strategy == 'random':
                    mask = np.random.random(weights.shape) > self.prune_ratio
                    pruned = weights * mask
                    
                # 结构化剪枝（通道级）
                elif strategy == 'structured':
                    if len(weights.shape) >= 2:
                        channel_norms = np.linalg.norm(weights, axis=tuple(range(1, len(weights.shape))))
                        threshold_value = np.percentile(channel_norms, 
                                                       self.prune_ratio * 100)
                        mask = channel_norms >= threshold_value
                        pruned = weights.copy()
                        pruned[~mask] = 0
                    else:
                        pruned = weights
                else:
                    pruned = weights
                    
                # 计算实际稀疏率
                actual_sparsity = np.sum(pruned == 0) / pruned.size
                
                logger.info(f"  {initializer.name}: "
                           f"剪枝后稀疏率 {actual_sparsity*100:.1f}%")
            else:
                pruned = weights
                actual_sparsity = np.sum(weights == 0) / weights.size
            
            # 决定格式
            if actual_sparsity > 0.7:
                format_type = 'csr'
            elif actual_sparsity > 0.3:
                format_type = 'sparse'
            else:
                format_type = 'dense'
            
            config = SparseLayerConfig(
                layer_name=initializer.name,
                use_sparse=(actual_sparsity > 0.3),
                threshold=self.threshold,
                format=format_type,
                pruned_weights=pruned
            )
            
            self.layer_configs[initializer.name] = config
            
            # 更新模型权重
            model.graph.initializer[i].CopyFrom(
                numpy_helper.from_array(pruned.astype(weights.dtype), 
                                       initializer.name)
            )
        
        onnx.save(model, output_path)
        logger.info(f"✓ 剪枝模型已保存: {output_path}")
        
        return self.layer_configs
    
    def convert_to_csr(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将权重转换为 CSR 格式
        
        Args:
            weights: 2D 权重矩阵
            
        Returns:
            (values, col_indices, row_ptr)
        """
        if len(weights.shape) != 2:
            # 展平高维张量
            weights = weights.reshape(weights.shape[0], -1)
        
        values = []
        col_indices = []
        row_ptr = [0]
        
        for row in weights:
            for col_idx, val in enumerate(row):
                if val != 0:
                    values.append(val)
                    col_indices.append(col_idx)
            row_ptr.append(len(values))
        
        return (np.array(values), 
                np.array(col_indices, dtype=np.int16),
                np.array(row_ptr, dtype=np.int32))
    
    def generate_sparse_instructions(self, layer_config: SparseLayerConfig) -> List[Dict]:
        """
        生成稀疏 CIM 指令
        
        Args:
            layer_config: 层配置
            
        Returns:
            CIM 指令列表
        """
        instructions = []
        
        if not layer_config.use_sparse:
            # 密集模式
            instructions.append({
                'opcode': 'CIM_DENSE_MATMUL',
                'sparse_enable': False,
                'threshold': 0
            })
            return instructions
        
        if layer_config.format == 'csr':
            # CSR 稀疏模式
            values, col_indices, row_ptr = self.convert_to_csr(
                layer_config.pruned_weights
            )
            
            instructions.append({
                'opcode': 'CIM_SPARSE_CSR_SETUP',
                'num_values': len(values),
                'row_count': len(row_ptr) - 1
            })
            
            instructions.append({
                'opcode': 'CIM_SPARSE_CSR_LOAD_VALUES',
                'data': values.tolist()
            })
            
            instructions.append({
                'opcode': 'CIM_SPARSE_CSR_LOAD_INDICES',
                'col_indices': col_indices.tolist(),
                'row_ptr': row_ptr.tolist()
            })
            
            instructions.append({
                'opcode': 'CIM_SPARSE_CSR_MATMUL',
                'sparse_enable': True
            })
            
        else:
            # 动态稀疏模式（跳过零值）
            instructions.append({
                'opcode': 'CIM_SPARSE_MATMUL',
                'sparse_enable': True,
                'threshold': int(layer_config.threshold * 128)  # 转为 int8 阈值
            })
        
        return instructions
    
    def optimize_model(self, input_path: str, output_path: str,
                      enable_pruning: bool = True,
                      prune_strategy: str = 'magnitude') -> Dict:
        """
        完整的稀疏优化流程
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
            enable_pruning: 是否启用剪枝
            prune_strategy: 剪枝策略
            
        Returns:
            优化报告
        """
        logger.info("=" * 50)
        logger.info("🚀 开始稀疏优化")
        logger.info("=" * 50)
        
        # 1. 分析原始稀疏度
        original_stats = self.analyze_sparsity(input_path)
        
        # 2. 剪枝（可选）
        if enable_pruning and self.prune_ratio > 0:
            layer_configs = self.prune_weights(input_path, output_path, 
                                              prune_strategy)
        else:
            # 不剪枝，直接复制
            import shutil
            shutil.copy(input_path, output_path)
            layer_configs = {}
        
        # 3. 重新分析
        final_stats = self.analyze_sparsity(output_path)
        
        # 4. 生成优化报告
        report = {
            'input_model': input_path,
            'output_model': output_path,
            'pruning_enabled': enable_pruning,
            'prune_ratio': self.prune_ratio,
            'prune_strategy': prune_strategy,
            'original_sparsity': {
                name: stats.sparsity_ratio 
                for name, stats in original_stats.items()
            },
            'final_sparsity': {
                name: stats.sparsity_ratio 
                for name, stats in final_stats.items()
            },
            'sparse_layers': sum(1 for s in final_stats.values() 
                                if s.recommended_mode != 'dense'),
            'estimated_speedup': self._estimate_speedup(final_stats)
        }
        
        logger.info("\n" + "=" * 50)
        logger.info("📈 优化报告")
        logger.info("=" * 50)
        logger.info(f"  稀疏层数: {report['sparse_layers']}")
        logger.info(f"  预估加速: {report['estimated_speedup']:.2f}x")
        logger.info(f"  预估功耗降低: {(1-1/report['estimated_speedup'])*100:.0f}%")
        
        return report
    
    def _estimate_speedup(self, stats: Dict[str, SparsityStats]) -> float:
        """估算加速比"""
        if not stats:
            return 1.0
        
        total_elements = sum(s.total_elements for s in stats.values())
        skipped_elements = sum(s.near_zero_elements for s in stats.values())
        
        if total_elements == 0:
            return 1.0
        
        # 简化模型：跳过的操作直接转化为加速
        # 实际需要考虑索引开销
        skip_ratio = skipped_elements / total_elements
        overhead = 0.1  # 10% 索引开销
        
        effective_skip = skip_ratio * (1 - overhead)
        speedup = 1 / (1 - effective_skip) if effective_skip < 1 else 10.0
        
        return min(speedup, 5.0)  # 最大 5x


def analyze_model_sparsity(model_path: str, threshold: float = 0.01) -> Dict:
    """
    便捷函数：分析模型稀疏度
    """
    optimizer = SparsityOptimizer(threshold=threshold)
    return optimizer.analyze_sparsity(model_path)


def prune_model(input_path: str, output_path: str, 
               prune_ratio: float = 0.3,
               strategy: str = 'magnitude') -> Dict:
    """
    便捷函数：剪枝模型
    """
    optimizer = SparsityOptimizer(prune_ratio=prune_ratio)
    return optimizer.prune_weights(input_path, output_path, strategy)


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='稀疏优化器')
    parser.add_argument('--model', required=True, help='输入 ONNX 模型')
    parser.add_argument('--output', help='输出模型路径')
    parser.add_argument('--analyze', action='store_true', help='仅分析稀疏度')
    parser.add_argument('--prune', action='store_true', help='启用剪枝')
    parser.add_argument('--prune-ratio', type=float, default=0.3, 
                       help='剪枝比例 (0.0-1.0)')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='近零阈值')
    parser.add_argument('--strategy', default='magnitude',
                       choices=['magnitude', 'random', 'structured'],
                       help='剪枝策略')
    
    args = parser.parse_args()
    
    optimizer = SparsityOptimizer(
        threshold=args.threshold,
        prune_ratio=args.prune_ratio
    )
    
    if args.analyze:
        # 仅分析
        stats = optimizer.analyze_sparsity(args.model)
        print("\n稀疏度分析结果:")
        for name, s in stats.items():
            print(f"  {name}: {s.sparsity_ratio*100:.1f}% ({s.recommended_mode})")
    else:
        # 完整优化
        if not args.output:
            args.output = args.model.replace('.onnx', '_sparse.onnx')
        
        report = optimizer.optimize_model(
            args.model, args.output,
            enable_pruning=args.prune,
            prune_strategy=args.strategy
        )
        
        print(f"\n✅ 优化完成! 输出: {args.output}")
        print(f"   预估加速: {report['estimated_speedup']:.2f}x")


if __name__ == '__main__':
    main()
