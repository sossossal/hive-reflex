#!/usr/bin/env python3
"""
IMC-22 SDK Python 绑定
使用 pybind11 封装 C SDK，提供 Pythonic 接口

@file imc22/__init__.py
@version 2.1.0
"""

import numpy as np
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
from enum import IntEnum
import ctypes
import os

__version__ = "2.1.0"
__all__ = ['CIM', 'Power', 'DVFS', 'NeuralReflex', 'Simulator']


# ============================================================================
# 枚举定义
# ============================================================================

class PowerMode(IntEnum):
    """电源模式"""
    ACTIVE = 2
    STANDBY = 1
    DEEPSLEEP = 0


class DVFSFreq(IntEnum):
    """DVFS 频率等级"""
    FREQ_1MHZ = 0
    FREQ_10MHZ = 1
    FREQ_50MHZ = 2
    FREQ_100MHZ = 3


class ActivationType(IntEnum):
    """激活函数类型"""
    NONE = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3
    SOFTMAX = 4


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class CIMConfig:
    """CIM 配置"""
    mac_count: int = 256
    data_width: int = 8
    sparse_enable: bool = True
    sparse_threshold: int = 2


@dataclass
class InferenceResult:
    """推理结果"""
    output: np.ndarray
    latency_us: float
    sparsity_ratio: float
    energy_estimate_uj: float


@dataclass
class PowerState:
    """电源状态"""
    mode: PowerMode
    voltage_mv: int
    current_ma: float
    power_mw: float


# ============================================================================
# CIM 核心接口
# ============================================================================

class CIM:
    """
    CIM (Computing-in-Memory) 核心控制接口
    
    示例：
    >>> cim = CIM()
    >>> cim.load_model("reflex_net.bin")
    >>> result = cim.infer(sensor_data)
    >>> print(f"推理延迟: {result.latency_us}μs")
    """
    
    def __init__(self, config: Optional[CIMConfig] = None, 
                 use_simulator: bool = True):
        """
        初始化 CIM 接口
        
        Args:
            config: CIM 配置，None 使用默认
            use_simulator: 是否使用软件模拟器（无硬件时）
        """
        self.config = config or CIMConfig()
        self.use_simulator = use_simulator
        self._model_loaded = False
        self._weights = None
        self._model_info = {}
        
        # 尝试加载 C 库
        self._lib = None
        if not use_simulator:
            try:
                self._lib = self._load_native_lib()
            except Exception as e:
                print(f"⚠️ 无法加载原生库: {e}, 回退到模拟器")
                self.use_simulator = True
    
    def _load_native_lib(self):
        """加载原生 C 库"""
        lib_paths = [
            "libimc22.so",
            "libimc22.dll",
            "/usr/local/lib/libimc22.so"
        ]
        
        for path in lib_paths:
            if os.path.exists(path):
                return ctypes.CDLL(path)
        
        raise FileNotFoundError("找不到 IMC-22 原生库")
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径 (.bin 或 .onnx)
            
        Returns:
            是否成功
        """
        if model_path.endswith('.bin'):
            return self._load_binary_model(model_path)
        elif model_path.endswith('.onnx'):
            return self._load_onnx_model(model_path)
        else:
            raise ValueError(f"不支持的模型格式: {model_path}")
    
    def _load_binary_model(self, path: str) -> bool:
        """加载二进制模型"""
        with open(path, 'rb') as f:
            data = f.read()
        
        # 解析模型头
        magic = data[:4]
        if magic != b'TML1':
            raise ValueError("无效的模型魔数")
        
        self._model_info = {
            'format': 'binary',
            'size': len(data)
        }
        self._model_loaded = True
        return True
    
    def _load_onnx_model(self, path: str) -> bool:
        """加载 ONNX 模型"""
        try:
            import onnx
            model = onnx.load(path)
            
            # 提取权重
            self._weights = {}
            for init in model.graph.initializer:
                from onnx import numpy_helper
                self._weights[init.name] = numpy_helper.to_array(init)
            
            self._model_info = {
                'format': 'onnx',
                'layers': len(model.graph.node)
            }
            self._model_loaded = True
            return True
            
        except ImportError:
            raise ImportError("需要安装 onnx: pip install onnx")
    
    def infer(self, input_data: np.ndarray) -> InferenceResult:
        """
        执行推理
        
        Args:
            input_data: 输入数据 (numpy array)
            
        Returns:
            推理结果
        """
        if not self._model_loaded:
            raise RuntimeError("请先加载模型")
        
        import time
        start = time.perf_counter()
        
        if self.use_simulator:
            output = self._simulate_inference(input_data)
        else:
            output = self._hardware_inference(input_data)
        
        latency = (time.perf_counter() - start) * 1e6
        
        # 估算稀疏率
        sparsity = self._estimate_sparsity(input_data)
        
        return InferenceResult(
            output=output,
            latency_us=latency,
            sparsity_ratio=sparsity,
            energy_estimate_uj=latency * 0.05  # 简化功耗模型
        )
    
    def _simulate_inference(self, input_data: np.ndarray) -> np.ndarray:
        """软件模拟推理"""
        if self._weights is None:
            # 简单的线性变换
            return input_data * 0.5
        
        # 简化的前向传播
        x = input_data.flatten().astype(np.float32)
        
        for name, w in self._weights.items():
            if len(w.shape) == 2:  # 全连接层
                if x.shape[0] == w.shape[1]:
                    x = np.dot(x, w.T)
                    x = np.maximum(x, 0)  # ReLU
        
        return x
    
    def _hardware_inference(self, input_data: np.ndarray) -> np.ndarray:
        """硬件推理"""
        # 通过 C 库调用硬件
        raise NotImplementedError("硬件推理需要原生库支持")
    
    def _estimate_sparsity(self, data: np.ndarray) -> float:
        """估算稀疏率"""
        total = data.size
        zeros = np.sum(np.abs(data) < self.config.sparse_threshold / 128.0)
        return zeros / total if total > 0 else 0.0
    
    @property
    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self._model_loaded


# ============================================================================
# 电源管理接口
# ============================================================================

class Power:
    """
    电源管理接口
    
    示例：
    >>> pwr = Power()
    >>> pwr.set_mode(PowerMode.STANDBY)
    >>> print(f"当前功率: {pwr.state.power_mw}mW")
    """
    
    def __init__(self):
        self._mode = PowerMode.ACTIVE
        self._wakeup_sources = ['can', 'gpio']
    
    def set_mode(self, mode: PowerMode) -> bool:
        """设置电源模式"""
        self._mode = mode
        return True
    
    def set_wakeup_sources(self, sources: List[str]) -> None:
        """设置唤醒源"""
        self._wakeup_sources = sources
    
    @property
    def state(self) -> PowerState:
        """获取电源状态"""
        power_table = {
            PowerMode.ACTIVE: (1000, 50.0, 50.0),
            PowerMode.STANDBY: (600, 1.0, 0.6),
            PowerMode.DEEPSLEEP: (400, 0.001, 0.0004)
        }
        
        mv, ma, mw = power_table.get(self._mode, (1000, 50.0, 50.0))
        
        return PowerState(
            mode=self._mode,
            voltage_mv=mv,
            current_ma=ma,
            power_mw=mw
        )


# ============================================================================
# DVFS 接口
# ============================================================================

class DVFS:
    """
    动态电压频率缩放接口
    
    示例：
    >>> dvfs = DVFS()
    >>> dvfs.enable_auto_scale(util_low=20, util_high=80)
    >>> dvfs.report_utilization(75)
    """
    
    def __init__(self):
        self._enabled = False
        self._freq = DVFSFreq.FREQ_100MHZ
        self._auto_scale = False
        self._util_thresholds = (50, 200)
    
    def enable(self) -> bool:
        """启用 DVFS"""
        self._enabled = True
        return True
    
    def disable(self) -> bool:
        """禁用 DVFS"""
        self._enabled = False
        return True
    
    def set_frequency(self, freq: DVFSFreq) -> bool:
        """设置频率"""
        self._freq = freq
        return True
    
    def enable_auto_scale(self, util_low: int = 50, util_high: int = 200) -> bool:
        """启用自动缩放"""
        self._auto_scale = True
        self._util_thresholds = (util_low, util_high)
        return True
    
    def report_utilization(self, util: int) -> None:
        """报告利用率"""
        if self._auto_scale and self._enabled:
            if util >= self._util_thresholds[1]:
                self._freq = DVFSFreq.FREQ_100MHZ
            elif util <= self._util_thresholds[0]:
                self._freq = DVFSFreq.FREQ_10MHZ
    
    @property
    def current_freq(self) -> DVFSFreq:
        """获取当前频率"""
        return self._freq


# ============================================================================
# 神经反射控制接口
# ============================================================================

class NeuralReflex:
    """
    神经反射控制接口（集成 TinyML 自适应）
    
    示例：
    >>> reflex = NeuralReflex()
    >>> weights = reflex.compute_blend(torque=5.0, velocity=1.2)
    >>> print(f"PID 权重: {weights['pid']:.2f}")
    """
    
    def __init__(self):
        self._pid_weight = 0.5
        self._neural_weight = 0.5
        self._compliance = 0.5
        self._adaptive_enabled = True
    
    def compute_blend(self, torque: float, velocity: float,
                     position_error: float = 0.0,
                     external_force: float = 0.0) -> Dict[str, float]:
        """
        计算混合权重
        
        Args:
            torque: 当前力矩
            velocity: 当前速度
            position_error: 位置误差
            external_force: 外部力
            
        Returns:
            权重字典 {'pid': float, 'neural': float, 'compliance': float}
        """
        if not self._adaptive_enabled:
            return {
                'pid': self._pid_weight,
                'neural': self._neural_weight,
                'compliance': self._compliance
            }
        
        # 简化的自适应逻辑
        load_level = abs(torque) / 10.0
        
        if load_level > 0.8:
            # 高负载：偏向 PID
            pid_w = 0.8 + 0.2 * (1 - load_level)
        else:
            # 正常：平衡
            error_mag = abs(position_error) + abs(velocity * 0.1)
            pid_w = 0.4 + 0.3 * min(error_mag, 1.0)
        
        neural_w = 1.0 - pid_w
        compliance = 0.5 * (1 - load_level)
        
        self._pid_weight = pid_w
        self._neural_weight = neural_w
        self._compliance = compliance
        
        return {
            'pid': pid_w,
            'neural': neural_w,
            'compliance': compliance
        }
    
    def set_adaptive(self, enabled: bool) -> None:
        """启用/禁用自适应模式"""
        self._adaptive_enabled = enabled
    
    def force_weights(self, pid: float, neural: float) -> None:
        """强制设置权重"""
        self._pid_weight = max(0, min(1, pid))
        self._neural_weight = max(0, min(1, neural))
        self._adaptive_enabled = False


# ============================================================================
# NumPy CIM 模拟器
# ============================================================================

class Simulator:
    """
    NumPy 纯 Python CIM 模拟器
    
    用于无硬件环境下的开发和测试
    
    示例：
    >>> sim = Simulator(mac_count=256)
    >>> result = sim.matmul(activations, weights, sparse=True)
    >>> print(f"稀疏加速: {result['speedup']:.2f}x")
    """
    
    def __init__(self, mac_count: int = 256, data_width: int = 8):
        self.mac_count = mac_count
        self.data_width = data_width
        self._sparse_threshold = 2
    
    def matmul(self, input_data: np.ndarray, weights: np.ndarray,
              sparse: bool = True, threshold: int = 2) -> Dict:
        """
        模拟 CIM 矩阵乘法
        
        Args:
            input_data: 输入激活值
            weights: 权重矩阵
            sparse: 是否启用稀疏模式
            threshold: 稀疏阈值
            
        Returns:
            结果字典
        """
        import time
        
        start = time.perf_counter()
        
        # 量化到 int8
        input_q = self._quantize(input_data)
        weight_q = self._quantize(weights)
        
        if sparse:
            result, stats = self._sparse_matmul(input_q, weight_q, threshold)
        else:
            result = np.dot(input_q.flatten(), weight_q.reshape(-1, weight_q.shape[-1]))
            stats = {'skipped': 0, 'total': input_q.size * weight_q.shape[-1]}
        
        elapsed = time.perf_counter() - start
        
        sparsity = stats['skipped'] / max(stats['total'], 1)
        
        return {
            'output': self._dequantize(result),
            'latency_s': elapsed,
            'sparsity': sparsity,
            'speedup': 1.0 / (1.0 - sparsity * 0.9) if sparsity > 0 else 1.0,
            'ops_skipped': stats['skipped'],
            'ops_total': stats['total']
        }
    
    def _quantize(self, data: np.ndarray) -> np.ndarray:
        """量化到 int8"""
        scale = np.abs(data).max() / 127
        if scale == 0:
            return np.zeros_like(data, dtype=np.int8)
        return np.clip(np.round(data / scale), -128, 127).astype(np.int8)
    
    def _dequantize(self, data: np.ndarray) -> np.ndarray:
        """反量化到 float32"""
        return data.astype(np.float32) / 127.0
    
    def _sparse_matmul(self, input_data: np.ndarray, weights: np.ndarray,
                      threshold: int) -> tuple:
        """稀疏矩阵乘法"""
        input_flat = input_data.flatten()
        
        # 创建掩码
        input_mask = np.abs(input_flat) >= threshold
        weight_mask = np.abs(weights) >= threshold
        
        combined_mask = input_mask[:, None] & weight_mask.reshape(-1, weights.shape[-1])
        
        # 只计算非零部分
        result = np.zeros(weights.shape[-1], dtype=np.int32)
        
        for i in range(len(input_flat)):
            if input_mask[i]:
                for j in range(weights.shape[-1]):
                    if weight_mask.flat[i * weights.shape[-1] + j]:
                        result[j] += int(input_flat[i]) * int(weights.flat[i * weights.shape[-1] + j])
        
        total_ops = len(input_flat) * weights.shape[-1]
        skipped_ops = total_ops - combined_mask.sum()
        
        return result, {'skipped': skipped_ops, 'total': total_ops}


# ============================================================================
# 便捷函数
# ============================================================================

def quick_infer(model_path: str, input_data: np.ndarray) -> np.ndarray:
    """快速推理"""
    cim = CIM(use_simulator=True)
    cim.load_model(model_path)
    result = cim.infer(input_data)
    return result.output


def estimate_power(mode: str = 'active') -> float:
    """估算功耗"""
    power_map = {
        'active': 50.0,
        'standby': 0.5,
        'deepsleep': 0.0001
    }
    return power_map.get(mode.lower(), 50.0)
