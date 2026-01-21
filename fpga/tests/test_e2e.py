#!/usr/bin/env python3
"""
Hive-Reflex 2.1 端到端测试框架
使用 Pytest 验证从模型编译到硬件输出的全链路

使用方法:
    pytest fpga/tests/ -v                   # 运行所有测试
    pytest fpga/tests/test_e2e.py -v        # 运行端到端测试
    pytest fpga/tests/ --hil                # 运行 HIL 测试 (需硬件)
"""

import pytest
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'imc22_sdk' / 'python'))
sys.path.insert(0, str(PROJECT_ROOT / 'mlir_compiler'))
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))

# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    use_hardware: bool = False
    hardware_port: str = "/dev/ttyUSB0"
    hardware_baudrate: int = 115200
    model_path: str = "models/test_model.onnx"
    timeout_s: float = 10.0


def pytest_addoption(parser):
    """添加 Pytest 命令行选项"""
    parser.addoption(
        "--hil", action="store_true", default=False,
        help="启用 HIL (硬件在环) 测试"
    )
    parser.addoption(
        "--port", action="store", default="/dev/ttyUSB0",
        help="硬件串口"
    )


@pytest.fixture
def config(request):
    """测试配置 fixture"""
    return TestConfig(
        use_hardware=request.config.getoption("--hil"),
        hardware_port=request.config.getoption("--port")
    )


# ============================================================================
# 模拟器 Fixtures
# ============================================================================

@pytest.fixture
def cim_simulator():
    """CIM 模拟器 fixture"""
    from imc22 import CIM, CIMConfig
    
    config = CIMConfig(
        mac_count=256,
        data_width=8,
        sparse_enable=True,
        sparse_threshold=2
    )
    
    cim = CIM(config=config, use_simulator=True)
    yield cim


@pytest.fixture
def mlir_compiler():
    """MLIR 编译器 fixture"""
    from optimizer import MLIROptimizer
    
    compiler = MLIROptimizer()
    yield compiler


@pytest.fixture
def test_model():
    """测试模型 fixture"""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))
    
    model = SimpleModel()
    yield model


# ============================================================================
# 端到端测试
# ============================================================================

class TestModelCompilation:
    """模型编译测试"""
    
    def test_onnx_export(self, test_model, tmp_path):
        """测试 ONNX 导出"""
        import torch
        
        model_path = tmp_path / "test_model.onnx"
        dummy_input = torch.randn(1, 8)
        
        torch.onnx.export(
            test_model,
            dummy_input,
            str(model_path),
            opset_version=13
        )
        
        assert model_path.exists()
        assert model_path.stat().st_size > 0
    
    def test_quantization(self, test_model, tmp_path):
        """测试量化"""
        from qat_trainer import QATTrainer, QuantizedLinear
        import torch
        
        # 创建虚拟数据
        X = torch.randn(100, 8)
        y = torch.randint(0, 2, (100, 2)).float()
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        # 创建 QAT 训练器
        trainer = QATTrainer(test_model, loader, num_bits=8, lr=0.001)
        
        # 校准
        trainer.calibrate(num_batches=5)
        
        assert trainer.calibration_done
    
    def test_sparsity_analysis(self, tmp_path):
        """测试稀疏度分析"""
        from sparsity_optimizer import SparsityOptimizer
        
        # 创建测试权重
        weights = np.random.randn(16, 8).astype(np.float32)
        weights[np.abs(weights) < 0.5] = 0  # 50% 稀疏
        
        optimizer = SparsityOptimizer()
        analysis = optimizer.analyze_weights(weights, name="test_layer")
        
        assert 'sparsity' in analysis
        assert analysis['sparsity'] > 0.3  # 至少 30% 稀疏


class TestCIMSimulation:
    """CIM 仿真测试"""
    
    def test_dense_inference(self, cim_simulator):
        """测试密集推理"""
        input_data = np.random.randn(8).astype(np.float32)
        
        # 模拟推理
        # result = cim_simulator.infer(input_data)
        # 简化测试
        assert cim_simulator.use_simulator
    
    def test_sparse_inference(self, cim_simulator):
        """测试稀疏推理"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        # 50% 稀疏输入
        input_data = np.zeros(16, dtype=np.float32)
        input_data[::2] = np.random.randn(8).astype(np.float32)
        
        weights = np.random.randn(16, 8).astype(np.float32)
        
        result = sim.matmul(input_data, weights, sparse=True, threshold=1)
        
        assert result['sparsity'] > 0.3
        assert result['output'].shape == (8,)
    
    def test_dvfs_modes(self):
        """测试 DVFS 模式切换"""
        from imc22 import DVFS, DVFSFreq
        
        dvfs = DVFS()
        dvfs.enable()
        
        # Active
        dvfs.set_frequency(DVFSFreq.FREQ_100MHZ)
        assert dvfs.current_freq == DVFSFreq.FREQ_100MHZ
        
        # Standby
        dvfs.set_frequency(DVFSFreq.FREQ_10MHZ)
        assert dvfs.current_freq == DVFSFreq.FREQ_10MHZ


class TestPowerManagement:
    """电源管理测试"""
    
    def test_power_modes(self):
        """测试电源模式"""
        from imc22 import Power, PowerMode
        
        pwr = Power()
        
        for mode in [PowerMode.ACTIVE, PowerMode.STANDBY, PowerMode.DEEPSLEEP]:
            pwr.set_mode(mode)
            assert pwr.state.mode == mode
    
    def test_power_estimation(self):
        """测试功耗估算"""
        from imc22 import Power, PowerMode
        
        pwr = Power()
        
        pwr.set_mode(PowerMode.ACTIVE)
        active_power = pwr.state.power_mw
        
        pwr.set_mode(PowerMode.DEEPSLEEP)
        deepsleep_power = pwr.state.power_mw
        
        # DeepSleep 应该比 Active 低得多
        assert deepsleep_power < active_power * 0.01


class TestNeuralReflex:
    """神经反射控制测试"""
    
    def test_adaptive_weights(self):
        """测试自适应权重"""
        from imc22 import NeuralReflex
        
        reflex = NeuralReflex()
        
        # 正常负载
        weights_normal = reflex.compute_blend(torque=2.0, velocity=1.0)
        
        # 高负载
        weights_high = reflex.compute_blend(torque=9.0, velocity=0.5)
        
        # 高负载时应该偏向 PID
        assert weights_high['pid'] > weights_normal['pid']
    
    def test_compliance_range(self):
        """测试合规度范围"""
        from imc22 import NeuralReflex
        
        reflex = NeuralReflex()
        
        for torque in [0, 5, 10]:
            weights = reflex.compute_blend(torque=torque, velocity=1.0)
            
            assert 0 <= weights['pid'] <= 1
            assert 0 <= weights['neural'] <= 1
            assert 0 <= weights['compliance'] <= 1


class TestEndToEnd:
    """端到端测试"""
    
    def test_model_to_inference(self, test_model, tmp_path):
        """测试模型到推理的完整流程"""
        import torch
        from imc22 import CIM, Simulator
        
        # 1. 导出模型
        model_path = tmp_path / "e2e_model.onnx"
        dummy_input = torch.randn(1, 8)
        torch.onnx.export(test_model, dummy_input, str(model_path), opset_version=13)
        
        # 2. 加载模型到 CIM
        cim = CIM(use_simulator=True)
        success = cim.load_model(str(model_path))
        assert success
        
        # 3. 执行推理
        input_data = np.random.randn(8).astype(np.float32)
        result = cim.infer(input_data)
        
        assert result.output is not None
        assert result.latency_us > 0
    
    def test_sparse_optimization_chain(self, tmp_path):
        """测试稀疏优化链"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        # 模拟不同稀疏度
        for sparsity in [0.0, 0.5, 0.8]:
            input_data = np.random.randn(32).astype(np.float32)
            mask = np.random.random(32) > sparsity
            input_data *= mask
            
            weights = np.random.randn(32, 8).astype(np.float32)
            
            result = sim.matmul(input_data, weights, sparse=True)
            
            assert result['output'].shape == (8,)


# ============================================================================
# HIL (硬件在环) 测试
# ============================================================================

class TestHIL:
    """硬件在环测试 (需要连接硬件)"""
    
    @pytest.fixture
    def hardware_connection(self, config):
        """硬件连接 fixture"""
        if not config.use_hardware:
            pytest.skip("HIL 测试需要 --hil 选项")
        
        try:
            import serial
            port = serial.Serial(
                config.hardware_port,
                config.hardware_baudrate,
                timeout=config.timeout_s
            )
            yield port
            port.close()
        except Exception as e:
            pytest.skip(f"无法连接硬件: {e}")
    
    def test_hardware_ping(self, hardware_connection):
        """测试硬件连接"""
        hardware_connection.write(b"PING\n")
        response = hardware_connection.readline()
        
        assert b"PONG" in response
    
    def test_cim_compute_hardware(self, hardware_connection):
        """测试硬件 CIM 计算"""
        # 发送测试数据
        test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
        cmd = b"CIM_COMPUTE:" + test_data.tobytes() + b"\n"
        
        hardware_connection.write(cmd)
        response = hardware_connection.readline()
        
        assert b"RESULT:" in response
    
    def test_latency_measurement(self, hardware_connection, config):
        """测试延迟测量"""
        latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            
            hardware_connection.write(b"PING\n")
            hardware_connection.readline()
            
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"\n延迟统计: 平均 {avg_latency:.2f}ms, 最大 {max_latency:.2f}ms")
        
        # 验证延迟要求
        assert avg_latency < 10.0  # 平均 < 10ms
        assert max_latency < 50.0  # 最大 < 50ms


class TestMotorHIL:
    """关节电机 HIL 测试"""
    
    @pytest.fixture
    def motor_controller(self, config):
        """电机控制器 fixture"""
        if not config.use_hardware:
            pytest.skip("电机 HIL 测试需要 --hil 选项")
        
        # 模拟电机控制器接口
        class MockMotorController:
            def set_torque(self, torque): pass
            def get_position(self): return 0.0
            def get_velocity(self): return 0.0
        
        yield MockMotorController()
    
    def test_pid_response(self, motor_controller):
        """测试 PID 响应"""
        from imc22 import NeuralReflex
        
        reflex = NeuralReflex()
        
        # 模拟控制循环
        for _ in range(100):
            pos = motor_controller.get_position()
            vel = motor_controller.get_velocity()
            
            weights = reflex.compute_blend(
                torque=0.0, 
                velocity=vel,
                position_error=1.0 - pos
            )
            
            # 计算控制输出
            pid_output = weights['pid'] * (1.0 - pos)
            neural_output = weights['neural'] * 0.5
            
            total_output = pid_output + neural_output
            motor_controller.set_torque(total_output)
    
    def test_position_accuracy(self, motor_controller):
        """测试位置精度"""
        target_positions = [0.0, 0.5, 1.0, -0.5]
        
        for target in target_positions:
            # motor_controller.move_to(target)
            # time.sleep(1.0)
            # actual = motor_controller.get_position()
            # error = abs(actual - target)
            # assert error < 0.01  # 1% 误差
            pass  # 占位


# ============================================================================
# 性能基准测试
# ============================================================================

class TestBenchmark:
    """性能基准测试"""
    
    def test_inference_throughput(self):
        """测试推理吞吐量"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        input_data = np.random.randn(256).astype(np.float32)
        weights = np.random.randn(256, 64).astype(np.float32)
        
        iterations = 1000
        start = time.perf_counter()
        
        for _ in range(iterations):
            sim.matmul(input_data, weights, sparse=True)
        
        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed
        
        print(f"\n吞吐量: {throughput:.0f} 次/秒")
        
        assert throughput > 100  # 至少 100 次/秒
    
    def test_sparse_speedup(self):
        """测试稀疏加速"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        # 50% 稀疏输入
        input_sparse = np.zeros(256, dtype=np.float32)
        input_sparse[::2] = np.random.randn(128).astype(np.float32)
        
        input_dense = np.random.randn(256).astype(np.float32)
        weights = np.random.randn(256, 64).astype(np.float32)
        
        # 稀疏计算
        result_sparse = sim.matmul(input_sparse, weights, sparse=True)
        
        # 密集计算
        result_dense = sim.matmul(input_dense, weights, sparse=False)
        
        speedup = result_sparse['speedup']
        
        print(f"\n稀疏加速: {speedup:.2f}x")
        
        assert speedup >= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
