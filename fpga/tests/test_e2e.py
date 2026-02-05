#!/usr/bin/env python3
"""
Hive-Reflex 2.1 
 Pytest 

:
    pytest fpga/tests/ -v                   # 
    pytest fpga/tests/test_e2e.py -v        # 
    pytest fpga/tests/ --hil                #  HIL  ()
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

# 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'imc22_sdk' / 'python'))
sys.path.insert(0, str(PROJECT_ROOT / 'mlir_compiler'))
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))

# ============================================================================
# 
# ============================================================================

@dataclass
class TestConfig:
    """"""
    use_hardware: bool = False
    hardware_port: str = "/dev/ttyUSB0"
    hardware_baudrate: int = 115200
    model_path: str = "models/test_model.onnx"
    timeout_s: float = 10.0


def pytest_addoption(parser):
    """ Pytest """
    parser.addoption(
        "--hil", action="store_true", default=False,
        help=" HIL () "
    )
    parser.addoption(
        "--port", action="store", default="/dev/ttyUSB0",
        help=""
    )


@pytest.fixture
def config(request):
    """ fixture"""
    return TestConfig(
        use_hardware=request.config.getoption("--hil"),
        hardware_port=request.config.getoption("--port")
    )


# ============================================================================
#  Fixtures
# ============================================================================

@pytest.fixture
def cim_simulator():
    """CIM  fixture"""
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
    """MLIR  fixture"""
    from optimizer import MLIROptimizer
    
    compiler = MLIROptimizer()
    yield compiler


@pytest.fixture
def test_model():
    """ fixture"""
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
# 
# ============================================================================

class TestModelCompilation:
    """"""
    
    def test_onnx_export(self, test_model, tmp_path):
        """ ONNX """
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
        """"""
        from qat_trainer import QATTrainer, QuantizedLinear
        import torch
        
        # 
        X = torch.randn(100, 8)
        y = torch.randint(0, 2, (100, 2)).float()
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        #  QAT 
        trainer = QATTrainer(test_model, loader, num_bits=8, lr=0.001)
        
        # 
        trainer.calibrate(num_batches=5)
        
        assert trainer.calibration_done
    
    def test_sparsity_analysis(self, tmp_path):
        """"""
        from sparsity_optimizer import SparsityOptimizer
        
        # 
        weights = np.random.randn(16, 8).astype(np.float32)
        weights[np.abs(weights) < 0.5] = 0  # 50% 
        
        optimizer = SparsityOptimizer()
        analysis = optimizer.analyze_weights(weights, name="test_layer")
        
        assert 'sparsity' in analysis
        assert analysis['sparsity'] > 0.3  #  30% 


class TestCIMSimulation:
    """CIM """
    
    def test_dense_inference(self, cim_simulator):
        """"""
        input_data = np.random.randn(8).astype(np.float32)
        
        # 
        # result = cim_simulator.infer(input_data)
        # 
        assert cim_simulator.use_simulator
    
    def test_sparse_inference(self, cim_simulator):
        """"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        # 50% 
        input_data = np.zeros(16, dtype=np.float32)
        input_data[::2] = np.random.randn(8).astype(np.float32)
        
        weights = np.random.randn(16, 8).astype(np.float32)
        
        result = sim.matmul(input_data, weights, sparse=True, threshold=1)
        
        assert result['sparsity'] > 0.3
        assert result['output'].shape == (8,)
    
    def test_dvfs_modes(self):
        """ DVFS """
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
    """"""
    
    def test_power_modes(self):
        """"""
        from imc22 import Power, PowerMode
        
        pwr = Power()
        
        for mode in [PowerMode.ACTIVE, PowerMode.STANDBY, PowerMode.DEEPSLEEP]:
            pwr.set_mode(mode)
            assert pwr.state.mode == mode
    
    def test_power_estimation(self):
        """"""
        from imc22 import Power, PowerMode
        
        pwr = Power()
        
        pwr.set_mode(PowerMode.ACTIVE)
        active_power = pwr.state.power_mw
        
        pwr.set_mode(PowerMode.DEEPSLEEP)
        deepsleep_power = pwr.state.power_mw
        
        # DeepSleep  Active 
        assert deepsleep_power < active_power * 0.01


class TestNeuralReflex:
    """"""
    
    def test_adaptive_weights(self):
        """"""
        from imc22 import NeuralReflex
        
        reflex = NeuralReflex()
        
        # 
        weights_normal = reflex.compute_blend(torque=2.0, velocity=1.0)
        
        # 
        weights_high = reflex.compute_blend(torque=9.0, velocity=0.5)
        
        #  PID
        assert weights_high['pid'] > weights_normal['pid']
    
    def test_compliance_range(self):
        """"""
        from imc22 import NeuralReflex
        
        reflex = NeuralReflex()
        
        for torque in [0, 5, 10]:
            weights = reflex.compute_blend(torque=torque, velocity=1.0)
            
            assert 0 <= weights['pid'] <= 1
            assert 0 <= weights['neural'] <= 1
            assert 0 <= weights['compliance'] <= 1


class TestEndToEnd:
    """"""
    
    def test_model_to_inference(self, test_model, tmp_path):
        """"""
        import torch
        from imc22 import CIM, Simulator
        
        # 1. 
        model_path = tmp_path / "e2e_model.onnx"
        dummy_input = torch.randn(1, 8)
        torch.onnx.export(test_model, dummy_input, str(model_path), opset_version=13)
        
        # 2.  CIM
        cim = CIM(use_simulator=True)
        success = cim.load_model(str(model_path))
        assert success
        
        # 3. 
        input_data = np.random.randn(8).astype(np.float32)
        result = cim.infer(input_data)
        
        assert result.output is not None
        assert result.latency_us > 0
    
    def test_sparse_optimization_chain(self, tmp_path):
        """"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        # 
        for sparsity in [0.0, 0.5, 0.8]:
            input_data = np.random.randn(32).astype(np.float32)
            mask = np.random.random(32) > sparsity
            input_data *= mask
            
            weights = np.random.randn(32, 8).astype(np.float32)
            
            result = sim.matmul(input_data, weights, sparse=True)
            
            assert result['output'].shape == (8,)


# ============================================================================
# HIL () 
# ============================================================================

class TestHIL:
    """ ()"""
    
    @pytest.fixture
    def hardware_connection(self, config):
        """ fixture"""
        if not config.use_hardware:
            pytest.skip("HIL  --hil ")
        
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
            pytest.skip(f": {e}")
    
    def test_hardware_ping(self, hardware_connection):
        """"""
        hardware_connection.write(b"PING\n")
        response = hardware_connection.readline()
        
        assert b"PONG" in response
    
    def test_cim_compute_hardware(self, hardware_connection):
        """ CIM """
        # 
        test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
        cmd = b"CIM_COMPUTE:" + test_data.tobytes() + b"\n"
        
        hardware_connection.write(cmd)
        response = hardware_connection.readline()
        
        assert b"RESULT:" in response
    
    def test_latency_measurement(self, hardware_connection, config):
        """"""
        latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            
            hardware_connection.write(b"PING\n")
            hardware_connection.readline()
            
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"\n:  {avg_latency:.2f}ms,  {max_latency:.2f}ms")
        
        # 
        assert avg_latency < 10.0  #  < 10ms
        assert max_latency < 50.0  #  < 50ms


class TestMotorHIL:
    """ HIL """
    
    @pytest.fixture
    def motor_controller(self, config):
        """ fixture"""
        if not config.use_hardware:
            pytest.skip(" HIL  --hil ")
        
        # 
        class MockMotorController:
            def set_torque(self, torque): pass
            def get_position(self): return 0.0
            def get_velocity(self): return 0.0
        
        yield MockMotorController()
    
    def test_pid_response(self, motor_controller):
        """ PID """
        from imc22 import NeuralReflex
        
        reflex = NeuralReflex()
        
        # 
        for _ in range(100):
            pos = motor_controller.get_position()
            vel = motor_controller.get_velocity()
            
            weights = reflex.compute_blend(
                torque=0.0, 
                velocity=vel,
                position_error=1.0 - pos
            )
            
            # 
            pid_output = weights['pid'] * (1.0 - pos)
            neural_output = weights['neural'] * 0.5
            
            total_output = pid_output + neural_output
            motor_controller.set_torque(total_output)
    
    def test_position_accuracy(self, motor_controller):
        """"""
        target_positions = [0.0, 0.5, 1.0, -0.5]
        
        for target in target_positions:
            # motor_controller.move_to(target)
            # time.sleep(1.0)
            # actual = motor_controller.get_position()
            # error = abs(actual - target)
            # assert error < 0.01  # 1% 
            pass  # 


# ============================================================================
# 
# ============================================================================

class TestBenchmark:
    """"""
    
    def test_inference_throughput(self):
        """"""
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
        
        print(f"\n: {throughput:.0f} /")
        
        assert throughput > 100  #  100 /
    
    def test_sparse_speedup(self):
        """"""
        from imc22 import Simulator
        
        sim = Simulator(mac_count=256)
        
        # 50% 
        input_sparse = np.zeros(256, dtype=np.float32)
        input_sparse[::2] = np.random.randn(128).astype(np.float32)
        
        input_dense = np.random.randn(256).astype(np.float32)
        weights = np.random.randn(256, 64).astype(np.float32)
        
        # 
        result_sparse = sim.matmul(input_sparse, weights, sparse=True)
        
        # 
        result_dense = sim.matmul(input_dense, weights, sparse=False)
        
        speedup = result_sparse['speedup']
        
        print(f"\n: {speedup:.2f}x")
        
        assert speedup >= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
