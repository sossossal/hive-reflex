#!/usr/bin/env python3
"""
IMC-22 Python SDK 测试脚本
验证所有 SDK 模块功能
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("IMC-22 Python SDK 功能测试")
    print("=" * 50)
    print()
    
    # 导入测试
    print("[1] 导入测试...")
    try:
        from imc22 import (
            CIM, Power, DVFS, NeuralReflex, Simulator,
            PowerMode, DVFSFreq, CIMConfig, __version__
        )
        print(f"    IMC-22 SDK 版本: {__version__}")
        print("    导入成功!")
    except Exception as e:
        print(f"    导入失败: {e}")
        return 1
    print()
    
    # CIM 测试
    print("[2] CIM 测试...")
    try:
        cim = CIM(use_simulator=True)
        print(f"    CIM 就绪: {cim.is_ready}")
        print(f"    使用模拟器: {cim.use_simulator}")
        print("    CIM 测试通过!")
    except Exception as e:
        print(f"    CIM 测试失败: {e}")
        return 1
    print()
    
    # Power 测试
    print("[3] Power 测试...")
    try:
        pwr = Power()
        pwr.set_mode(PowerMode.STANDBY)
        state = pwr.state
        print(f"    电源模式: {state.mode.name}")
        print(f"    电压: {state.voltage_mv}mV")
        print(f"    功率: {state.power_mw}mW")
        print("    Power 测试通过!")
    except Exception as e:
        print(f"    Power 测试失败: {e}")
        return 1
    print()
    
    # DVFS 测试
    print("[4] DVFS 测试...")
    try:
        dvfs = DVFS()
        dvfs.enable()
        dvfs.enable_auto_scale(util_low=50, util_high=200)
        dvfs.report_utilization(75)
        print(f"    当前频率: {dvfs.current_freq.name}")
        dvfs.report_utilization(250)  # 高利用率
        print(f"    高负载频率: {dvfs.current_freq.name}")
        print("    DVFS 测试通过!")
    except Exception as e:
        print(f"    DVFS 测试失败: {e}")
        return 1
    print()
    
    # NeuralReflex 测试
    print("[5] NeuralReflex 测试...")
    try:
        reflex = NeuralReflex()
        weights = reflex.compute_blend(torque=5.0, velocity=1.2, position_error=0.1)
        print(f"    PID 权重: {weights['pid']:.3f}")
        print(f"    Neural 权重: {weights['neural']:.3f}")
        print(f"    合规度: {weights['compliance']:.3f}")
        
        # 高负载测试
        weights_high = reflex.compute_blend(torque=10.0, velocity=0.5)
        print(f"    高负载 PID 权重: {weights_high['pid']:.3f}")
        print("    NeuralReflex 测试通过!")
    except Exception as e:
        print(f"    NeuralReflex 测试失败: {e}")
        return 1
    print()
    
    # Simulator 测试
    print("[6] Simulator 测试...")
    try:
        import numpy as np
        
        sim = Simulator(mac_count=256, data_width=8)
        
        input_data = np.random.randn(16).astype(np.float32)
        weights = np.random.randn(16, 8).astype(np.float32)
        
        result = sim.matmul(input_data, weights, sparse=True, threshold=2)
        
        print(f"    输出形状: {result['output'].shape}")
        print(f"    延迟: {result['latency_s']*1000:.2f} ms")
        print(f"    稀疏率: {result['sparsity']*100:.1f}%")
        print(f"    加速比: {result['speedup']:.2f}x")
        print("    Simulator 测试通过!")
    except Exception as e:
        print(f"    Simulator 测试失败: {e}")
        return 1
    print()
    
    # 综合测试
    print("[7] 综合流程测试...")
    try:
        import numpy as np
        
        # 模拟完整推理流程
        cim = CIM(use_simulator=True)
        dvfs = DVFS()
        reflex = NeuralReflex()
        
        # 启用 DVFS
        dvfs.enable()
        dvfs.enable_auto_scale(50, 200)
        
        # 模拟传感器数据
        sensor_data = np.random.randn(8).astype(np.float32)
        
        # 计算自适应权重
        weights = reflex.compute_blend(
            torque=sensor_data[0],
            velocity=sensor_data[1],
            position_error=sensor_data[2]
        )
        
        # 报告利用率
        dvfs.report_utilization(int(abs(sensor_data[0]) * 25))
        
        print(f"    自适应权重: PID={weights['pid']:.2f}, Neural={weights['neural']:.2f}")
        print(f"    DVFS 频率: {dvfs.current_freq.name}")
        print("    综合流程测试通过!")
    except Exception as e:
        print(f"    综合流程测试失败: {e}")
        return 1
    print()
    
    print("=" * 50)
    print("所有测试通过!")
    print("=" * 50)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
