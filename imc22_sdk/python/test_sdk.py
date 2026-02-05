#!/usr/bin/env python3
"""
IMC-22 Python SDK 
 SDK 
"""

import sys
import os

# 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("IMC-22 Python SDK ")
    print("=" * 50)
    print()
    
    # 
    print("[1] ...")
    try:
        from imc22 import (
            CIM, Power, DVFS, NeuralReflex, Simulator,
            PowerMode, DVFSFreq, CIMConfig, __version__
        )
        print(f"    IMC-22 SDK : {__version__}")
        print("    !")
    except Exception as e:
        print(f"    : {e}")
        return 1
    print()
    
    # CIM 
    print("[2] CIM ...")
    try:
        cim = CIM(use_simulator=True)
        print(f"    CIM : {cim.is_ready}")
        print(f"    : {cim.use_simulator}")
        print("    CIM !")
    except Exception as e:
        print(f"    CIM : {e}")
        return 1
    print()
    
    # Power 
    print("[3] Power ...")
    try:
        pwr = Power()
        pwr.set_mode(PowerMode.STANDBY)
        state = pwr.state
        print(f"    : {state.mode.name}")
        print(f"    : {state.voltage_mv}mV")
        print(f"    : {state.power_mw}mW")
        print("    Power !")
    except Exception as e:
        print(f"    Power : {e}")
        return 1
    print()
    
    # DVFS 
    print("[4] DVFS ...")
    try:
        dvfs = DVFS()
        dvfs.enable()
        dvfs.enable_auto_scale(util_low=50, util_high=200)
        dvfs.report_utilization(75)
        print(f"    : {dvfs.current_freq.name}")
        dvfs.report_utilization(250)  # 
        print(f"    : {dvfs.current_freq.name}")
        print("    DVFS !")
    except Exception as e:
        print(f"    DVFS : {e}")
        return 1
    print()
    
    # NeuralReflex 
    print("[5] NeuralReflex ...")
    try:
        reflex = NeuralReflex()
        weights = reflex.compute_blend(torque=5.0, velocity=1.2, position_error=0.1)
        print(f"    PID : {weights['pid']:.3f}")
        print(f"    Neural : {weights['neural']:.3f}")
        print(f"    : {weights['compliance']:.3f}")
        
        # 
        weights_high = reflex.compute_blend(torque=10.0, velocity=0.5)
        print(f"     PID : {weights_high['pid']:.3f}")
        print("    NeuralReflex !")
    except Exception as e:
        print(f"    NeuralReflex : {e}")
        return 1
    print()
    
    # Simulator 
    print("[6] Simulator ...")
    try:
        import numpy as np
        
        sim = Simulator(mac_count=256, data_width=8)
        
        input_data = np.random.randn(16).astype(np.float32)
        weights = np.random.randn(16, 8).astype(np.float32)
        
        result = sim.matmul(input_data, weights, sparse=True, threshold=2)
        
        print(f"    : {result['output'].shape}")
        print(f"    : {result['latency_s']*1000:.2f} ms")
        print(f"    : {result['sparsity']*100:.1f}%")
        print(f"    : {result['speedup']:.2f}x")
        print("    Simulator !")
    except Exception as e:
        print(f"    Simulator : {e}")
        return 1
    print()
    
    # 
    print("[7] ...")
    try:
        import numpy as np
        
        # 
        cim = CIM(use_simulator=True)
        dvfs = DVFS()
        reflex = NeuralReflex()
        
        #  DVFS
        dvfs.enable()
        dvfs.enable_auto_scale(50, 200)
        
        # 
        sensor_data = np.random.randn(8).astype(np.float32)
        
        # 
        weights = reflex.compute_blend(
            torque=sensor_data[0],
            velocity=sensor_data[1],
            position_error=sensor_data[2]
        )
        
        # 
        dvfs.report_utilization(int(abs(sensor_data[0]) * 25))
        
        print(f"    : PID={weights['pid']:.2f}, Neural={weights['neural']:.2f}")
        print(f"    DVFS : {dvfs.current_freq.name}")
        print("    !")
    except Exception as e:
        print(f"    : {e}")
        return 1
    print()
    
    print("=" * 50)
    print("!")
    print("=" * 50)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
