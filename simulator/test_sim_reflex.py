
import unittest
import numpy as np
import os
import sys

# Add parent dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim_reflex import SimReflex

class TestSimReflex(unittest.TestCase):
    def test_fc_layer(self):
        # 1. Setup Simulator
        sim = SimReflex()
        
        # 2. Prepare Data
        # Input: [1.0, 2.0] (Float32)
        input_data = np.array([1.0, 2.0], dtype=np.float32)
        
        # Weights: [[1, 2], [3, 4]] (Int8) -> Stored in Flash
        # Shape: (2, 2)
        # Expected Output: [1*1 + 2*2, 1*3 + 2*4] = [5.0, 11.0]
        # Bias: [0.0, 1.0] (Float32) -> Output becomes [5.0, 12.0]
        
        weights = np.array([1, 2, 3, 4], dtype=np.int8)
        bias = np.array([0.0, 1.0], dtype=np.float32)
        
        # Pack Flash: Weights + Bias
        flash_data = weights.tobytes() + bias.tobytes()
        sim.load_weights(flash_data)
        
        # Write Input to SRAM Addr 0
        sim.mem_write(0, input_data.tobytes())
        
        # 3. Execute CIM FC
        # cim_fully_connected(input_addr, output_addr, weight_offset, in_dim, out_dim)
        sim.cim_fully_connected(
            input_addr=0,
            output_addr=100,
            weight_offset=0,
            input_dim=2,
            output_dim=2,
            relu=1
        )
        
        # 4. Read Output
        output_bytes = sim.mem_read(100, 8) # 2 floats * 4
        output_floats = np.frombuffer(output_bytes, dtype=np.float32)
        
        print(f"Input: {input_data}")
        print(f"Weights: {weights}")
        print(f"Bias: {bias}")
        print(f"Output: {output_floats}")
        
        # Verify
        self.assertTrue(np.allclose(output_floats, [5.0, 12.0]))
        self.assertEqual(sim.stats['mac_ops'], 4)

if __name__ == '__main__':
    unittest.main()
