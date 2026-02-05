#!/usr/bin/env python3
"""
Test script for IO Integration in RTL Pruning
"""
import sys
import os
import unittest
import json
from unittest.mock import MagicMock

# Mock onnx
try:
    import onnx
except ImportError:
    sys.modules['onnx'] = MagicMock()
    sys.modules['onnx.numpy_helper'] = MagicMock()

# Peth setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtl_pruner import RTLPruner

class TestBackPrunerIO(unittest.TestCase):
    def test_io_config(self):
        # 1. Setup Dummy Data
        layers = [{'op_type': 'Gemm'}]
        peak_mem = 32768
        
        # 2. Create Temporary Config
        io_config = {
            "i2c_count": 2,
            "spi_count": 1,
            "uart_count": 0,
            "gpio_count": 8
        }
        config_path = "temp_io_config.json"
        with open(config_path, 'w') as f:
            json.dump(io_config, f)
            
        # 3. Run Pruner
        pruner = RTLPruner(layers, peak_mem)
        pruner.analyze()
        pruner.parse_io_config(config_path)
        
        # 4. Verify Internal State
        self.assertEqual(pruner.config['IO_I2C'], 2)
        self.assertEqual(pruner.config['IO_SPI'], 1)
        self.assertEqual(pruner.config['IO_UART'], 0)
        self.assertEqual(pruner.config['IO_GPIO_COUNT'], 8)
        
        # 5. Generate and Verify Logic
        output_file = "test_soc_io.vh"
        pruner.generate_config(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
            
        print("\nGenerated IO Config:")
        print(content)
        
        self.assertIn("`define ENABLE_I2C", content)
        self.assertIn("`define I2C_COUNT 2", content)
        self.assertIn("`define ENABLE_SPI", content)
        self.assertIn("// `define ENABLE_UART // Pruned", content)
        self.assertIn("`define GPIO_COUNT 8", content)

        # Cleanup
        if os.path.exists(config_path): os.remove(config_path)
        if os.path.exists(output_file): os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
