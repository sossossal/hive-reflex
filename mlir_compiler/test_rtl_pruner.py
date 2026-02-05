#!/usr/bin/env python3
"""
Test script for RTL Pruner
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock onnx if not available
try:
    import onnx
except ImportError:
    sys.modules['onnx'] = MagicMock()
    sys.modules['onnx.numpy_helper'] = MagicMock()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtl_pruner import RTLPruner

class TestRTLPruner(unittest.TestCase):
    def test_pruning_logic(self):
        # 1. Test Full Model
        layers_full = [
            {'op_type': 'Conv'},
            {'op_type': 'MatMul'},
            {'op_type': 'Softmax'},
            {'op_type': 'Relu'}
        ]
        peak_mem = 65536 # 64KB
        
        pruner = RTLPruner(layers_full, peak_mem)
        pruner.analyze()
        
        self.assertEqual(pruner.config['HAS_CONV'], 1)
        self.assertEqual(pruner.config['HAS_FC'], 1)
        self.assertEqual(pruner.config['HAS_TRANSFORMER'], 1)
        self.assertEqual(pruner.config['SRAM_NUM_BANKS'], 2) # 64KB / 32KB = 2
        
        # 2. Test Slim Model (Just FC)
        layers_slim = [
             {'op_type': 'Gemm'},
             {'op_type': 'Relu'}
        ]
        peak_mem = 10240 # 10KB
        
        pruner = RTLPruner(layers_slim, peak_mem)
        pruner.analyze()
        
        self.assertEqual(pruner.config['HAS_CONV'], 0)
        self.assertEqual(pruner.config['HAS_FC'], 1)
        self.assertEqual(pruner.config['HAS_TRANSFORMER'], 0)
        self.assertEqual(pruner.config['SRAM_NUM_BANKS'], 1) # Minimum 1
        
        # Generate output to check formatting
        output_file = "test_soc_config.vh"
        pruner.generate_config(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
            
        print("\nGenerated Config:")
        print(content)
        
        self.assertIn("`define ENABLE_CIM_GEMM_ENGINE", content)
        self.assertIn("// `define ENABLE_CIM_CONV_ENGINE", content)
        
        os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
