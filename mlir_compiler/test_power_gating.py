#!/usr/bin/env python3
"""
Test script for Power Gating in RTL Pruning
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock onnx
try:
    import onnx
except ImportError:
    sys.modules['onnx'] = MagicMock()
    sys.modules['onnx.numpy_helper'] = MagicMock()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtl_pruner import RTLPruner

class TestPowerGating(unittest.TestCase):
    def test_power_gating_generation(self):
        # Case A: Only GEMM used (e.g. simple signals)
        layers_a = [{'op_type': 'Gemm'}]
        pruner_a = RTLPruner(layers_a, 1024)
        pruner_a.analyze()
        
        outfile_a = "test_power_a.vh"
        pruner_a.generate_config(outfile_a)
        
        with open(outfile_a, 'r') as f:
            content = f.read()
            
        print("\n[Case A] GEMM Only:")
        print(content)
        
        self.assertIn("`define POWER_GATE_CONV_DOMAIN", content)
        self.assertIn("// `define POWER_GATE_GEMM_DOMAIN // Active", content)
        self.assertIn("`define POWER_GATE_TRANSFORMER_DOMAIN", content)

        # Case B: Conv + Transformer (Complex)
        layers_b = [{'op_type': 'Conv'}, {'op_type': 'Softmax'}]
        pruner_b = RTLPruner(layers_b, 1024)
        pruner_b.analyze()
        
        outfile_b = "test_power_b.vh"
        pruner_b.generate_config(outfile_b)
        
        with open(outfile_b, 'r') as f:
            content_b = f.read()
            
        print("\n[Case B] Conv + Transformer:")
        
        self.assertIn("// `define POWER_GATE_CONV_DOMAIN // Active", content_b)
        self.assertIn("`define POWER_GATE_GEMM_DOMAIN", content_b) # Implicitly not found
        self.assertIn("// `define POWER_GATE_TRANSFORMER_DOMAIN // Active", content_b)

        # Cleanup
        if os.path.exists(outfile_a): os.remove(outfile_a)
        if os.path.exists(outfile_b): os.remove(outfile_b)

if __name__ == '__main__':
    unittest.main()
