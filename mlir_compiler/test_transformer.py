#!/usr/bin/env python3
"""
Test script for Transformer Compiler Support
Generates a dummy ONNX model with Transformer ops and verifies code generation.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from codegen_cim import CIMCodeGenerator

def create_dummy_transformer_model(output_path="dummy_transformer.onnx"):
    """Create a simple ONNX model with Softmax and LayerNorm"""
    
    # Input
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 12, 64])
    
    # Softmax Node
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['input'],
        outputs=['prob'],
        axis=-1,
        name='Softmax_1'
    )
    
    # LayerNorm Params (simplified, usually has scale/bias inputs)
    # For this test we just need the node to exist
    layernorm_node = helper.make_node(
        'LayerNormalization',
        inputs=['prob'], # scale/bias skipped for brevity in dummy
        outputs=['output'],
        epsilon=1e-5,
        name='LayerNorm_1'
    )
    
    # Graph
    graph = helper.make_graph(
        [softmax_node, layernorm_node],
        'DummyTransformer',
        [input_info],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 12, 64])]
    )
    
    # Model
    model = helper.make_model(graph, producer_name='hive-reflex-test')
    onnx.save(model, output_path)
    print(f"Created dummy model: {output_path}")
    return output_path

def test_code_generation():
    model_path = create_dummy_transformer_model()
    
    model = onnx.load(model_path)
    generator = CIMCodeGenerator(model)
    
    output_c = "test_transformer_gen.c"
    generator.generate(output_c, "test_weights.bin", "test_config.json")
    
    # Verify content
    with open(output_c, 'r') as f:
        content = f.read()
        
    print("\nVerifying Generated Code...")
    
    checks = [
        ("CIM_Softmax_Optimized", "Softmax support"),
        ("CIM_LayerNorm_RISCV", "LayerNorm support")
    ]
    
    all_passed = True
    for signature, description in checks:
        if signature in content:
            print(f" {description}: Found {signature}")
        else:
            print(f" {description}: Missing {signature}")
            all_passed = False
            
    if all_passed:
        print("\n All Transformer Compiler tests passed!")
    else:
        print("\n Some tests failed.")

if __name__ == "__main__":
    test_code_generation()
