#!/usr/bin/env python3
"""
Transformer Operator Support for CIM Compiler
Handles decomposition and mapping of Transformer-specific layers:
- Multi-Head Attention (MHA) decomposition
- Softmax (Hardware friendly implementation)
- LayerNormalization
- Gelu/Silu Activations
"""

from typing import List, Dict, Optional
import numpy as np

class TransformerMapper:
    """
    Maps Transformer operations to CIM/RISC-V instructions.
    
    Since the current CIM core mainly accelerates Matrix Multiplication (MAC),
    operations like Softmax and LayerNorm are often hybrid:
    - MAC parts -> CIM Core
    - Vector parts -> RISC-V Vector Extension (RVV) or Optimized C
    """
    
    def __init__(self):
        pass

    def analyze_node(self, node) -> Optional[Dict]:
        """
        Analyze an ONNX node to check if it's a supported Transformer operator.
        Returns layer config dict if supported, else None.
        """
        op_type = node.op_type
        
        if op_type == 'Softmax':
            return self._analyze_softmax(node)
        elif op_type == 'LayerNormalization':
            return self._analyze_layernorm(node)
        elif op_type == 'RMSNorm': # Custom op / ONNX simplification
            return self._analyze_rmsnorm(node)
        elif op_type == 'MatMul':
            # Check if this is part of Attention (e.g. Q*K^T)
            return None 
        elif op_type in ['Gelu', 'Erf']:
            return {'type': 'activation', 'activation': 'gelu'}
        elif op_type in ['Sigmoid']: # SiLU = x * Sigmoid(x), usually decomposed or explicit 'Swish'
             # If we detect the fusion manually or just generic activation
             pass
        elif op_type == 'Mul':
            # Check for SiLU pattern: x * Sigmoid(x) handled in optimizer or just simple activation
            pass
        
        return None

    def _analyze_rmsnorm(self, node) -> Dict:
        """Analyze RMSNorm node (Simplified LayerNorm without mean subtraction)"""
        epsilon = 1e-5
        for attr in node.attribute:
           if attr.name == 'epsilon':
               epsilon = attr.f
               
        return {
            'name': node.name,
            'type': 'rmsnorm',
            'epsilon': epsilon,
            'inputs': list(node.input),
            'outputs': list(node.output)
        }

    def _analyze_softmax(self, node) -> Dict:
        """Analyze Softmax node"""
        axis = -1
        for attr in node.attribute:
            if attr.name == 'axis':
                axis = attr.i
        
        return {
            'name': node.name,
            'type': 'softmax',
            'axis': axis,
            'inputs': list(node.input),
            'outputs': list(node.output)
        }

    def _analyze_layernorm(self, node) -> Dict:
        """Analyze LayerNormalization node"""
        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == 'epsilon':
                epsilon = attr.f
                
        return {
            'name': node.name,
            'type': 'layernorm',
            'epsilon': epsilon,
            'inputs': list(node.input),
            'outputs': list(node.output)
        }

    def generate_c_code(self, layer: Dict, input_var: str, output_var: str) -> List[str]:
        """Generate C code for the operator"""
        op_type = layer.get('type')
        code = []
        
        if op_type == 'softmax':
            code = [
                f"    // Softmax (Axis {layer.get('axis', -1)})",
                f"    CIM_Softmax_Optimized({input_var}, {output_var}, layer_size);"
            ]
        elif op_type == 'layernorm':
            epsilon = layer.get('epsilon', 1e-5)
            # Assuming params (scale/bias) are loaded in weights or inputs
            code = [
                f"    // LayerNorm (eps={epsilon})",
                f"    // Note: Scale/Bias assumed to be handled in pre-processing or fused",
                f"    CIM_LayerNorm_RISCV({input_var}, {output_var}, layer_size, {epsilon}f);"
            ]
        elif op_type == 'activation' and layer.get('activation') == 'gelu':
             code = [
                f"    // Gelu Activation",
                f"    CIM_Gelu_Approx({input_var}, {output_var}, layer_size);"
            ]
        elif op_type == 'rmsnorm':
            epsilon = layer.get('epsilon', 1e-5)
            code = [
                f"    // RMSNorm (Llama style, eps={epsilon})",
                f"    CIM_RMSNorm_RISCV({input_var}, {output_var}, layer_size, {epsilon}f);"
            ]
        elif op_type == 'activation' and layer.get('activation') == 'silu':
             code = [
                f"    // SiLU / Swish Activation",
                f"    CIM_SiLU_Approx({input_var}, {output_var}, layer_size);"
            ]
            
        return code

    def fusion_rules(self):
        """Define fusion patterns for Transformer blocks"""
        return [
            # Q*K^T + Scale -> Mask -> Softmax
            ('MatMul', 'Div', 'Softmax'), 
            # Add + LayerNorm
            ('Add', 'LayerNormalization')
        ]
