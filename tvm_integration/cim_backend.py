"""
TVM BYOC Backend for Hive-Reflex CIM
====================================

This module defines the graph partitioning patterns to offload
supported operators to the CIM accelerator.
"""

import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.dataflow_pattern import is_op, wildcard

def make_pattern(op_name):
    """Create a basic pattern for a single operator"""
    return is_op(op_name)(wildcard(), wildcard())

def make_fc_pattern():
    """Pattern for Fully Connected (Dense): nn.dense + optional nn.bias_add + optional nn.relu"""
    dense = is_op('nn.dense')(wildcard(), wildcard())
    bias_add = is_op('nn.bias_add')(dense, wildcard())
    relu = is_op('nn.relu')(bias_add | dense)
    return relu | bias_add | dense

def make_cim_patterns():
    """Define supported patterns for CIM offloading"""
    return [
        ('cim.dense', make_fc_pattern()),
        # ('cim.conv2d', is_op('nn.conv2d')(wildcard(), wildcard())), # Todo
    ]

@tvm.ir.register_op_attr("nn.dense", "target.cim")
def dense_attr(attrs, args):
    """Check if dense layer is supported by hardware constraints"""
    # e.g. check if dimensions fit in SRAM
    return True

def annotate_cim_ops(mod, params):
    """
    Annotate Relay graph for CIM accelerator.
    High-level entry point for the backend.
    """
    class CIMAnnotator(relay.ExprMutator):
        def visit_call(self, call):
            # Simplified manual annotation logic for demo
            # In production, use MergeComposite and Pattern partitioning
            if call.op.name == 'nn.dense':
                return compiler_begin(super().visit_call(call), "cim")
            return super().visit_call(call)
            
    # Standard BYOC flow usually uses: 
    # transform.MergeCompilerRegions()
    # transform.PartitionGraph()
    # Here we define the pattern table for standard partitioning
    pattern_table = make_cim_patterns()
    mod = relay.transform.MergeComposite(pattern_table)(mod)
    mod = relay.transform.AnnotateTarget("cim")(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    
    return mod
