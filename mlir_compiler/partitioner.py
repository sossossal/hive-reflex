
import math

class CostModel:
    """
    Estimates execution cycles for CPU and CIM targets.
    """
    def __init__(self, sram_limit=64*1024, mac_efficiency=0.8):
        self.sram_limit = sram_limit
        self.mac_efficiency = mac_efficiency
        
        # Supported CIM Ops
        self.cim_ops = ['Gemm', 'MatMul', 'Conv', 'Relu'] 
        
        # Hard constraints (e.g. Layers that MUST be CPU)
        self.cpu_forced_ops = ['Softmax', 'ReduceMean', 'Reshape', 'Transpose']

    def is_cim_compatible(self, op_type, input_shape=None):
        if op_type in self.cpu_forced_ops:
            return False
        if op_type in self.cim_ops:
            return True
        # Default fallback to CPU for unknown ops
        return False

    def estimate_cpu_cycles(self, op_type, shape):
        """Estimate RISC-V Cycles (approx)"""
        # Linear layer: O(N*M) * 4 cycles/mac
        if op_type in ['Gemm', 'MatMul', 'fc']:
            macs = shape[0] * shape[1] # Simple approx
            return macs * 4 
        elif op_type in ['Relu']:
            return shape[0] * shape[1] * 2 # Compare + Branch
        elif op_type in ['Softmax']:
             # Exp + Sum + Div
             return shape[0] * shape[1] * 50
        return 1000 # Default overhead

    def estimate_cim_cycles(self, op_type, shape):
        """Estimate CIM Cycles (256 MACs parallel)"""
        if op_type in ['Gemm', 'MatMul', 'fc']:
            macs = shape[0] * shape[1]
            parallelism = 256
            cycles = math.ceil(macs / (parallelism * self.mac_efficiency))
            
            # Transfer Overhead (DMA setup)
            overhead = 50 
            return cycles + overhead
        elif op_type in ['Relu']:
            # Near zero cost if fused, small cost if standalone
            return shape[0] * shape[1] / 32 # Vectorized
        return float('inf') # Unsupported

class GraphPartitioner:
    """
    Splits a linear list of layers into subgraphs (CPU vs CIM).
    """
    def __init__(self, layers):
        self.layers = layers
        self.cost_model = CostModel()
        self.partitions = []

    def partition(self):
        """
        Greedy Partitioning Algorithm
        Returns list of partitions: [{'target': 'cim', 'layers': [...]}, ...]
        """
        current_partition = {'target': None, 'layers': []}
        
        for layer in self.layers:
            op_type = layer.get('op_type', layer.get('type')) # Handle both formats
            
            # 1. Determine best target for this layer
            shape = layer.get('shape', (10, 10)) # fallback shape
            
            is_compatible = self.cost_model.is_cim_compatible(op_type)
            
            if is_compatible:
                cim_cost = self.cost_model.estimate_cim_cycles(op_type, shape)
                cpu_cost = self.cost_model.estimate_cpu_cycles(op_type, shape)
                
                # Heuristic: If CIM is at least 2x faster, use it.
                # Otherwise keep on CPU to avoid transfer overhead.
                if cim_cost < (cpu_cost / 2):
                    best_target = 'cim'
                else:
                    best_target = 'cpu'
            else:
                best_target = 'cpu'
            
            # 2. Assign to partition
            if current_partition['target'] is None:
                current_partition['target'] = best_target
                current_partition['layers'].append(layer)
            elif current_partition['target'] == best_target:
                current_partition['layers'].append(layer)
            else:
                # Switch detected! Commit current partition
                self.partitions.append(current_partition)
                # Start new
                current_partition = {'target': best_target, 'layers': [layer]}
        
        # Commit last
        if current_partition['layers']:
            self.partitions.append(current_partition)
            
        return self.partitions

    def print_summary(self):
        print("\nUsing Cost-Based Heterogeneous Partitioning:")
        for i, p in enumerate(self.partitions):
            print(f"  Subgraph {i} [{p['target'].upper()}]: {len(p['layers'])} Ops")
            for l in p['layers']:
                print(f"    - {l['name']} ({l.get('op_type', 'Unknown')})")
