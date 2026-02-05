
import sys
import os

# Add parent path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from partitioner import GraphPartitioner

def test_auto_partition():
    print(" Testing Auto-Graph Partitioner")
    print("==================================")
    
    # 1. Define Mixed Graph
    # MatMul (CIM) -> Softmax (CPU) -> MatMul (CIM) -> Relu (CIM) -> ReduceMean (CPU)
    layers = [
        {'name': 'fc1', 'op_type': 'Gemm', 'shape': (128, 256)}, # Compatible, high arithmetic intensity
        {'name': 'sm1', 'op_type': 'Softmax', 'shape': (128, 10)}, # Incompatible (CPU forced)
        {'name': 'fc2', 'op_type': 'Gemm', 'shape': (128, 10)}, # Compatible
        {'name': 'act', 'op_type': 'Relu', 'shape': (128, 10)}, # Compatible, fuses with fc2
        {'name': 'avg', 'op_type': 'ReduceMean', 'shape': (1, 10)} # Incompatible
    ]
    
    # 2. Run Partitioner
    partitioner = GraphPartitioner(layers)
    partitions = partitioner.partition()
    partitioner.print_summary()
    
    # 3. Assertions
    # Expect 3 Partitions: [CIM, CPU, CIM, CPU]
    # Wait, FC2(CIM) and Relu(CIM) should merge.
    
    assert len(partitions) == 4, f"Expected 4 partitions, got {len(partitions)}"
    
    p0 = partitions[0]
    assert p0['target'] == 'cim'
    assert len(p0['layers']) == 1
    
    p1 = partitions[1]
    assert p1['target'] == 'cpu'
    assert p1['layers'][0]['name'] == 'sm1'
    
    p2 = partitions[2]
    assert p2['target'] == 'cim'
    assert len(p2['layers']) == 2 # fc2 + relu
    
    p3 = partitions[3]
    assert p3['target'] == 'cpu'
    
    print("\n Partition Logic Verified.")

if __name__ == "__main__":
    test_auto_partition()
