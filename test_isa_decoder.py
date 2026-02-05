
import sys
import os

# Ensure paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.sim_reflex import SimReflex
from simulator.profiler import CIM_ISA

def test_isa_simulation():
    print(" CIM ISA Co-Processor Unit Test")
    print("================================")
    
    # 1. Setup
    sim = SimReflex(sram_size_kb=64)
    isa = CIM_ISA(sim)
    
    # 2. Mock Data in TCM (0x0000)
    # Input Vector of size 10
    sim.mem_write(0, bytes([1]*10)) # All 1s (approx)
    
    print("[ISA] Decoding Instruction: CIM.VMM (Vector-Matrix Multiply)")
    # Instruction: CIM.VMM rd=0x4000, rs1=0x0000, rs2=10(rows), imm=10(cols)
    opcode = CIM_ISA.OP_CIM_VMM
    rd_addr = 0x4000
    rs1_addr = 0x0000
    rows = 10
    cols = 10
    
    cycles = isa.decode_and_execute(opcode, rd_addr, rs1_addr, rows, immediate=cols)
    
    print(f"  -> Decoded & Executed in {cycles} pipeline cycles (+ compute latency)")
    print(f"  -> Simulator Stats: {sim.stats}")
    
    # Verify Compute Happened
    # Since we didn't load weights, sim_reflex uses random/default weights? 
    # Actually sim_reflex.cim_fully_connected needs weights loaded at config_addr.
    # But for Behavioral Simulation, it mainly counts cycles and moves memory.
    
    if sim.stats['mac_ops'] > 0:
        print(" SUCCESS: Instruction triggered MAC hardware.")
    else:
        print(" FAILURE: No MAC ops recorded.")
        sys.exit(1)

if __name__ == "__main__":
    test_isa_simulation()
