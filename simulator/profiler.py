
import sys
import os
import json
import numpy as np

# Adjust path to find simulator
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from simulator.sim_reflex import SimReflex

class PerformanceProfiler:
    """
    Hive-Reflex Performance Profiler.
    Runs a model on SimReflex and collects detailed cycle-accurate stats.
    """
    def __init__(self, mac_array_size=256, frequency_mhz=100):
        self.sim = SimReflex()
        self.mac_array_size = mac_array_size
        self.frequency_mhz = frequency_mhz
        self.layer_stats = []

class CIM_ISA:
    """
    Tightly-Coupled Custom ISA Extension for RISC-V.
    Simulates the decoding and execution of 'custom-X' instructions.
    
    Instruction Format (Simulated):
    [OpCode (8b)] | [Rd (8b)] | [Rs1 (8b)] | [Rs2 (8b)]
    """
    OP_CIM_RESET = 0x00
    OP_CIM_LOAD  = 0x01 # Load weights to macro
    OP_CIM_VMM   = 0x02 # Vector-Matrix Multiply
    OP_CIM_ACT   = 0x03 # Activation (Relu/Tanh)
    
    def __init__(self, simulator):
        self.sim = simulator
        
    def decode_and_execute(self, opcode, rd, rs1, rs2, immediate=0):
        """
        Mimics CPU Pipeline Stage: Decode -> Execute
        """
        cycles = 1 # Base decode cost
        
        if opcode == self.OP_CIM_RESET:
            self.sim.reset_stats()
            
        elif opcode == self.OP_CIM_VMM:
            # rs1: Input TCM Address
            # rs2: Output Size (Rows)
            # immediate: Input Size (Cols)
            in_addr = rs1
            out_rows = rs2
            in_cols = immediate
            
            # Tightly Coupled Execution (Direct MAC array access)
            # Note: We reuse the high-level behavioral model for correctness
            # but model the latency as if it triggered the hardware sequencer directly.
            self.sim.cim_fully_connected(in_addr, rd, 0, in_cols, out_rows, relu=0)
            cycles += 1 # Dispatch overhead is low in TCA
            
        elif opcode == self.OP_CIM_ACT:
            # activation logic
            pass
            
        return cycles


    def profile_model(self, model_config):
        """
        Run simulation trace and record stats per layer.
        """
        print(f"[Profiler] Profiling Model: {model_config.get('model_name', 'Unknown')}")
        print(f"   HW Config: {self.mac_array_size} MACs @ {self.frequency_mhz} MHz")
        print("=" * 65)
        print(f"{'Layer Name':<20} | {'Type':<10} | {'Cycles':<10} | {'Time(us)':<10} | {'MAC Util':<8}")
        print("-" * 65)
        
        total_cycles = 0
        
        # Simulate loading weights (One-time cost)
        # In a real profiler, we'd count flash read cycles here.
        
        # Current memory pointers (Virtual)
        curr_in = 0
        curr_out = 16384 # 0x4000 (16KB offset) 
        
        for i, layer in enumerate(model_config['layers']):
            stat = {'name': layer['name'], 'type': layer['type']}
            
            # Snapshot stats before
            cycles_start = self.sim.stats['cycles']
            macs_start = self.sim.stats['mac_ops']
            
            # Execute (Simulated)
            # We map generic layers to SimReflex calls
            if layer['type'] == 'fc':
                # Parse shape "[Out, In]" string to tuple
                # Note: config generator currently saves tuple as string, e.g. "(64, 128)"
                shape_str = layer['shape'].replace('(', '').replace(')', '')
                shape = [int(s) for s in shape_str.split(',')]
                
                output_dim, input_dim = shape
                
                self.sim.cim_fully_connected(
                    curr_in, curr_out, 0, input_dim, output_dim, relu=1
                )
                
            elif layer['type'] == 'conv': 
                # Placeholder for Conv support in SimReflex
                # For now, we estimate based on MACs if SimReflex doesn't support it
                pass
            
            # Snapshot stats after
            cycles_end = self.sim.stats['cycles']
            macs_end = self.sim.stats['mac_ops']
            
            # Calculate Delta
            layer_cycles = cycles_end - cycles_start
            layer_macs = macs_end - macs_start
            
            total_cycles += layer_cycles
            
            # Metrics
            # Theoretical max MACs = Cycles * MAC_Array_Size
            # Utilization = Actual MACs / Theoretical Max
            if layer_cycles > 0:
                utilization = layer_macs / (layer_cycles * self.mac_array_size)
            else:
                utilization = 0
                
            time_us = layer_cycles / self.frequency_mhz
            
            stat['cycles'] = int(layer_cycles)
            stat['time_us'] = time_us
            stat['util'] = utilization
            self.layer_stats.append(stat)
            
            print(f"{layer['name']:<20} | {layer['type']:<10} | {int(layer_cycles):<10} | {time_us:<10.2f} | {utilization*100:<6.1f}%")
            
            # Ping-Pong Buffer Strategy (Safe within 64KB)
            # Buffer A: 0 (0x0000)
            # Buffer B: 16384 (0x4000)
            # Max layer output size must trigger error if > 16KB
            
            if curr_out == 16384:
                 curr_in = 16384
                 curr_out = 0
            else:
                 curr_in = 0
                 curr_out = 16384

        print("-" * 65)
        print(f"Total Cycles: {int(total_cycles)}")
        print(f"Total Time  : {total_cycles / self.frequency_mhz / 1000:.3f} ms")
        print("=" * 65)

    def export_report(self, output_path):
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'type', 'cycles', 'time_us', 'util'])
            writer.writeheader()
            writer.writerows(self.layer_stats)
        print(f"📄 Report saved to {output_path}")

if __name__ == "__main__":
    # Test stub
    pass
