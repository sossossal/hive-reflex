import numpy as np
import struct

class SimReflex:
    """
    Hive-Reflex Behavior Simulator (Bit-Approximate)
    Models the CIM accelerator and RISC-V environment in Python.
    """
    def __init__(self, sram_size_kb=64, flash_size_mb=4):
        self.sram = bytearray(sram_size_kb * 1024)
        self.flash = bytearray(flash_size_mb * 1024 * 1024)
        self.registers = {'PC': 0, 'SP': 0}
        
        # Performance Counters
        self.stats = {
            'cycles': 0,
            'mac_ops': 0,
            'sram_reads': 0,
            'sram_writes': 0
        }

    def load_weights(self, weights_bin):
        """Load weights binary into simulated Flash"""
        size = len(weights_bin)
        self.flash[0:size] = weights_bin
        print(f"[Sim] Loaded {size} bytes weights to Flash")

    def mem_write(self, addr, data):
        """Write bytes to SRAM"""
        size = len(data)
        if addr + size > len(self.sram):
            raise MemoryError(f"SRAM Write Out of Bounds: {addr}")
        self.sram[addr:addr+size] = data
        self.stats['sram_writes'] += size

    def mem_read(self, addr, size):
        """Read bytes from SRAM"""
        if addr + size > len(self.sram):
            raise MemoryError(f"SRAM Read Out of Bounds: {addr}")
        self.stats['sram_reads'] += size
        return self.sram[addr:addr+size]

    def cim_fully_connected(self, input_addr, output_addr, weight_offset, input_dim, output_dim, relu=0):
        """
        Simulate CIM_FullyConnected hardware instruction.
        Performs Matrix Vector Multiplication (GEMM) using Int8/Float hybrid logic.
        
        Real Hardware: 
          - Input: Float32 (converted to Int8 on the fly or pre-quantized)
          - Weights: Int8 (stored in Flash)
          - Output: Float32
        
        Simulator:
          - We use numpy for behavior, but enforce Int8 casting to check precision.
        """
        # 1. Read Input (Float32)
        input_bytes = self.mem_read(input_addr, input_dim * 4)
        input_floats = np.frombuffer(input_bytes, dtype=np.float32)

        # 2. Read Weights (Int8 from Flash)
        # Weight Shape: [OutputDim, InputDim] usually, packed linearly
        weight_size = input_dim * output_dim
        weight_bytes = self.flash[weight_offset : weight_offset + weight_size]
        weights_int8 = np.frombuffer(weight_bytes, dtype=np.int8).reshape(output_dim, input_dim)
        
        # 3. Simulate Computation (Quantization Aware)
        #  S = Scale Factor (Simplified for now, assume 1.0 or stored separately)
        #  Y = (X_int8 * W_int8) * Scale
        
        #  Naive Float simulation for now (v0.1)
        #  TODO v0.2: Implement bit-true Int8 MAC
        res = np.dot(weights_int8.astype(np.float32), input_floats)
        
        # Bias (Assuming bias follows weights in Flash as Float32 - Simulating the Firmware logic)
        bias_offset = weight_offset + weight_size
        bias_size = output_dim * 4
        bias_bytes = self.flash[bias_offset : bias_offset + bias_size]
        bias_floats = np.frombuffer(bias_bytes, dtype=np.float32)
        
        res += bias_floats
        
        # 4. Activation
        if relu:
            res = np.maximum(res, 0)
            
        # 5. Write Output
        self.mem_write(output_addr, res.tobytes())
        
        # Stats
        self.stats['mac_ops'] += input_dim * output_dim
        self.stats['cycles'] += (input_dim * output_dim) / 32 # Assuming 32 MACs/cycle

    def cim_conv2d(self, input_addr, output_addr, weight_offset, in_ch, out_ch, height, width, k=3, stride=1, pad=1):
        """
        Simulate Convolution - Placeholder
        """
        pass

    def run_inference_test(self, input_data, model_config):
        """
        Run a simulated inference based on a simplified config
        """
        # Load Input to Heap (Addr 0)
        self.mem_write(0, input_data.tobytes())
        
        current_input_addr = 0
        current_output_addr = len(input_data) * 4 
        
        print("[Sim] Starting Inference...")
        
        # Very simple linear execution of layers
        for layer in model_config['layers']:
            if layer['type'] == 'fc':
                shape = eval(layer['shape']) # (Out, In)
                # print(f"  Execute FC: {shape}")
                self.cim_fully_connected(
                    current_input_addr, 
                    current_output_addr,
                    0, # TODO: Track simulated weight offset
                    shape[1], # In (Cols)
                    shape[0], # Out (Rows)
                    relu=1
                )
                # Swap buffers
                current_input_addr = current_output_addr
                current_output_addr += shape[0] * 4 # Next buffer
                
        print(f"[Sim] Inference Done. Cycles: {int(self.stats['cycles'])}")
        
        # Return result from last output address
        # Size? Need to track from last layer shape
        return self.mem_read(current_input_addr, 40) # Arbitrary size read for debug

