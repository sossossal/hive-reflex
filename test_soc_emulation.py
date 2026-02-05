
import os
import sys
import subprocess
import time

def test_soc_emulation():
    print(" Hive-Reflex SoC Co-Simulation Test")
    print("=====================================")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_zoo", "micro_kws.onnx")
    compiler_path = os.path.join(base_dir, "mlir_compiler", "codegen_cim.py")
    output_py = os.path.join(base_dir, "firmware_sim.py")
    
    # 1. Compile Model to Python Digital Twin
    print("[1] Compiling Micro-KWS to Python Firmware...")
    cmd_compile = [
        sys.executable, compiler_path,
        "--model", model_path,
        "--output-python", output_py
    ]
    
    try:
        subprocess.run(cmd_compile, check=True, text=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
        print(f" Compilation Failed: {e}")
        return
        
    if not os.path.exists(output_py):
        print(" Output file not found!")
        return
        
    print(f" Generated: {output_py}")
    
    # 2. Run the Digital Twin
    print("\n[2] Booting Virtual SoC...")
    cmd_boot = [sys.executable, output_py]
    
    try:
        start_t = time.time()
        result = subprocess.run(cmd_boot, capture_output=True, text=True, encoding='utf-8', errors='replace')
        end_t = time.time()
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f" Simulation Success ({end_t - start_t:.2f}s)")
            if "[SoC-FW] System Halted." in result.stdout:
                print("   Found valid boot log.")
        else:
            print(" Simulation Crashed")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            
    except Exception as e:
        print(f" Execution Error: {e}")

    # Cleanup
    # if os.path.exists(output_py): os.remove(output_py)

if __name__ == "__main__":
    test_soc_emulation()
