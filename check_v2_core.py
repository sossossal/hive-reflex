
import subprocess
import sys
import os

def run_test(name, cmd_args):
    print(f"\n[Test] Verifying: {name}...")
    print("-" * 50)
    try:
        # Use sys.executable to ensure same python env
        full_cmd = [sys.executable] + cmd_args
        result = subprocess.run(
            full_cmd, 
            capture_output=True, 
            text=True,
            check=False,
            encoding='utf-8', 
            errors='ignore'
        )
        
        if result.returncode == 0:
            print("PASS")
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                 print(f"   | {line}")
            return True
        else:
            print("FAIL")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("Hive-Reflex v2.0 Core Barrier Verification Suite")
    print("=================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Heterogeneous Compiler
    t1 = run_test("1. Heterogeneous Compiler (Auto-Partitioning)", 
                  [os.path.join(base_dir, "mlir_compiler", "test_partitioner.py")])
                  
    # 2. Accuracy Recovery (Calibration)
    t2 = run_test("2. Accuracy Recovery (On-Chip Calibration)", 
                  [os.path.join(base_dir, "test_soc_emulation.py")])
                  
    # 3. Tightly Coupled Arch (ISA)
    t3 = run_test("3. Tightly Coupled Arch (Custom CIM ISA)", 
                  [os.path.join(base_dir, "test_isa_decoder.py")])
                  
    # 4. Co-Simulation (SystemC)
    t4 = run_test("4. Co-Simulation (SystemC Export)", 
                  [os.path.join(base_dir, "test_systemc_export.py")])
                  
    print("\n" + "="*50)
    print("Final Report")
    print("-" * 50)
    print(f"1. Compiler:      {'PASS' if t1 else 'FAIL'}")
    print(f"2. Accuracy:      {'PASS' if t2 else 'FAIL'}")
    print(f"3. Architecture:  {'PASS' if t3 else 'FAIL'}")
    print(f"4. Co-Simulation: {'PASS' if t4 else 'FAIL'}")
    
    if t1 and t2 and t3 and t4:
        print("\nRELEASE CANDIDATE READY")
    else:
        print("\nISSUES DETECTED")

if __name__ == "__main__":
    main()
