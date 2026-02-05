
import sys
import os
import subprocess
import time

def run_test(name, script_path, cwd=None):
    print(f"\n[Testing] {name}...")
    print("-" * 50)
    start_time = time.time()
    try:
        # Use python from current env
        cmd = [sys.executable, script_path]
        result = subprocess.run(
            cmd, 
            cwd=cwd if cwd else os.path.dirname(script_path),
            capture_output=True, 
            text=True,
            encoding='utf-8', 
            errors='replace' # Handle whatever windows throws at us
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASS ({duration:.2f}s)")
            # print(result.stdout) # Optional: Print concise output
            return True
        else:
            print(f"❌ FAIL ({duration:.2f}s)")
            print("ERROR OUTPUT:")
            print(result.stderr)
            print("STDOUT:")
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ CRASH: {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 Hive-Reflex v2.0 Feature Verification Suite")
    print("=============================================")
    
    tests = [
        # 1. Sim-Reflex Simulator
        ("Simulator Unit Test", os.path.join(base_dir, "simulator", "test_sim_reflex.py")),
        
        # 2. Performance Profiler
        ("Profiler Demo (MobileNet)", os.path.join(base_dir, "simulator", "run_profiling.py")),
        
        # 3. Knowledge Distillation
        ("Distillation Basic Demo", os.path.join(base_dir, "distillation", "run_distillation_demo.py")),
        
        # 4. Adaptive QAT
        ("Adaptive QAT (HW-in-Loop)", os.path.join(base_dir, "distillation", "run_qat_distillation.py")),
    ]
    
    results = []
    for name, path in tests:
        if os.path.exists(path):
            success = run_test(name, path)
            results.append((name, success))
        else:
            print(f"⚠️ SKIPPED: Script not found {path}")
            results.append((name, False))
            
    print("\n\n📊 Summary")
    print("=" * 50)
    all_pass = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {name}")
        if not success: all_pass = False
        
    if all_pass:
        print("\n✨ All v2.0 features are operational! Ready to evolve.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
