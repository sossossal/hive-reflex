#!/usr/bin/env python3
"""
Hive-Reflex Model Zoo Builder
Exports all reference models to ONNX and attempts to compile them.
"""
import os
import subprocess
import sys
import glob

def run_script(script_path):
    print(f"\n🚀 Running {script_path}...")
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ Success: {script_path}")
            return True
        else:
            print(result.stdout)
            print(result.stderr)
            print(f"❌ Failed: {script_path}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def build_all():
    print("🦁 Building Hive-Reflex Model Zoo 🦁")
    print("=" * 40)
    
    # 1. Export Scripts
    zoo_dir = os.path.dirname(os.path.abspath(__file__))
    export_scripts = glob.glob(os.path.join(zoo_dir, "export_*.py"))
    
    success_count = 0
    total_count = len(export_scripts)
    
    for script in export_scripts:
        if run_script(script):
            success_count += 1
            
    print("=" * 40)
    print(f"Build Complete: {success_count}/{total_count} Models Exported.")
    
    # 2. Compilation (Optional - if ONNX exists)
    # We would loop over *.onnx and call codegen_cim.py here
    
if __name__ == "__main__":
    build_all()
