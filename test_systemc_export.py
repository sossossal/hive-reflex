
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulator.systemc_exporter import SystemCExporter

def test_export():
    print(" SystemC Co-Simulation Export Test")
    print("====================================")
    
    out_dir = os.path.join(os.path.dirname(__file__), "sc_export")
    os.makedirs(out_dir, exist_ok=True)
    
    config = {'mac_size': 1024, 'sram_kb': 256}
    exporter = SystemCExporter(out_dir, config)
    exporter.generate()
    
    # Verify files
    h_path = os.path.join(out_dir, "cim_core.h")
    cpp_path = os.path.join(out_dir, "cim_core.cpp")
    
    if os.path.exists(h_path) and os.path.exists(cpp_path):
        print(f" Generated files found in {out_dir}")
        
        with open(h_path, 'r') as f:
            content = f.read()
            # Check for standard SystemC class definition
            if "class CIM_Core : public sc_module" in content and "MAC_ARRAY_SIZE = 1024" in content:
                print(" Header content valid (sc_module base class, Config Correct)")
            else:
                print(" Header content mismatch")
                print("--- CONTENT START ---")
                print(content)
                print("--- CONTENT END ---")
                sys.exit(1)
                
    else:
        print(" Files missing")
        sys.exit(1)

if __name__ == "__main__":
    test_export()
