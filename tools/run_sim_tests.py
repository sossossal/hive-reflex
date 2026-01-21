"""
IMC-22 芯片仿真测试框架
用于自动化运行和验证仿真结果
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

class SimulationTester:
    """仿真测试器"""
    
    def __init__(self, qemu_path="qemu-system-riscv32"):
        self.qemu = qemu_path
        self.results = []
    
    def run_test(self, binary: str, timeout: int = 10) -> Tuple[bool, str]:
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"Running: {Path(binary).name}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                [
                    self.qemu,
                    '-nographic',
                    '-machine', 'virt',
                    '-kernel', binary
                ],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            
            output = result.stdout
            print(output)
            
            # 检查输出中的测试结果
            if "All tests PASSED" in output or "PASS" in output:
                return True, "PASS"
            elif "FAILED" in output or "FAIL" in output:
                return False, "FAIL"
            else:
                return False, "NO_RESULT"
                
        except subprocess.TimeoutExpired:
            print(f"✗ Test timed out after {timeout}s")
            return False, "TIMEOUT"
        except Exception as e:
            print(f"✗ Error running test: {e}")
            return False, "ERROR"
    
    def run_all_tests(self, test_dir: str):
        """运行所有测试"""
        test_path = Path(test_dir)
        
        # 查找所有测试二进制文件
        test_files = list(test_path.glob("test_*.bin"))
        
        if not test_files:
            print("No test files found!")
            return
        
        print(f"\nFound {len(test_files)} test(s)")
        
        # 运行每个测试
        for test_file in test_files:
            passed, status = self.run_test(str(test_file))
            self.results.append({
                'name': test_file.stem,
                'passed': passed,
                'status': status
            })
        
        # 打印汇总
        self.print_summary()
    
    def print_summary(self):
        """打印测试汇总"""
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        
        for result in self.results:
            status_icon = "✓" if result['passed'] else "✗"
            print(f"{status_icon} {result['name']:<30} {result['status']}")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        print(f"\n{'='*60}")
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        print(f"{'='*60}\n")
        
        return failed == 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IMC-22 Simulation Test Framework')
    parser.add_argument('--test-dir', default='build', 
                       help='Directory containing test binaries')
    parser.add_argument('--qemu', default='qemu-system-riscv32',
                       help='Path to QEMU executable')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Timeout for each test (seconds)')
    parser.add_argument('tests', nargs='*',
                       help='Specific tests to run')
    
    args = parser.parse_args()
    
    tester = SimulationTester(qemu_path=args.qemu)
    
    if args.tests:
        # 运行指定的测试
        for test in args.tests:
            passed, _ = tester.run_test(test, timeout=args.timeout)
            tester.results.append({
                'name': Path(test).stem,
                'passed': passed
            })
    else:
        # 运行所有测试
        tester.run_all_tests(args.test_dir)
    
    # 检查结果
    all_passed = tester.print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
