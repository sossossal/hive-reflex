#!/usr/bin/env python3
"""
Hive-Reflex 2.1 RTL 资源估算工具
无需 Vivado，使用静态分析估算 FPGA 资源和功耗

使用方法:
    python analyze_rtl.py                    # 分析所有 RTL
    python analyze_rtl.py --file sparse_cim  # 分析特定文件
"""

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# ============================================================================
# 资源估算模型 (基于 Xilinx UltraScale+)
# ============================================================================

RESOURCE_MODEL = {
    'reg': {
        'lut_per_bit': 0,
        'ff_per_bit': 1,
        'power_per_bit_uw': 0.5  # 典型动态功耗
    },
    'wire': {
        'lut_per_bit': 0.1,
        'power_per_bit_uw': 0.2
    },
    'always_comb': {
        'lut_per_block': 2
    },
    'always_ff': {
        'ff_per_block': 4,
        'lut_per_block': 1
    },
    'multiplier': {
        'dsp_8bit': 0.25,  # 4 个 8-bit 乘用 1 DSP
        'dsp_16bit': 0.5,
        'power_per_dsp_mw': 5
    },
    'memory': {
        'bram_per_kb': 0.5,  # 每 KB 约 0.5 BRAM
        'power_per_bram_mw': 2
    }
}

# 典型电压功耗关系
VOLTAGE_POWER_SCALE = {
    1.0: 1.0,    # Active (100%)
    0.6: 0.36,   # Standby (36% = 0.6^2)
    0.4: 0.16    # DeepSleep (16% = 0.4^2)
}


@dataclass
class RTLAnalysis:
    """RTL 分析结果"""
    filename: str
    lines: int
    registers: int
    wires: int
    always_blocks: int
    multipliers: int
    memory_bits: int
    estimated_luts: int
    estimated_ffs: int
    estimated_dsps: int
    estimated_brams: float
    estimated_power_mw: float


def count_bit_width(declaration: str) -> int:
    """解析位宽声明"""
    match = re.search(r'\[(\d+):(\d+)\]', declaration)
    if match:
        high, low = int(match.group(1)), int(match.group(2))
        return abs(high - low) + 1
    return 1


def analyze_rtl_file(filepath: str) -> RTLAnalysis:
    """分析单个 RTL 文件"""
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.count('\n')
    
    # 统计寄存器
    reg_patterns = re.findall(r'(?:reg|logic)\s+(?:signed\s+)?(?:\[\d+:\d+\])?\s*(\w+)', content)
    reg_bits = 0
    for match in re.finditer(r'(?:reg|logic)\s+(?:signed\s+)?(\[\d+:\d+\])?\s*\w+', content):
        width_str = match.group(1) or ''
        reg_bits += count_bit_width(width_str)
    
    # 统计线网
    wire_patterns = re.findall(r'(?:wire|input|output)\s+(?:signed\s+)?(?:\[\d+:\d+\])?\s*(\w+)', content)
    wire_bits = 0
    for match in re.finditer(r'(?:wire|input|output)\s+(?:signed\s+)?(\[\d+:\d+\])?\s*\w+', content):
        width_str = match.group(1) or ''
        wire_bits += count_bit_width(width_str)
    
    # 统计 always 块
    always_comb = len(re.findall(r'always\s*@\s*\(\s*\*\s*\)', content))
    always_comb += len(re.findall(r'always_comb', content))
    always_ff = len(re.findall(r'always\s*@\s*\(\s*(?:pos|neg)edge', content))
    always_ff += len(re.findall(r'always_ff', content))
    
    # 统计乘法器
    multipliers = len(re.findall(r'\*', content))
    # 过滤注释中的 *
    multipliers -= len(re.findall(r'/\*|\*/', content))
    multipliers = max(0, multipliers // 2)  # 粗略估计
    
    # 统计存储器
    memory_match = re.findall(r'\[\d+:\d+\]\s*\w+\s*\[0?:?(\d+)\]', content)
    memory_bits = sum(int(m) * 8 for m in memory_match) if memory_match else 0
    
    # 资源估算
    estimated_ffs = reg_bits
    estimated_luts = (wire_bits * RESOURCE_MODEL['wire']['lut_per_bit'] +
                      always_comb * RESOURCE_MODEL['always_comb']['lut_per_block'] +
                      always_ff * RESOURCE_MODEL['always_ff']['lut_per_block'])
    estimated_dsps = multipliers * RESOURCE_MODEL['multiplier']['dsp_8bit']
    estimated_brams = memory_bits / 1024 / 8 * RESOURCE_MODEL['memory']['bram_per_kb']
    
    # 功耗估算 (Active 模式)
    power_logic = (reg_bits * RESOURCE_MODEL['reg']['power_per_bit_uw'] +
                   wire_bits * RESOURCE_MODEL['wire']['power_per_bit_uw']) / 1000
    power_dsp = estimated_dsps * RESOURCE_MODEL['multiplier']['power_per_dsp_mw']
    power_bram = estimated_brams * RESOURCE_MODEL['memory']['power_per_bram_mw']
    estimated_power = power_logic + power_dsp + power_bram
    
    return RTLAnalysis(
        filename=os.path.basename(filepath),
        lines=lines,
        registers=reg_bits,
        wires=wire_bits,
        always_blocks=always_comb + always_ff,
        multipliers=multipliers,
        memory_bits=memory_bits,
        estimated_luts=int(estimated_luts),
        estimated_ffs=estimated_ffs,
        estimated_dsps=int(estimated_dsps),
        estimated_brams=estimated_brams,
        estimated_power_mw=estimated_power
    )


def analyze_rtl_directory(rtl_dir: str) -> List[RTLAnalysis]:
    """分析目录下所有 RTL 文件"""
    
    results = []
    
    for filename in os.listdir(rtl_dir):
        if filename.endswith('.v') or filename.endswith('.sv'):
            filepath = os.path.join(rtl_dir, filename)
            try:
                result = analyze_rtl_file(filepath)
                results.append(result)
            except Exception as e:
                print(f"  警告: 分析 {filename} 失败 - {e}")
    
    return results


def estimate_dvfs_power(base_power_mw: float) -> Dict[str, float]:
    """估算 DVFS 各模式功耗"""
    
    return {
        'active_1v_mw': base_power_mw,
        'standby_0.6v_mw': base_power_mw * VOLTAGE_POWER_SCALE[0.6],
        'deepsleep_0.4v_mw': base_power_mw * VOLTAGE_POWER_SCALE[0.4],
        'deepsleep_uw': base_power_mw * VOLTAGE_POWER_SCALE[0.4] * 1000,
        'deepsleep_nw': base_power_mw * VOLTAGE_POWER_SCALE[0.4] * 1000000
    }


def print_analysis_report(results: List[RTLAnalysis]):
    """打印分析报告"""
    
    print("\n" + "=" * 70)
    print("Hive-Reflex 2.1 RTL 资源估算报告")
    print("=" * 70)
    
    # 按模块输出
    print("\n模块资源估算:")
    print("-" * 70)
    print(f"{'文件名':<30} {'行数':>6} {'LUT':>8} {'FF':>8} {'DSP':>6} {'BRAM':>6}")
    print("-" * 70)
    
    total_luts = 0
    total_ffs = 0
    total_dsps = 0
    total_brams = 0
    total_power = 0
    
    for r in sorted(results, key=lambda x: x.filename):
        print(f"{r.filename:<30} {r.lines:>6} {r.estimated_luts:>8} "
              f"{r.estimated_ffs:>8} {r.estimated_dsps:>6} {r.estimated_brams:>6.1f}")
        total_luts += r.estimated_luts
        total_ffs += r.estimated_ffs
        total_dsps += r.estimated_dsps
        total_brams += r.estimated_brams
        total_power += r.estimated_power_mw
    
    print("-" * 70)
    print(f"{'合计':<30} {'':<6} {total_luts:>8} {total_ffs:>8} "
          f"{total_dsps:>6} {total_brams:>6.1f}")
    
    # FPGA 利用率估算 (ZCU102)
    zcu102_resources = {
        'luts': 274080,
        'ffs': 548160,
        'dsps': 2520,
        'brams': 912
    }
    
    print("\n" + "=" * 70)
    print("ZCU102 资源利用率估算:")
    print("-" * 70)
    print(f"  LUT:  {total_luts:>8} / {zcu102_resources['luts']:>8} ({total_luts/zcu102_resources['luts']*100:>5.2f}%)")
    print(f"  FF:   {total_ffs:>8} / {zcu102_resources['ffs']:>8} ({total_ffs/zcu102_resources['ffs']*100:>5.2f}%)")
    print(f"  DSP:  {total_dsps:>8} / {zcu102_resources['dsps']:>8} ({total_dsps/zcu102_resources['dsps']*100:>5.2f}%)")
    print(f"  BRAM: {total_brams:>8.1f} / {zcu102_resources['brams']:>8} ({total_brams/zcu102_resources['brams']*100:>5.2f}%)")
    
    # 功耗估算
    dvfs_power = estimate_dvfs_power(total_power)
    
    print("\n" + "=" * 70)
    print("DVFS 功耗估算:")
    print("-" * 70)
    print(f"  Active (1.0V, 100MHz):    {dvfs_power['active_1v_mw']:>10.2f} mW")
    print(f"  Standby (0.6V, 10MHz):    {dvfs_power['standby_0.6v_mw']:>10.2f} mW")
    print(f"  DeepSleep (0.4V, 1MHz):   {dvfs_power['deepsleep_uw']:>10.2f} μW")
    print(f"                            {dvfs_power['deepsleep_nw']:>10.0f} nW")
    
    # nW 目标验证
    target_nw = 100  # 100 nW 目标
    if dvfs_power['deepsleep_nw'] < target_nw:
        print(f"\n  ✓ DeepSleep 功耗满足 {target_nw} nW 目标!")
    else:
        print(f"\n  ⚠ DeepSleep 功耗 ({dvfs_power['deepsleep_nw']:.0f} nW) 超过 {target_nw} nW 目标")
        print(f"    需要进一步优化：时钟门控、更激进的电压缩放")
    
    print("\n" + "=" * 70)
    
    return {
        'total_luts': total_luts,
        'total_ffs': total_ffs,
        'total_dsps': total_dsps,
        'total_brams': total_brams,
        'power_active_mw': dvfs_power['active_1v_mw'],
        'power_deepsleep_nw': dvfs_power['deepsleep_nw']
    }


def main():
    parser = argparse.ArgumentParser(description='RTL 资源估算')
    parser.add_argument('--rtl-dir', default='../rtl', help='RTL 目录')
    parser.add_argument('--file', help='分析特定文件')
    parser.add_argument('--output', help='输出 JSON 文件')
    
    args = parser.parse_args()
    
    print("Hive-Reflex 2.1 RTL 资源估算工具")
    print("(无需 Vivado 的静态分析)")
    print()
    
    rtl_dir = os.path.abspath(args.rtl_dir)
    
    if args.file:
        # 分析特定文件
        filepath = os.path.join(rtl_dir, args.file + '.v')
        if not os.path.exists(filepath):
            filepath = os.path.join(rtl_dir, args.file + '.sv')
        if not os.path.exists(filepath):
            print(f"错误: 找不到文件 {args.file}")
            return 1
        
        results = [analyze_rtl_file(filepath)]
    else:
        # 分析所有文件
        print(f"分析目录: {rtl_dir}")
        results = analyze_rtl_directory(rtl_dir)
    
    if not results:
        print("未找到 RTL 文件")
        return 1
    
    summary = print_analysis_report(results)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'modules': [{
                    'filename': r.filename,
                    'lines': r.lines,
                    'estimated_luts': r.estimated_luts,
                    'estimated_ffs': r.estimated_ffs,
                    'estimated_dsps': r.estimated_dsps,
                    'estimated_brams': r.estimated_brams,
                    'estimated_power_mw': r.estimated_power_mw
                } for r in results],
                'summary': summary
            }, f, indent=2)
        print(f"\n结果已保存: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
