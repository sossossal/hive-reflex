#!/usr/bin/env python3
"""
Hive-Reflex 2.1 功耗优化分析工具
计算时钟门控后的真实功耗估算

使用方法:
    python power_estimator.py
"""

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class PowerDomain:
    """电源域"""
    name: str
    base_power_mw: float
    clock_gated: bool = True
    activity_factor: float = 1.0


def estimate_power_with_clock_gating():
    """
    估算时钟门控后的功耗
    
    时钟门控策略:
    - DeepSleep: 99% 模块时钟关闭，仅保留唤醒逻辑
    - Standby: 70% 模块时钟关闭，仅活动模块运行
    - Active: 100% 时钟运行
    """
    
    print("=" * 60)
    print("Hive-Reflex 2.1 功耗优化分析")
    print("=" * 60)
    print()
    
    # 电源域定义
    power_domains = {
        'cim_core': PowerDomain('CIM 核心', 25.0, clock_gated=True),
        'sparse_mac': PowerDomain('稀疏 MAC', 15.0, clock_gated=True),
        'dvfs_ctrl': PowerDomain('DVFS 控制器', 5.0, clock_gated=True),
        'riscv_cpu': PowerDomain('RISC-V CPU', 8.0, clock_gated=True),
        'uart': PowerDomain('UART', 2.0, clock_gated=True),
        'wakeup': PowerDomain('唤醒逻辑', 0.01, clock_gated=False),  # 始终运行
    }
    
    total_base = sum(d.base_power_mw for d in power_domains.values())
    
    print(f"基础功耗 (所有模块 Active): {total_base:.2f} mW")
    print()
    
    # 模式配置
    modes = {
        'Active': {
            'voltage': 1.0,
            'frequency_mhz': 100,
            'activity': {
                'cim_core': 1.0,
                'sparse_mac': 0.8,  # 稀疏跳过 20%
                'dvfs_ctrl': 0.1,
                'riscv_cpu': 1.0,
                'uart': 0.3,
                'wakeup': 1.0,
            }
        },
        'Standby': {
            'voltage': 0.6,
            'frequency_mhz': 10,
            'activity': {
                'cim_core': 0.0,    # 时钟关闭
                'sparse_mac': 0.0,  # 时钟关闭
                'dvfs_ctrl': 0.1,
                'riscv_cpu': 0.1,   # 低速运行
                'uart': 0.0,        # 时钟关闭
                'wakeup': 1.0,
            }
        },
        'DeepSleep': {
            'voltage': 0.4,
            'frequency_mhz': 1,
            'activity': {
                'cim_core': 0.0,    # 时钟关闭
                'sparse_mac': 0.0,  # 时钟关闭
                'dvfs_ctrl': 0.0,   # 时钟关闭
                'riscv_cpu': 0.0,   # 时钟关闭
                'uart': 0.0,        # 时钟关闭
                'wakeup': 1.0,      # 始终运行
            }
        }
    }
    
    # 计算各模式功耗
    results = {}
    
    print("=" * 60)
    print("功耗估算（含时钟门控）")
    print("=" * 60)
    
    for mode_name, config in modes.items():
        voltage = config['voltage']
        freq_ratio = config['frequency_mhz'] / 100.0
        
        # 电压缩放: P ∝ V²
        voltage_scale = voltage ** 2
        
        # 频率缩放: P ∝ f
        freq_scale = freq_ratio
        
        mode_power = 0
        print(f"\n{mode_name} ({config['voltage']}V, {config['frequency_mhz']}MHz):")
        print("-" * 40)
        
        for domain_name, domain in power_domains.items():
            activity = config['activity'][domain_name]
            
            if domain.clock_gated and activity == 0:
                # 时钟门控: 仅静态功耗 (约 1% 动态功耗)
                domain_power = domain.base_power_mw * 0.01 * voltage_scale
            else:
                # 正常功耗: 动态 + 静态
                dynamic = domain.base_power_mw * activity * voltage_scale * freq_scale
                static = domain.base_power_mw * 0.05 * voltage_scale  # 5% 静态
                domain_power = dynamic + static
            
            mode_power += domain_power
            
            if domain_power > 0.0001:
                print(f"  {domain.name:<20}: {domain_power*1000:>10.2f} μW")
        
        print(f"  {'总计':<20}: {mode_power*1000:>10.2f} μW")
        
        results[mode_name] = {
            'power_mw': mode_power,
            'power_uw': mode_power * 1000,
            'power_nw': mode_power * 1000000
        }
    
    # 验证目标
    print("\n" + "=" * 60)
    print("目标验证")
    print("=" * 60)
    
    target_deepsleep_nw = 100  # 100 nW 目标
    actual_deepsleep_nw = results['DeepSleep']['power_nw']
    
    print(f"\n  DeepSleep 功耗: {actual_deepsleep_nw:.0f} nW")
    print(f"  目标功耗:       {target_deepsleep_nw} nW")
    
    if actual_deepsleep_nw <= target_deepsleep_nw:
        print(f"\n  ✓ 满足 nW 级待机功耗目标!")
    else:
        print(f"\n  ⚠ 当前功耗 ({actual_deepsleep_nw:.0f} nW) 略高于目标")
        print(f"  建议优化:")
        print(f"    - 进一步降低唤醒逻辑时钟 (0.1MHz)")
        print(f"    - 使用更低泄漏工艺 (ULL 库)")
        print(f"    - 添加电源门控 (Power Gating)")
    
    # 节能效果
    print("\n" + "=" * 60)
    print("节能效果")
    print("=" * 60)
    
    active_power = results['Active']['power_mw']
    standby_power = results['Standby']['power_uw']
    deepsleep_power = results['DeepSleep']['power_nw']
    
    print(f"\n  Active → Standby:   {active_power:.2f} mW → {standby_power:.2f} μW")
    print(f"                      节能 {(1 - standby_power/1000/active_power)*100:.1f}%")
    
    print(f"\n  Active → DeepSleep: {active_power:.2f} mW → {deepsleep_power:.0f} nW")
    print(f"                      节能 {(1 - deepsleep_power/1000000/active_power)*100:.4f}%")
    
    # 典型使用场景功耗
    print("\n" + "=" * 60)
    print("典型使用场景")
    print("=" * 60)
    
    scenarios = [
        ('高负载 CIM 推理', 1.0, 0.0, 0.0),
        ('空闲等待 (1s)', 0.1, 0.8, 0.1),
        ('低功耗监控', 0.01, 0.09, 0.9),
        ('极低功耗待机', 0.0, 0.0, 1.0),
    ]
    
    for scenario, active_pct, standby_pct, deepsleep_pct in scenarios:
        avg_power = (active_power * active_pct + 
                     standby_power / 1000 * standby_pct +
                     deepsleep_power / 1000000 * deepsleep_pct)
        print(f"\n  {scenario}:")
        print(f"    Active: {active_pct*100:.0f}%, Standby: {standby_pct*100:.0f}%, "
              f"DeepSleep: {deepsleep_pct*100:.0f}%")
        print(f"    平均功耗: {avg_power:.4f} mW ({avg_power*1000:.2f} μW)")
    
    print("\n" + "=" * 60)
    
    return results


def main():
    results = estimate_power_with_clock_gating()
    
    # 保存结果
    output = {
        'active_mw': results['Active']['power_mw'],
        'standby_uw': results['Standby']['power_uw'],
        'deepsleep_nw': results['DeepSleep']['power_nw'],
        'clock_gating': True,
        'target_met': results['DeepSleep']['power_nw'] <= 100
    }
    
    with open('../reports/power_estimate.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n结果已保存: ../reports/power_estimate.json")


if __name__ == '__main__':
    main()
