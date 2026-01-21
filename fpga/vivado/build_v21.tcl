# Hive-Reflex 2.1 Vivado 综合脚本
# 支持稀疏计算、DVFS 和功耗分析
#
# 使用方法:
#   vivado -mode batch -source build_v21.tcl
#   或在 Vivado GUI 中: source build_v21.tcl

# ============================================================================
# 配置参数
# ============================================================================

set project_name "hive_reflex_v21"
set top_module "hive_reflex_top"
set target_part "xczu9eg-ffvb1156-2-e"  # ZCU102
set num_jobs 8

# 版本信息
set version "2.1.0"
puts "=============================================="
puts "Hive-Reflex $version FPGA 综合"
puts "=============================================="
puts ""

# ============================================================================
# 创建项目
# ============================================================================

# 检查项目是否存在
if {[file exists ./vivado_project/${project_name}.xpr]} {
    puts "打开现有项目..."
    open_project ./vivado_project/${project_name}.xpr
} else {
    puts "创建新项目..."
    create_project ${project_name} ./vivado_project -part ${target_part} -force
    
    # 添加 RTL 源文件
    add_files -norecurse [glob ../../rtl/*.v]
    
    # 设置顶层模块
    set_property top ${top_module} [current_fileset]
    
    # 添加约束文件
    add_files -fileset constrs_1 -norecurse ../constraints/hive_reflex_v21.xdc
    
    # 更新编译顺序
    update_compile_order -fileset sources_1
}

# 创建输出目录
file mkdir ./reports
file mkdir ./output

# ============================================================================
# 综合配置
# ============================================================================

puts "配置综合选项..."

# 综合策略: 性能优化 + 低功耗
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]

# 启用资源共享
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-resource_sharing on} -objects [get_runs synth_1]

# 启用 FSM 编码优化
set_property -name {STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION} -value {one_hot} -objects [get_runs synth_1]

# 用户综合属性
set_property KEEP_HIERARCHY soft [get_cells -hier -filter {NAME =~ *cim*}]

# ============================================================================
# 综合
# ============================================================================

puts ""
puts "=============================================="
puts "步骤 1/4: 综合"
puts "=============================================="

reset_run synth_1
launch_runs synth_1 -jobs ${num_jobs}
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: 综合失败!"
    exit 1
}

puts "✓ 综合完成"

# 打开综合结果生成报告
open_run synth_1

# 综合报告
report_utilization -file ./reports/utilization_synth.txt
report_timing_summary -file ./reports/timing_synth.txt -max_paths 20
report_clock_utilization -file ./reports/clock_synth.txt

puts ""
puts "资源利用率预览:"
report_utilization -hierarchical -hierarchical_depth 2

# ============================================================================
# 实现配置
# ============================================================================

puts ""
puts "=============================================="
puts "步骤 2/4: 实现"
puts "=============================================="

# 实现策略: 性能 + 功耗优化
set_property strategy Performance_ExploreWithRemap [get_runs impl_1]

# 布局布线优化
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraPostPlacementOpt [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE MoreGlobalIterations [get_runs impl_1]

# 物理优化
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

reset_run impl_1
launch_runs impl_1 -jobs ${num_jobs}
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: 实现失败!"
    exit 1
}

puts "✓ 实现完成"

# 打开实现结果
open_run impl_1

# 实现报告
report_utilization -file ./reports/utilization_impl.txt
report_timing_summary -file ./reports/timing_impl.txt -max_paths 50
report_clock_utilization -file ./reports/clock_impl.txt
report_methodology -file ./reports/methodology.txt

# ============================================================================
# 功耗分析
# ============================================================================

puts ""
puts "=============================================="
puts "步骤 3/4: 功耗分析"
puts "=============================================="

# 设置功耗分析参数
set_operating_conditions -process typical -voltage 1.0 -grade industrial

# 生成功耗报告
report_power -file ./reports/power_active.txt

# 估算待机功耗 (降低活动因子)
puts "估算待机功耗..."
set_switching_activity -deassert_resets -static_probability 0.01 -toggle_rate 0.001 [all_nets]
report_power -file ./reports/power_standby.txt -name power_standby

# 功耗摘要
puts ""
puts "功耗估计摘要:"
puts "  Active 模式: 查看 reports/power_active.txt"
puts "  Standby 模式: 查看 reports/power_standby.txt"

# 提取功耗数据
set power_report [open "./reports/power_active.txt" r]
set power_content [read $power_report]
close $power_report

if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $power_content match total_power]} {
    puts "  总功耗 (Active): ${total_power} W"
}

# ============================================================================
# 生成比特流
# ============================================================================

puts ""
puts "=============================================="
puts "步骤 4/4: 生成比特流"
puts "=============================================="

# 比特流配置
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 66 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.SPI_32BIT_ADDR YES [current_design]

launch_runs impl_1 -to_step write_bitstream -jobs ${num_jobs}
wait_on_run impl_1

set bitfile "./vivado_project/${project_name}.runs/impl_1/${top_module}.bit"
if {[file exists $bitfile]} {
    puts "✓ 比特流生成成功"
    
    # 复制到输出目录
    file copy -force $bitfile ./output/${top_module}.bit
    
    # 生成二进制文件 (用于 SD 卡启动)
    write_cfgmem -format bin -interface spix4 -size 128 -loadbit "up 0x00000000 $bitfile" -force ./output/${top_module}.bin
    
    puts "✓ 输出文件:"
    puts "    比特流: ./output/${top_module}.bit"
    puts "    二进制: ./output/${top_module}.bin"
} else {
    puts "ERROR: 比特流生成失败!"
    exit 1
}

# ============================================================================
# 最终报告
# ============================================================================

puts ""
puts "=============================================="
puts "构建完成!"
puts "=============================================="
puts ""
puts "版本: Hive-Reflex $version"
puts "目标: $target_part (ZCU102)"
puts ""
puts "输出文件:"
puts "  比特流: ./output/${top_module}.bit"
puts ""
puts "报告目录: ./reports/"
puts "  - utilization_synth.txt  综合资源"
puts "  - utilization_impl.txt   实现资源"
puts "  - timing_impl.txt        时序分析"
puts "  - power_active.txt       Active 功耗"
puts "  - power_standby.txt      Standby 功耗"
puts ""
puts "下一步:"
puts "  1. 检查 timing_impl.txt 确认时序收敛"
puts "  2. 比较 power_active.txt 和 power_standby.txt"
puts "  3. 下载比特流到 ZCU102 测试"
puts ""
