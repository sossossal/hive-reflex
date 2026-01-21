# Hive-Reflex 2.1 综合约束文件
# 针对 DVFS、稀疏计算和低功耗设计优化
#
# 目标平台: Xilinx ZCU102 / Zynq UltraScale+

# ============================================================================
# 时钟约束
# ============================================================================

# 系统时钟 100MHz
create_clock -period 10.000 -name sys_clk [get_ports sys_clk_p]

# DVFS 分频时钟 (动态生成)
create_generated_clock -name clk_div2  -source [get_ports sys_clk_p] -divide_by 2  [get_pins dvfs_*/clk_div2_reg/Q]
create_generated_clock -name clk_div10 -source [get_ports sys_clk_p] -divide_by 10 [get_pins dvfs_*/clk_div10_reg/Q]
create_generated_clock -name clk_div100 -source [get_ports sys_clk_p] -divide_by 100 [get_pins dvfs_*/clk_div100_reg/Q]

# 时钟组 (互斥时钟)
set_clock_groups -logically_exclusive \
    -group [get_clocks sys_clk] \
    -group [get_clocks clk_div2] \
    -group [get_clocks clk_div10] \
    -group [get_clocks clk_div100]

# 输入/输出延迟
set_input_delay  -clock sys_clk 2.0 [all_inputs]
set_output_delay -clock sys_clk 2.0 [all_outputs]

# 时钟不确定性
set_clock_uncertainty 0.3 [all_clocks]

# ============================================================================
# 功耗优化约束
# ============================================================================

# 电压域约束 (信息性)
# Active:    1.0V core, 1.8V I/O
# Standby:   0.6V core, 1.8V I/O  
# DeepSleep: 0.4V core, 1.2V I/O

# 时钟门控属性
set_property CLOCK_BUFFER_TYPE BUFGCE [get_nets dvfs_*/gated_clk]
set_property GATED_CLOCK TRUE [get_nets dvfs_*/gated_clk]

# 使能低功耗模式
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 66 [current_design]

# ============================================================================
# 引脚分配 (ZCU102)
# ============================================================================

# 系统时钟 (差分)
set_property PACKAGE_PIN H9   [get_ports sys_clk_p]
set_property PACKAGE_PIN G9   [get_ports sys_clk_n]
set_property IOSTANDARD LVDS  [get_ports sys_clk_p]
set_property IOSTANDARD LVDS  [get_ports sys_clk_n]

# 复位按钮
set_property PACKAGE_PIN AM13 [get_ports sys_rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports sys_rst_n]

# LED 指示灯
set_property PACKAGE_PIN AG14 [get_ports {led[0]}]
set_property PACKAGE_PIN AF13 [get_ports {led[1]}]
set_property PACKAGE_PIN AE13 [get_ports {led[2]}]
set_property PACKAGE_PIN AJ14 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]
set_property SLEW SLOW [get_ports {led[*]}]
set_property DRIVE 4 [get_ports {led[*]}]

# UART
set_property PACKAGE_PIN A20 [get_ports uart_tx]
set_property PACKAGE_PIN B20 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rx]

# JTAG (调试)
set_property PACKAGE_PIN R20 [get_ports tck]
set_property PACKAGE_PIN P20 [get_ports tms]
set_property PACKAGE_PIN N20 [get_ports tdi]
set_property PACKAGE_PIN M20 [get_ports tdo]
set_property IOSTANDARD LVCMOS18 [get_ports {tck tms tdi tdo}]

# PMU 电压控制接口 (假设外部 PMU)
set_property PACKAGE_PIN H20 [get_ports pmu_voltage_request]
set_property PACKAGE_PIN J20 [get_ports pmu_voltage_stable]
set_property PACKAGE_PIN K20 [get_ports {pmu_voltage_target[0]}]
set_property PACKAGE_PIN K21 [get_ports {pmu_voltage_target[1]}]
set_property PACKAGE_PIN L20 [get_ports {pmu_voltage_target[2]}]
set_property PACKAGE_PIN L21 [get_ports {pmu_voltage_target[3]}]
set_property PACKAGE_PIN M21 [get_ports {pmu_voltage_target[4]}]
set_property PACKAGE_PIN N21 [get_ports {pmu_voltage_target[5]}]
set_property PACKAGE_PIN P21 [get_ports {pmu_voltage_target[6]}]
set_property PACKAGE_PIN R21 [get_ports {pmu_voltage_target[7]}]
set_property IOSTANDARD LVCMOS18 [get_ports pmu_*]

# ============================================================================
# 时序约束
# ============================================================================

# 伪路径
set_false_path -from [get_clocks sys_clk] -to [get_ports {led[*]}]
set_false_path -from [get_ports sys_rst_n] -to [all_registers]

# DVFS 状态转换 (多周期路径)
set_multicycle_path -setup 3 -from [get_cells dvfs_*/state_reg*] -to [get_cells dvfs_*/voltage_level_reg*]
set_multicycle_path -hold 2 -from [get_cells dvfs_*/state_reg*] -to [get_cells dvfs_*/voltage_level_reg*]

# 稀疏 MAC 累加树 (多周期路径 - 3 级流水线)
set_multicycle_path -setup 3 -from [get_cells sparse_cim_*/sum_stage1*] -to [get_cells sparse_cim_*/sum_stage3*]
set_multicycle_path -hold 2 -from [get_cells sparse_cim_*/sum_stage1*] -to [get_cells sparse_cim_*/sum_stage3*]

# CIM 控制器 (允许较慢路径)
set_multicycle_path -setup 2 -from [get_cells cim_*/ctrl_reg*] -to [get_cells cim_*/mac_*]
set_multicycle_path -hold 1 -from [get_cells cim_*/ctrl_reg*] -to [get_cells cim_*/mac_*]

# ============================================================================
# 物理约束
# ============================================================================

# CIM 核心放置区域
create_pblock pblock_cim
add_cells_to_pblock [get_pblocks pblock_cim] [get_cells -hier -filter {NAME =~ *cim_core* || NAME =~ *sparse_cim*}]
resize_pblock [get_pblocks pblock_cim] -add {SLICE_X0Y0:SLICE_X80Y120}

# DVFS 控制器放置
create_pblock pblock_dvfs
add_cells_to_pblock [get_pblocks pblock_dvfs] [get_cells -hier -filter {NAME =~ *dvfs_*}]
resize_pblock [get_pblocks pblock_dvfs] -add {SLICE_X0Y120:SLICE_X40Y160}

# BRAM 放置 (权重存储)
create_pblock pblock_bram
add_cells_to_pblock [get_pblocks pblock_bram] [get_cells -hier -filter {NAME =~ *sram* || NAME =~ *weight*}]
resize_pblock [get_pblocks pblock_bram] -add {RAMB36_X0Y0:RAMB36_X5Y20}

# ============================================================================
# 功耗分析约束
# ============================================================================

# 设置典型运行模式活动因子
set_switching_activity -static_probability 0.5 -toggle_rate 10 [get_nets -hier -filter {NAME =~ *cim*}]
set_switching_activity -static_probability 0.2 -toggle_rate 2 [get_nets -hier -filter {NAME =~ *dvfs*}]
set_switching_activity -static_probability 0.1 -toggle_rate 0.5 [get_nets -hier -filter {NAME =~ *sparse* && NAME =~ *skip*}]

# 待机模式活动因子 (用于功耗估计)
# set_switching_activity -static_probability 0.01 -toggle_rate 0.001 [all_nets]

puts "Hive-Reflex 2.1 综合约束加载完成"
puts "  - DVFS 多时钟域支持"
puts "  - 稀疏计算多周期路径"
puts "  - 功耗优化约束"
