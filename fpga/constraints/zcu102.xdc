# ZCU102 时钟和引脚约束文件

# ========================================================================
# 时钟约束
# ========================================================================

# 系统时钟 100MHz
create_clock -period 10.000 -name sys_clk [get_ports sys_clk_p]

# 输入延迟
set_input_delay -clock sys_clk 2.0 [all_inputs]

# 输出延迟
set_output_delay -clock sys_clk 2.0 [all_outputs]

# 时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# ========================================================================
# 引脚分配 (ZCU102)
# ========================================================================

# 系统时钟 (差分)
set_property PACKAGE_PIN H9  [get_ports sys_clk_p]
set_property PACKAGE_PIN G9  [get_ports sys_clk_n]
set_property IOSTANDARD LVDS [get_ports sys_clk_p]
set_property IOSTANDARD LVDS [get_ports sys_clk_n]

# 复位按钮
set_property PACKAGE_PIN AM13 [get_ports sys_rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports sys_rst_n]

# LED 指示灯
set_property PACKAGE_PIN AG14 [get_ports {led[0]}]
set_property PACKAGE_PIN AF13 [get_ports {led[1]}]
set_property PACKAGE_PIN AE13 [get_ports {led[2]}]
set_property PACKAGE_PIN AJ14 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

# UART
set_property PACKAGE_PIN A20  [get_ports uart_tx]
set_property PACKAGE_PIN B20  [get_ports uart_rx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rx]

# ========================================================================
# 时序约束
# ========================================================================

# 伪路径 (跨时钟域)
set_false_path -from [get_clocks sys_clk] -to [get_ports {led[*]}]

# 多周期路径 (CIM 计算)
# set_multicycle_path -setup 2 -from [get_cells cim_*] -to [get_cells cim_*]

# ========================================================================
# 物理约束
# ========================================================================

# 放置约束 (可选, 优化布局)
# create_pblock pblock_cim
# add_cells_to_pblock [get_pblocks pblock_cim] [get_cells cim_core_inst]
# resize_pblock [get_pblocks pblock_cim] -add {SLICE_X0Y0:SLICE_X50Y50}

puts "ZCU102 约束文件加载完成"
