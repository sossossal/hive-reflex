# Vivado 项目配置文件
# Hive-Reflex 2.0 FPGA 验证

# 项目设置
set project_name "hive_reflex_fpga"
set project_dir "./vivado_project"
set part "xczu9eg-ffvb1156-2-e"  # ZCU102 FPGA

# 创建项目
create_project $project_name $project_dir -part $part -force

# 设置项目属性
set_property target_language Verilog [current_project]
set_property simulator_language Verilog [current_project]

# 添加 RTL 源文件
set rtl_files [glob -nocomplain ../rtl/*.v]
if {[llength $rtl_files] > 0} {
    add_files -fileset sources_1 $rtl_files
    puts "Added [llength $rtl_files] RTL files"
}

# 添加约束文件
set constraint_files [glob -nocomplain ../constraints/*.xdc]
if {[llength $constraint_files] > 0} {
    add_files -fileset constrs_1 $constraint_files
    puts "Added [llength $constraint_files] constraint files"
}

# 添加仿真文件
set sim_files [glob -nocomplain ../sim/*.v]
if {[llength $sim_files] > 0} {
    add_files -fileset sim_1 $sim_files
    puts "Added [llength $sim_files] simulation files"
}

# 设置顶层模块
set_property top hive_reflex_top [current_fileset]

# IP 设置
set_property  ip_repo_paths  ../ip [current_project]
update_ip_catalog

puts "Vivado 项目创建完成: $project_name"
puts "FPGA 器件: $part"
