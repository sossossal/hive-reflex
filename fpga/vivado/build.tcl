# Vivado 综合和实现脚本

# 打开项目
open_project ./vivado_project/hive_reflex_fpga.xpr

# 复位运行
reset_run synth_1
reset_run impl_1

puts "=========================================="
puts "开始综合..."
puts "=========================================="

# 综合
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: 综合失败!"
    exit 1
}

puts "✓ 综合完成"

# 综合报告
open_run synth_1
report_utilization -file ./reports/utilization_synth.txt
report_timing_summary -file ./reports/timing_synth.txt

puts "=========================================="
puts "开始实现..."
puts "=========================================="

# 实现
launch_runs impl_1 -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: 实现失败!"
    exit 1
}

puts "✓ 实现完成"

# 实现报告
open_run impl_1
report_utilization -file ./reports/utilization_impl.txt
report_timing_summary -file ./reports/timing_impl.txt
report_power -file ./reports/power.txt

puts "=========================================="
puts "生成比特流..."
puts "=========================================="

# 生成比特流
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

if {[file exists ./vivado_project/hive_reflex_fpga.runs/impl_1/hive_reflex_top.bit]} {
    puts "✓ 比特流生成成功"
    
    # 复制到输出目录
    file mkdir ./output
    file copy -force ./vivado_project/hive_reflex_fpga.runs/impl_1/hive_reflex_top.bit ./output/
    file copy -force ./vivado_project/hive_reflex_fpga.runs/impl_1/hive_reflex_top.bin ./output/
    
    puts "✓ 比特流已复制到 ./output/"
} else {
    puts "ERROR: 比特流生成失败!"
    exit 1
}

puts ""
puts "=========================================="
puts "构建完成!"
puts "=========================================="
puts "比特流: ./output/hive_reflex_top.bit"
puts "报告目录: ./reports/"
puts ""
