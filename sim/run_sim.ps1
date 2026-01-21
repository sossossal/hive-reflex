# MAC 单元 Verilator 仿真脚本 (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MAC Unit Verilator 仿真 (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 创建构建目录
$BuildDir = "build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# 文件路径
$RtlFile = "..\rtl\mac_unit.v"
$TbFile = "mac_unit_tb.v"

Write-Host "步骤 1/3: 编译 RTL" -ForegroundColor Green
Write-Host "------------------------------------------" -ForegroundColor Gray

# 检查 Verilator
if (-not (Get-Command verilator -ErrorAction SilentlyContinue)) {
    Write-Host "错误: Verilator 未安装" -ForegroundColor Red
    Write-Host "请先运行: .\setup_fpga_env.ps1" -ForegroundColor Yellow
    exit 1
}

# 编译 (使用 iverilog 作为备选)
if (Get-Command iverilog -ErrorAction SilentlyContinue) {
    Write-Host "  使用 Icarus Verilog 编译..." -ForegroundColor Gray
    
    iverilog -o "$BuildDir\mac_unit_sim.vvp" $RtlFile $TbFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ 编译成功" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ 编译失败" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "步骤 2/3: 运行仿真" -ForegroundColor Green
    Write-Host "------------------------------------------" -ForegroundColor Gray
    
    vvp "$BuildDir\mac_unit_sim.vvp"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ 仿真完成" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ 仿真失败" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "  警告: 未找到仿真器" -ForegroundColor Yellow
    Write-Host "  请安装 Icarus Verilog 或 Verilator" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "步骤 3/3: 查看波形" -ForegroundColor Green
Write-Host "------------------------------------------" -ForegroundColor Gray

if (Test-Path "mac_unit_tb.vcd") {
    Write-Host "  ✓ 波形文件生成: mac_unit_tb.vcd" -ForegroundColor Green
    Write-Host ""
    Write-Host "使用 GTKWave 查看:" -ForegroundColor White
    Write-Host "  gtkwave mac_unit_tb.vcd" -ForegroundColor Gray
    
    # 自动打开 GTKWave (如果可用)
    if (Get-Command gtkwave -ErrorAction SilentlyContinue) {
        Write-Host ""
        Write-Host "是否打开 GTKWave? (y/n)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq 'y') {
            Start-Process gtkwave "mac_unit_tb.vcd"
        }
    }
}
else {
    Write-Host "  ⚠ 未生成波形文件" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "仿真完成!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
