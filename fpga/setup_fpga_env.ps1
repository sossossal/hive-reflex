# Hive-Reflex 2.0 FPGA 环境搭建脚本 (Windows PowerShell)
# 需要管理员权限运行

Write-Host "╔════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Hive-Reflex 2.0 FPGA 环境搭建 (Windows)   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 检查管理员权限
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  请以管理员身份运行此脚本!" -ForegroundColor Red
    Write-Host "   右键点击 PowerShell → 以管理员身份运行" -ForegroundColor Yellow
    exit 1
}

# 工作目录
$WorkDir = "$env:USERPROFILE\fpga_dev"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Set-Location $WorkDir

Write-Host "工作目录: $WorkDir" -ForegroundColor Gray
Write-Host ""

# ========================================================================
# 步骤 1: 安装 Chocolatey (包管理器)
# ========================================================================
Write-Host "步骤 1/7: 安装 Chocolatey" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "  安装 Chocolatey..." -ForegroundColor Gray
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Host "  ✓ Chocolatey 安装完成" -ForegroundColor Green
}
else {
    Write-Host "  ✓ Chocolatey 已安装" -ForegroundColor Green
}

# ========================================================================
# 步骤 2: 安装基础工具
# ========================================================================
Write-Host ""
Write-Host "步骤 2/7: 安装基础开发工具" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

$tools = @(
    "git",
    "python3",
    "cmake",
    "make"
)

foreach ($tool in $tools) {
    Write-Host "  安装 $tool..." -ForegroundColor Gray
    choco install $tool -y --no-progress
}

Write-Host "  ✓ 基础工具安装完成" -ForegroundColor Green

# ========================================================================
# 步骤 3: 安装 RISC-V 工具链
# ========================================================================
Write-Host ""
Write-Host "步骤 3/7: 安装 RISC-V 工具链" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

$riscvPath = "C:\riscv"

if (-not (Test-Path "$riscvPath\bin\riscv32-unknown-elf-gcc.exe")) {
    Write-Host "  下载 RISC-V 工具链..." -ForegroundColor Gray
    
    $url = "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v12.2.0-3/xpack-riscv-none-elf-gcc-12.2.0-3-win32-x64.zip"
    $zipFile = "$WorkDir\riscv-toolchain.zip"
    
    Invoke-WebRequest -Uri $url -OutFile $zipFile
    
    # 解压
    Expand-Archive -Path $zipFile -DestinationPath $WorkDir -Force
    
    # 移动到 C:\riscv
    Move-Item -Path "$WorkDir\xpack-riscv-none-elf-gcc-*" -Destination $riscvPath -Force
    
    # 添加到 PATH
    $env:Path += ";$riscvPath\bin"
    [Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
    
    Remove-Item $zipFile
    
    Write-Host "  ✓ RISC-V 工具链安装完成" -ForegroundColor Green
}
else {
    Write-Host "  ✓ RISC-V 工具链已安装" -ForegroundColor Green
}

# ========================================================================
# 步骤 4: 安装 Python 验证工具
# ========================================================================
Write-Host ""
Write-Host "步骤 4/7: 安装 Python 验证工具" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

pip install --upgrade pip
pip install cocotb pytest numpy matplotlib pyserial

Write-Host "  ✓ Python 工具安装完成" -ForegroundColor Green

# ========================================================================
# 步骤 5: 安装 GTKWave (波形查看器)
# ========================================================================
Write-Host ""
Write-Host "步骤 5/7: 安装 GTKWave" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

choco install gtkwave -y --no-progress

Write-Host "  ✓ GTKWave 安装完成" -ForegroundColor Green

# ========================================================================
# 步骤 6: 下载 OpenOCD (预编译版本)
# ========================================================================
Write-Host ""
Write-Host "步骤 6/7: 安装 OpenOCD" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

$openocdPath = "C:\openocd"

if (-not (Test-Path "$openocdPath\bin\openocd.exe")) {
    Write-Host "  下载 OpenOCD..." -ForegroundColor Gray
    
    $url = "https://github.com/openocd-org/openocd/releases/download/v0.12.0/openocd-v0.12.0-i686-w64-mingw32.tar.gz"
    $tarFile = "$WorkDir\openocd.tar.gz"
    
    Invoke-WebRequest -Uri $url -OutFile $tarFile
    
    # 解压需要 7-Zip
    choco install 7zip -y --no-progress
    
    & "C:\Program Files\7-Zip\7z.exe" x $tarFile -o"$WorkDir"
    & "C:\Program Files\7-Zip\7z.exe" x "$WorkDir\openocd.tar" -o"$WorkDir"
    
    Move-Item -Path "$WorkDir\openocd-*" -Destination $openocdPath -Force
    
    # 添加到 PATH
    $env:Path += ";$openocdPath\bin"
    [Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
    
    Remove-Item $tarFile
    Remove-Item "$WorkDir\openocd.tar"
    
    Write-Host "  ✓ OpenOCD 安装完成" -ForegroundColor Green
}
else {
    Write-Host "  ✓ OpenOCD 已安装" -ForegroundColor Green
}

# ========================================================================
# 步骤 7: 克隆 Rocket Chip
# ========================================================================
Write-Host ""
Write-Host "步骤 7/7: 克隆 Rocket Chip" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Gray

if (-not (Test-Path "$WorkDir\rocket-chip")) {
    Write-Host "  克隆 Rocket Chip..." -ForegroundColor Gray
    git clone https://github.com/chipsalliance/rocket-chip.git "$WorkDir\rocket-chip"
    
    Set-Location "$WorkDir\rocket-chip"
    git submodule update --init --recursive
    Set-Location $WorkDir
    
    Write-Host "  ✓ Rocket Chip 克隆完成" -ForegroundColor Green
}
else {
    Write-Host "  ✓ Rocket Chip 已存在" -ForegroundColor Green
}

# ========================================================================
# 完成
# ========================================================================
Write-Host ""
Write-Host "╔════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  ✅ FPGA 开发环境搭建完成!                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "已安装的工具:" -ForegroundColor White
Write-Host "  ✓ RISC-V GCC" -ForegroundColor Green
Write-Host "  ✓ Python" -ForegroundColor Green
Write-Host "  ✓ GTKWave" -ForegroundColor Green
Write-Host "  ✓ OpenOCD" -ForegroundColor Green
Write-Host "  ✓ Git" -ForegroundColor Green
Write-Host ""

Write-Host "工作目录: $WorkDir" -ForegroundColor Gray
Write-Host ""

Write-Host "下一步:" -ForegroundColor White
Write-Host "  1. 安装 Xilinx Vivado (需要手动安装)" -ForegroundColor Yellow
Write-Host "     下载地址: https://www.xilinx.com/support/download.html" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. 或安装 Intel Quartus Prime Lite (免费)" -ForegroundColor Yellow
Write-Host "     下载地址: https://www.intel.com/programmable/downloads" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. 重启 PowerShell 使环境变量生效" -ForegroundColor Yellow
Write-Host ""
Write-Host "  4. 开始 FPGA 开发!" -ForegroundColor Yellow
Write-Host ""

# 创建环境检查脚本
$checkScript = @'
Write-Host "检查 FPGA 开发环境..." -ForegroundColor Cyan
Write-Host ""

$tools = @{
    "riscv32-unknown-elf-gcc" = "RISC-V GCC"
    "python" = "Python"
    "git" = "Git"
    "gtkwave" = "GTKWave"
    "openocd" = "OpenOCD"
}

$allOk = $true

foreach ($tool in $tools.Keys) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Write-Host "✓ $($tools[$tool])" -ForegroundColor Green
    } else {
        Write-Host "✗ $($tools[$tool]) (未安装)" -ForegroundColor Red
        $allOk = $false
    }
}

Write-Host ""

if ($allOk) {
    Write-Host "✅ 所有工具已就绪!" -ForegroundColor Green
} else {
    Write-Host "⚠️  部分工具缺失" -ForegroundColor Yellow
}
'@

Set-Content -Path "$WorkDir\check_env.ps1" -Value $checkScript

Write-Host "环境检查脚本已创建: $WorkDir\check_env.ps1" -ForegroundColor Gray
