# IMC-22 Windows 环境一键设置脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IMC-22 Windows Setup" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

# 1. 检查 QEMU
Write-Host "`n[1/3] Checking QEMU..." -ForegroundColor Yellow
$qemu = "C:\Program Files\qemu\qemu-system-riscv32.exe"
if (Test-Path $qemu) {
    Write-Host "✓ QEMU found at: $qemu" -ForegroundColor Green
    & $qemu --version | Select-Object -First 1
} else {
    Write-Host "✗ QEMU not found" -ForegroundColor Red
    Write-Host "  Install from: https://qemu.weilnetz.de/w64/" -ForegroundColor Yellow
    exit 1
}

# 2. 检查/下载 RISC-V 工具链
Write-Host "`n[2/3] Checking RISC-V toolchain..." -ForegroundColor Yellow
$toolchainPath = "C:\Tools\riscv-gcc"
$toolchainBin = "$toolchainPath\bin\riscv-none-elf-gcc.exe"

if (Test-Path $toolchainBin) {
    Write-Host "✓ Toolchain found at: $toolchainPath" -ForegroundColor Green
    & $toolchainBin --version | Select-Object -First 1
} else {
    Write-Host "Toolchain not found. Setting up..." -ForegroundColor Yellow
    
    # 创建目录
    New-Item -ItemType Directory -Force -Path $toolchainPath | Out-Null
    
    # 提示手动下载（因为文件很大）
    Write-Host "`nPlease download RISC-V toolchain manually:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases" -ForegroundColor Cyan
    Write-Host "2. Download: xpack-riscv-none-elf-gcc-*-win32-x64.zip" -ForegroundColor Cyan
    Write-Host "3. Extract to: $toolchainPath" -ForegroundColor Cyan
    Write-Host "`nPress any key after extraction to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    
    if (!(Test-Path $toolchainBin)) {
        Write-Host "✗ Toolchain still not found. Please check extraction path." -ForegroundColor Red
        exit 1
    }
}

# 3. 配置环境
Write-Host "`n[3/3] Configuring environment..." -ForegroundColor Yellow

# 查找实际的 bin 目录（可能在子目录中）
$actualBin = Get-ChildItem -Path $toolchainPath -Recurse -Filter "riscv-none-elf-gcc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($actualBin) {
    $binPath = Split-Path $actualBin.FullName
    $env:PATH = "$binPath;$env:PATH"
    Write-Host "✓ Added to PATH: $binPath" -ForegroundColor Green
} else {
    Write-Host "✗ Could not find toolchain binaries" -ForegroundColor Red
    exit 1
}

# 4. 验证
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nRISC-V GCC:" -ForegroundColor Yellow
try {
    riscv-none-elf-gcc --version | Select-Object -First 1
    Write-Host "✓ GCC working" -ForegroundColor Green
} catch {
    Write-Host "✗ GCC not in PATH" -ForegroundColor Red
}

Write-Host "`nQEMU:" -ForegroundColor Yellow  
& $qemu --version | Select-Object -First 1
Write-Host "✓ QEMU working" -ForegroundColor Green

# 5. 创建构建文件
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Creating build files..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 创建简化的 Makefile（如果不存在）
if (!(Test-Path "Makefile.windows")) {
    Write-Host "Creating Makefile.windows..." -ForegroundColor Yellow
    # Makefile 将由另一个文件创建
}

# 完成
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nEnvironment ready for current PowerShell session." -ForegroundColor Yellow
Write-Host "`nTo make PATH permanent:" -ForegroundColor Yellow
Write-Host "  1. Open: Settings → System → About → Advanced system settings" -ForegroundColor Cyan
Write-Host "  2. Click: Environment Variables" -ForegroundColor Cyan
Write-Host "  3. Edit PATH and add: $binPath" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Build project:" -ForegroundColor Cyan
Write-Host "     make -f Makefile.windows" -ForegroundColor Cyan
Write-Host "  2. Run simulation:" -ForegroundColor Cyan
Write-Host "     make -f Makefile.windows sim" -ForegroundColor Cyan

Write-Host "`n========================================" -ForegroundColor Green
