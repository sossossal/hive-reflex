# IMC-22 Windows 构建脚本
# 无需 Make，直接使用 PowerShell

param(
    [string]$Action = "build"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IMC-22 Windows Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 设置路径
$binPath = (Get-ChildItem "C:\Tools\riscv-gcc\" -Recurse -Filter "riscv-none-elf-gcc.exe" | Select-Object -First 1).DirectoryName
$env:PATH = "$binPath;$env:PATH"
$qemu = "C:\Program Files\qemu\qemu-system-riscv32.exe"

# 编译选项
$cflags = "-march=rv32imac", "-mabi=ilp32", "-O2", "-g", "-Wall", 
"-Iimc22_sdk", "-ffreestanding", "-ffunction-sections", "-fdata-sections"
$ldflags = "-Timc22_sdk/linker.ld", "-nostartfiles", "-Wl,--gc-sections"

# 源文件
$sources = @(
    "imc22_sdk/startup.c",
    "imc22_sdk/imc22_can.c", 
    "imc22_sdk/imc22_npu.c",
    "hive_node_ctrl.c"
)

# 创建build目录
if (!(Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
    New-Item -ItemType Directory -Path "build/imc22_sdk" | Out-Null
}

if ($Action -eq "clean") {
    Write-Host "`nCleaning..." -ForegroundColor Yellow
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }
    Write-Host "✓ Clean complete" -ForegroundColor Green
    exit 0
}

# 编译
Write-Host "`n[1/3] Compiling sources..." -ForegroundColor Yellow
$objects = @()

foreach ($src in $sources) {
    $obj = "build/$($src -replace '\.c$','.o')"
    $objects += $obj
    
    Write-Host "  CC $src" -ForegroundColor Gray
    & riscv-none-elf-gcc @cflags -c $src -o $obj
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Compilation failed: $src" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Compilation successful" -ForegroundColor Green

# 链接
Write-Host "`n[2/3] Linking..." -ForegroundColor Yellow
& riscv-none-elf-gcc @cflags @ldflags $objects -o build/hive_node.elf

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Linking failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Linking successful" -ForegroundColor Green

# 生成二进制
Write-Host "`n[3/3] Creating binary..." -ForegroundColor Yellow
& riscv-none-elf-objcopy -O binary build/hive_node.elf build/hive_node.bin

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Binary creation failed" -ForegroundColor Red
    exit 1
}

# 显示大小
& riscv-none-elf-size build/hive_node.elf

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nOutput files:" -ForegroundColor Yellow
Write-Host "  build/hive_node.elf" -ForegroundColor Cyan
Write-Host "  build/hive_node.bin" -ForegroundColor Cyan

# 如果是sim模式，运行仿真
if ($Action -eq "sim") {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Starting QEMU Simulation" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nPress Ctrl+A then X to exit QEMU`n" -ForegroundColor Yellow
    
    & $qemu -nographic -machine virt -kernel build/hive_node.elf
}

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  .\build.ps1 sim   - Run simulation" -ForegroundColor Cyan
Write-Host "  .\build.ps1 clean - Clean build" -ForegroundColor Cyan
