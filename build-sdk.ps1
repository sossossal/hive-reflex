# IMC-22 SDK 快速编译脚本
# 编译 SDK 核心组件

param(
    [switch]$Clean
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IMC-22 SDK Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 设置工具链
$binPath = (Get-ChildItem "C:\Tools\riscv-gcc\" -Recurse -Filter "riscv-none-elf-gcc.exe" | Select-Object -First 1).DirectoryName
$env:PATH = "$binPath;$env:PATH"

# 清理
if ($Clean) {
    Write-Host "Cleaning..." -ForegroundColor Yellow
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }
    Write-Host "✓ Clean complete" -ForegroundColor Green
    exit 0
}

# 创建目录
if (!(Test-Path "build/imc22_sdk")) {
    New-Item -ItemType Directory -Force -Path "build/imc22_sdk" | Out-Null
}

# 编译选项
$cflags = @(
    "-march=rv32imac",
    "-mabi=ilp32",
    "-O2", "-g",
    "-Wall",
    "-Iimc22_sdk",
    "-ffreestanding",
    "-ffunction-sections",
    "-fdata-sections"
)

$ldflags = @(
    "-Timc22_sdk/linker.ld",
    "-nostartfiles",
    "-Wl,--gc-sections"
)

Write-Host "`nCompiling SDK..." -ForegroundColor Yellow

# 编译 SDK 源文件
$sdkSources = @(
    "imc22_sdk/startup.c",
    "imc22_sdk/imc22_can.c",
    "imc22_sdk/imc22_npu.c"
)

$objects = @()
foreach ($src in $sdkSources) {
    $obj = "build/$($src -replace '\.c$','.o')"
    $objects += $obj
    
    Write-Host "  CC $src" -ForegroundColor Gray
    & riscv-none-elf-gcc @cflags -c $src -o $obj
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed: $src" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ SDK compiled successfully" -ForegroundColor Green
Write-Host "`nObject files:" -ForegroundColor Yellow
$objects | ForEach-Object { Write-Host "  $_" -ForegroundColor Cyan }

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "SDK Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
