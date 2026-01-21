# Hive-Reflex 2.0 完整构建脚本 (Windows PowerShell)
# 从神经网络训练到固件烧录的一键完成

param(
    [string]$ModelPath = "reflex_net_v2.onnx",
    [string]$Target = "imc22",
    [switch]$Flash = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

Write-Host "╔════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Hive-Reflex 2.0 完整构建脚本              ║" -ForegroundColor Cyan
Write-Host "║  神经网络 → 固件 → 烧录                   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 配置
$BuildDir = "build"
$OutputFirmware = "$BuildDir/hive_reflex.bin"
$ModelFlash = "$BuildDir/model.flash"

# 创建构建目录
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# ===================================================================
# 步骤 1: 导出神经网络模型
# ===================================================================
Write-Host "步骤 1/7: 导出神经网络模型" -ForegroundColor Green
Write-Host "  模型: $ModelPath" -ForegroundColor Gray

if (-not (Test-Path $ModelPath)) {
    Write-Host "  模型不存在, 从 PyTorch 导出..." -ForegroundColor Yellow
    python reflex_net_v2.py --quantize
    $ModelPath = "reflex_net_v2.onnx"
}

Write-Host "  ✓ 模型准备完成" -ForegroundColor Green
Write-Host ""

# ===================================================================
# 步骤 2: MLIR 编译器编译
# ===================================================================
Write-Host "步骤 2/7: MLIR 编译器编译模型" -ForegroundColor Green

python mlir_compiler/compile.py `
    --model $ModelPath `
    --output-c "$BuildDir/reflex_inference.c" `
    --output-weights "$BuildDir/reflex_weights.bin" `
    --opt 2

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ MLIR 编译失败" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ 编译完成" -ForegroundColor Green
Write-Host ""

# ===================================================================
# 步骤 3: 打包模型为 FLASH 格式
# ===================================================================
Write-Host "步骤 3/7: 打包模型为 FLASH 格式" -ForegroundColor Green

python tools/flash_model.py `
    --weights "$BuildDir/reflex_weights.bin" `
    --output $ModelFlash `
    --name "ReflexNetV2" `
    --input-size 12 `
    --output-size 1 `
    --hidden-size 16 `
    --has-lstm `
    --gen-script

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ 模型打包失败" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ 打包完成: $ModelFlash" -ForegroundColor Green
Write-Host ""

# ===================================================================
# 步骤 4: 编译 SDK
# ===================================================================
Write-Host "步骤 4/7: 编译 IMC-22 SDK" -ForegroundColor Green

$SdkSources = @(
    "imc22_sdk/imc22_can.c",
    "imc22_sdk/imc22_npu.c",
    "imc22_sdk/imc22_power.c",
    "imc22_sdk/imc22_cim.c",
    "imc22_sdk/model_loader.c",
    "imc22_sdk/startup.c"
)

Write-Host "  编译 SDK 源文件..." -ForegroundColor Gray
foreach ($src in $SdkSources) {
    Write-Host "    $src" -ForegroundColor DarkGray
}

# 使用 Make 编译
$env:APP_SRCS = "examples/example_reflex_inference.c"
& make clean | Out-Null
& make

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ SDK 编译失败" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ SDK 编译完成" -ForegroundColor Green
Write-Host ""

# ===================================================================
# 步骤 5: 链接固件
# ===================================================================
Write-Host "步骤 5/7: 链接最终固件" -ForegroundColor Green

if (Test-Path $OutputFirmware) {
    $size = (Get-Item $OutputFirmware).Length
    Write-Host "  固件大小: $size bytes" -ForegroundColor Gray
    Write-Host "  ✓ 固件链接完成: $OutputFirmware" -ForegroundColor Green
} else {
    Write-Host "  ✗ 固件生成失败" -ForegroundColor Red
    exit 1
}

Write-Host ""

# ===================================================================
# 步骤 6: 生成烧录映像
# ===================================================================
Write-Host "步骤 6/7: 生成完整烧录映像" -ForegroundColor Green

# 合并固件和模型
$CombinedImage = "$BuildDir/firmware_complete.bin"

# 读取固件
$firmwareBytes = [System.IO.File]::ReadAllBytes($OutputFirmware)

# 填充到模型分区起始地址 (0x08090000 - 0x08000000 = 0x90000 = 589824)
$paddingSize = 589824 - $firmwareBytes.Length
if ($paddingSize -gt 0) {
    $padding = New-Object byte[] $paddingSize
    $firmwareBytes += $padding
}

# 读取模型
$modelBytes = [System.IO.File]::ReadAllBytes($ModelFlash)

# 合并
$combined = $firmwareBytes + $modelBytes

# 写入
[System.IO.File]::WriteAllBytes($CombinedImage, $combined)

Write-Host "  ✓ 完整映像: $CombinedImage" -ForegroundColor Green
Write-Host "  总大小: $($combined.Length) bytes" -ForegroundColor Gray
Write-Host ""

# ===================================================================
# 步骤 7: 烧录 (可选)
# ===================================================================
if ($Flash) {
    Write-Host "步骤 7/7: 烧录到芯片" -ForegroundColor Green
    
    # 使用 OpenOCD
    $ocdScript = "$BuildDir/model.ocd"
    
    if (Test-Path $ocdScript) {
        Write-Host "  使用 OpenOCD 烧录..." -ForegroundColor Gray
        & openocd -f interface/jlink.cfg -f target/riscv.cfg -f $ocdScript
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ 烧录完成" -ForegroundColor Green
        } else {
            Write-Host "  ✗ 烧录失败" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⚠ OpenOCD 脚本不存在, 跳过烧录" -ForegroundColor Yellow
    }
} else {
    Write-Host "步骤 7/7: 烧录 (跳过, 使用 -Flash 启用)" -ForegroundColor Yellow
}

Write-Host ""

# ===================================================================
# 总结
# ===================================================================
Write-Host "╔════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  ✅ 构建完成!                              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "生成的文件:" -ForegroundColor White
Write-Host "  📦 固件:   $OutputFirmware" -ForegroundColor Gray
Write-Host "  📦 模型:   $ModelFlash" -ForegroundColor Gray
Write-Host "  📦 完整:   $CombinedImage" -ForegroundColor Gray
Write-Host ""
Write-Host "下一步:" -ForegroundColor White
Write-Host "  1. 连接 J-Link/ST-Link 调试器" -ForegroundColor Gray
Write-Host "  2. 运行: .\build-complete.ps1 -Flash" -ForegroundColor Gray
Write-Host "  3. 或手动烧录: openocd -f $BuildDir/model.ocd" -ForegroundColor Gray
Write-Host ""
