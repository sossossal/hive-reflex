# 使用 GitHub CLI 创建仓库并推送

# 检查是否安装 gh
if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "✓ 检测到 GitHub CLI" -ForegroundColor Green
    
    cd "d:\新建文件夹\hive-reflex"
    
    # 创建仓库
    Write-Host "创建 GitHub 仓库..." -ForegroundColor Cyan
    gh repo create hive-reflex `
        --public `
        --description "超低功耗 CIM 边缘 AI 加速器 - 稀疏计算 + DVFS + TinyML 自适应控制 + AI 反馈循环" `
        --source=. `
        --remote=origin `
        --push
    
    Write-Host ""
    Write-Host "✅ 仓库创建并推送成功!" -ForegroundColor Green
    
}
else {
    Write-Host "⚠ 未安装 GitHub CLI" -ForegroundColor Yellow
    Write-Host "请使用方案 A 手动创建仓库，或安装 GitHub CLI:" -ForegroundColor White
    Write-Host "  winget install GitHub.cli" -ForegroundColor Gray
}
