# Hive-Reflex GitHub 推送脚本
# 使用方法: 替换 YOUR_USERNAME 后运行

# 1. 设置你的 GitHub 用户名
$GITHUB_USERNAME = "YOUR_USERNAME"  # 替换为你的 GitHub 用户名

# 2. 进入项目目录
cd "d:\新建文件夹\hive-reflex"

# 3. 添加远程仓库
Write-Host "添加远程仓库..." -ForegroundColor Green
git remote add origin "https://github.com/$GITHUB_USERNAME/hive-reflex.git"

# 4. 重命名分支为 main
Write-Host "重命名分支为 main..." -ForegroundColor Green
git branch -M main

# 5. 推送到 GitHub
Write-Host "推送代码到 GitHub..." -ForegroundColor Green
git push -u origin main

Write-Host ""
Write-Host "✅ 代码已成功推送到 GitHub!" -ForegroundColor Green
Write-Host "仓库地址: https://github.com/$GITHUB_USERNAME/hive-reflex" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "1. 访问仓库页面" -ForegroundColor White
Write-Host "2. 添加 Topics (标签)" -ForegroundColor White
Write-Host "3. 创建 Release v2.1.0" -ForegroundColor White
