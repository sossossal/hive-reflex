// Hive-Reflex 2.0 FPGA 验证 - Week 1 启动指南

## 🎯 Week 1 目标：CIM 核心实现

本周任务：
1. ✅ 环境搭建 (已完成脚本)
2. ⏳ MAC 单元实现
3. ⏳ MAC 阵列实现
4. ⏳ CIM SRAM 实现
5. ⏳ 单元测试

## 📋 当前进度

### 步骤 1: 运行环境安装脚本

**Windows 用户 (推荐):**
```powershell
# 以管理员身份运行 PowerShell

cd d:\新建文件夹\hive-reflex\fpga

# 运行安装脚本
.\setup_fpga_env.ps1

# 预计时间: 30-60 分钟
# 需要网络连接下载组件
```

**注意事项:**
- 需要管理员权限
- 需要良好的网络环境
- 总下载大小约 2-3 GB
- 自动安装：RISC-V 工具链、Python 工具、GTKWave、OpenOCD

### 步骤 2: 手动安装 Vivado (可选但推荐)

由于 Vivado 需要手动安装：

1. **下载 Vivado**
   - 访问: https://www.xilinx.com/support/download.html
   - 选择: Vivado ML Standard 2023.2
   - 大小: ~40 GB
   - 下载时间: 根据网速

2. **安装 Vivado**
   - 运行安装程序
   - 选择"Vivado ML Standard"
   - 安装路径: C:\Xilinx
   - 时间: 1-2 小时

3. **获取许可证**
   - 注册 Xilinx 账号
   - 申请免费的 Webpack 许可证
   - 安装许可证文件

### 步骤 3: 验证环境

```powershell
# 运行检查脚本
.\check_env.ps1

# 预期输出:
# ✓ RISC-V GCC
# ✓ Python
# ✓ Git
# ✓ GTKWave
# ✓ OpenOCD
# ✅ 所有工具已就绪!
```

---

## 🚀 快速开始 (如果已有环境)

如果您已经有开发环境，可以直接开始 RTL 开发：

### 创建 Vivado 项目

```bash
cd d:\新建文件夹\hive-reflex\fpga\vivado

# 创建项目
vivado -mode batch -source create_project.tcl
```

### 查看已有的 RTL 文件

```bash
cd d:\新建文件夹\hive-reflex\rtl
dir *.v

# 应该看到:
# cim_mac_array.v - MAC 阵列基础实现
```

---

## 📊 里程碑 M1 标准

Week 1 结束时需要达到：

- [x] 开发环境搭建完成
- [ ] MAC 单元 RTL 实现并验证
- [ ] MAC 阵列 (256 MAC) 实现
- [ ] CIM SRAM (512KB) 实现
- [ ] 通过基础仿真测试
- [ ] 资源估算合理

**成功标准:**
- 代码可综合
- 仿真功能正确
- 满足 100MHz 时序

---

## 🔧 当前状态

```
✅ 环境搭建脚本已创建
✅ RTL 示例代码已创建
⏳ 等待运行安装脚本
⏳ 等待开始 RTL 开发
```

---

## 📝 下一步行动

### 立即执行:

**1. 运行环境安装 (如果还没有)**
```powershell
.\setup_fpga_env.ps1
```

**2. 检查环境**
```powershell
.\check_env.ps1
```

**3. 开始 RTL 开发**
- 查看 `cim_mac_array.v`
- 完善 MAC 单元实现
- 创建测试平台

---

**提示**: 如果网络条件不好，可以：
1. 先下载 RISC-V 工具链到本地
2. 使用国内镜像源
3. 分步安装各个组件

**预计完成时间**: 1-2 天 (含下载和安装)

---

准备好了吗？让我们开始 FPGA 验证！🚀
