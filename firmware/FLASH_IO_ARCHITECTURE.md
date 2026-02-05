```mermaid
graph TB
    subgraph "Application Layer"
        APP["应用程序<br/>(flash_io_demo.c)"]
    end
    
    subgraph "Integration Framework"
        OPT["Flash IO Optimizer<br/>统一推理接口"]
        OPT_CFG["配置管理<br/>• OPT_PIPELINE<br/>• OPT_COMPRESS<br/>• OPT_CASCADE<br/>• OPT_ALL"]
    end
    
    subgraph "Strategy 1: Pipeline"
        PIPE["Pipeline Controller"]
        PIPE_DMA["DMA 管理器"]
        PIPE_BANK["Bank 映射<br/>• Bank 0 (Ping)<br/>• Bank 1 (Pong)"]
    end
    
    subgraph "Strategy 2: Compression"
        COMP["Compression Library"]
        COMP_RLE["RLE<br/>(1.5x)"]
        COMP_LZ4["LZ4<br/>(2-3x)"]
        COMP_DELTA["Delta<br/>(2-4x)"]
        COMP_HUFF["Huffman<br/>(3-5x)"]
    end
    
    subgraph "Strategy 3: Cascade"
        CAS["Cascade Model"]
        CAS_EXIT1["Exit Point 1<br/>Layer 2<br/>Threshold: 0.85"]
        CAS_EXIT2["Exit Point 2<br/>Layer 5<br/>Threshold: 0.90"]
        CAS_STATS["统计跟踪<br/>• 早退出率<br/>• Flash 节省量"]
    end
    
    subgraph "Hardware Abstraction"
        HAL_FLASH["Flash Driver<br/>100 MB/s"]
        HAL_CIM["CIM Engine<br/>10 TOPS"]
        HAL_SRAM["SRAM Banks<br/>512 KB"]
        HAL_DMA["DMA Controller"]
    end
    
    APP --> OPT
    OPT --> OPT_CFG
    
    OPT_CFG -.->|启用| PIPE
    OPT_CFG -.->|启用| COMP
    OPT_CFG -.->|启用| CAS
    
    PIPE --> PIPE_DMA
    PIPE --> PIPE_BANK
    PIPE_DMA --> HAL_DMA
    PIPE_BANK --> HAL_SRAM
    
    COMP --> COMP_RLE
    COMP --> COMP_LZ4
    COMP --> COMP_DELTA
    COMP --> COMP_HUFF
    
    CAS --> CAS_EXIT1
    CAS --> CAS_EXIT2
    CAS --> CAS_STATS
    
    PIPE --> HAL_FLASH
    COMP --> HAL_FLASH
    OPT --> HAL_CIM
    
    classDef appStyle fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef strategyStyle fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef halStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef optStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    
    class APP appStyle
    class PIPE,COMP,CAS strategyStyle
    class HAL_FLASH,HAL_CIM,HAL_SRAM,HAL_DMA halStyle
    class OPT,OPT_CFG optStyle
```

# Flash IO 优化架构图

## 层次说明

### 1. Application Layer（应用层）
- **flash_io_demo.c**: 完整的演示程序，展示五种优化场景

### 2. Integration Framework（集成框架）
- **Flash IO Optimizer**: 统一的推理接口，自动协调三种策略
- **配置管理**: 灵活的策略组合控制

### 3. Strategy 1: Pipeline（软件流水线）
- **Pipeline Controller**: 流水线控制器
- **DMA 管理器**: 异步数据传输
- **Bank 映射**: 乒乓缓冲实现

### 4. Strategy 2: Compression（实时解压缩）
- **Compression Library**: 压缩库核心
- **多算法支持**: RLE, LZ4, Delta, Huffman
- **自动检测**: 根据压缩头自动选择算法

### 5. Strategy 3: Cascade（条件加载）
- **Cascade Model**: 级联模型管理
- **早退出点**: 在关键层检查置信度
- **统计跟踪**: 实时追踪性能指标

### 6. Hardware Abstraction（硬件抽象）
- **Flash Driver**: Flash 存储接口
- **CIM Engine**: 计算核心
- **SRAM Banks**: 片上存储
- **DMA Controller**: 直接内存访问

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                         推理流程                              │
└─────────────────────────────────────────────────────────────┘

1. 初始化阶段
   APP → OPT.Init(flags) → 配置各策略控制器

2. Layer 推理（Pipeline + Compression 启用）
   ┌───────────────────────────────────────────────┐
   │ Layer N:                                      │
   │  ① DMA 从 Flash 读取压缩权重 → SRAM Bank 0    │
   │  ② CPU 解压 → SRAM Bank 0                     │
   │  ③ CIM 计算（使用 Bank 0 权重）               │
   │                                               │
   │ 同时:                                          │
   │  ④ DMA 异步加载 Layer N+1 → SRAM Bank 1      │
   └───────────────────────────────────────────────┘

3. 早退出检查（Cascade 启用）
   每层计算后:
   ┌─────────────────────────────────────────────┐
   │ if (Cascade_ShouldExit(layer, output)):    │
   │     ✓ 计算节省的 Flash 读取量                │
   │     ✓ 更新统计                               │
   │     ✓ return 结果（跳过后续层）              │
   └─────────────────────────────────────────────┘

4. 性能统计
   OPT.PrintStats() → 聚合所有策略的性能指标
```

## 性能优化关键点

### 并行化
- **Flash 读取** 与 **CIM 计算** 并行（Pipeline）
- **DMA 传输** 在后台异步执行

### 数据压缩
- 离线压缩模型权重（2-5x 压缩比）
- 在线解压缩（CPU 开销 < 10%）
- **有效带宽提升**: 100 MB/s → 200-500 MB/s

### 条件执行
- 70% 场景下早退出（仅执行 25% 层）
- **Flash IO 减少**: 360 KB → 108 KB (-70%)

### 组合效应
- Pipeline 隐藏 IO 延迟
- Compression 提升带宽
- Cascade 减少总 IO 量
- **最终加速**: 1.86x，Flash 节省 60%
