/**
 * IMC-22 RISC-V 链接脚本
 * 内存布局和段分配
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

/* 内存区域定义 */
MEMORY
{
    /* FLASH (XIP 执行) */
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 2M
    
    /* SRAM (数据和栈) */
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 512K
    
    /* CIM SRAM (神经网络权重和激活) */
    CIM   (rw)  : ORIGIN = 0x50000000, LENGTH = 512K
}

/* 栈大小配置 */
STACK_SIZE = DEFINED(__stack_size__) ? __stack_size__ : 16K;
HEAP_SIZE  = DEFINED(__heap_size__)  ? __heap_size__  : 64K;

SECTIONS
{
    /* ===================================================================== */
    /* FLASH 段 (代码和只读数据)                                             */
    /* ===================================================================== */
    
    .text : {
        . = ALIGN(4);
        _stext = .;
        
        /* 启动代码必须在最前面 */
        KEEP(*(.text.startup))
        KEEP(*(.text._start))
        
        /* 中断向量表 */
        KEEP(*(.vectors))
        
        /* 代码段 */
        *(.text)
        *(.text*)
        
        /* 只读数据 */
        *(.rodata)
        *(.rodata*)
        
        /* C++ 构造和析构 */
        KEEP(*(.init))
        KEEP(*(.fini))
        
        . = ALIGN(4);
        _etext = .;
    } > FLASH
    
    /* ARM Exception 表 (如果需要) */
    .ARM.extab : {
        *(.ARM.extab*)
    } > FLASH
    
    .ARM.exidx : {
        *(.ARM.exidx*)
    } > FLASH
    
    /* ===================================================================== */
    /* SRAM 段 (初始化数据)                                                  */
    /* ===================================================================== */
    
    .data : {
        . = ALIGN(4);
        _sdata = .;
        
        *(.data)
        *(.data*)
        
        . = ALIGN(4);
        _edata = .;
    } > SRAM AT> FLASH
    
    _sidata = LOADADDR(.data);
    
    /* ===================================================================== */
    /* SRAM 段 (未初始化数据)                                                */
    /* ===================================================================== */
    
    .bss : {
        . = ALIGN(4);
        _sbss = .;
        
        *(.bss)
        *(.bss*)
        *(COMMON)
        
        . = ALIGN(4);
        _ebss = .;
    } > SRAM
    
    /* ===================================================================== */
    /* 堆 (Heap)                                                            */
    /* ===================================================================== */
    
    .heap : {
        . = ALIGN(8);
        _heap_start = .;
        . = . + HEAP_SIZE;
        _heap_end = .;
    } > SRAM
    
    /* ===================================================================== */
    /* 栈 (Stack)                                                           */
    /* ===================================================================== */
    
    .stack : {
        . = ALIGN(8);
        . = . + STACK_SIZE;
        _stack_top = .;
    } > SRAM
    
    /* ===================================================================== */
    /* CIM SRAM (神经网络权重)                                               */
    /* ===================================================================== */
    
    .cim_weights : {
        . = ALIGN(4);
        _cim_weights_start = .;
        
        /* 预留给运行时加载的权重 */
        
        _cim_weights_end = .;
    } > CIM
    
    .cim_activations : {
        . = ALIGN(4);
        _cim_activations_start = .;
        
        /* 预留给运行时的激活值缓冲 */
        
        _cim_activations_end = .;
    } > CIM
    
    /* ===================================================================== */
    /* 调试信息 (不加载到目标)                                               */
    /* ===================================================================== */
    
    .comment 0 : { *(.comment) }
    .debug_info 0 : { *(.debug_info) }
    .debug_abbrev 0 : { *(.debug_abbrev) }
    .debug_line 0 : { *(.debug_line) }
    .debug_frame 0 : { *(.debug_frame) }
    .debug_str 0 : { *(.debug_str) }
    .debug_loc 0 : { *(.debug_loc) }
    .debug_ranges 0 : { *(.debug_ranges) }
    
    /* ===================================================================== */
    /* 丢弃的段                                                              */
    /* ===================================================================== */
    
    /DISCARD/ : {
        *(.note*)
        *(.eh_frame*)
    }
}

/* 提供符号供代码使用 */
PROVIDE(_stack_start = ORIGIN(SRAM) + LENGTH(SRAM));
PROVIDE(_ram_start = ORIGIN(SRAM));
PROVIDE(_ram_end = ORIGIN(SRAM) + LENGTH(SRAM));
PROVIDE(_flash_start = ORIGIN(FLASH));
PROVIDE(_flash_end = ORIGIN(FLASH) + LENGTH(FLASH));
