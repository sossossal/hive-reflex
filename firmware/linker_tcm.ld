/**
 * @file linker_tcm.ld
 * @brief 链接脚本 - 支持 TCM 内存映射
 * 
 * 内存布局:
 *   TCM   (0x0000_0000 - 0x0000_7FFF): 32KB 紧耦合内存
 *   Flash (0x0800_0000 - 0x08FF_FFFF): 16MB XIP 代码/数据
 *   SRAM  (0x2000_0000 - 0x2007_FFFF): 512KB 共享 SRAM
 */

OUTPUT_FORMAT("elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    TCM   (rwx) : ORIGIN = 0x00000000, LENGTH = 32K
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 16M
    SRAM  (rw)  : ORIGIN = 0x20000000, LENGTH = 512K
}

SECTIONS
{
    /* ===================================================================
     * TCM 区域 - 关键代码和数据
     * =================================================================== */
    
    /* 中断向量表 (必须在 0x0000_0000) */
    .vectors : ALIGN(4)
    {
        KEEP(*(.vectors))
        KEEP(*(.vectors.*))
    } > TCM
    
    /* 热点代码 (放 TCM 加速) */
    .tcm_text : ALIGN(4)
    {
        *fixed_point_filter.o(.text .text.*)
        *can_sync.o(.text .text.*)
        *control_loop.o(.text .text.*)
        *imu_driver.o(.text .text.*)
        
        /* 中断处理函数 */
        *(.tcm_text)
        *(.tcm_text.*)
    } > TCM
    
    /* TCM 数据 (关键全局变量) */
    .tcm_data : ALIGN(4)
    {
        *(.tcm_data)
        *(.tcm_data.*)
    } > TCM
    
    /* TCM BSS */
    .tcm_bss (NOLOAD) : ALIGN(4)
    {
        *(.tcm_bss)
        *(.tcm_bss.*)
    } > TCM
    
    /* ===================================================================
     * Flash 区域 - XIP 代码和只读数据
     * =================================================================== */
    
    /* 普通代码 */
    .text : ALIGN(4)
    {
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
        
        /* C++ 支持 */
        *(.gnu.linkonce.t.*)
        *(.gnu.linkonce.r.*)
        
        /* 初始化函数 */
        KEEP(*(.init))
        KEEP(*(.fini))
    } > FLASH
    
    /* 构造/析构表 */
    .preinit_array : ALIGN(4)
    {
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
    } > FLASH
    
    .init_array : ALIGN(4)
    {
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN (__init_array_end = .);
    } > FLASH
    
    .fini_array : ALIGN(4)
    {
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN (__fini_array_end = .);
    } > FLASH
    
    /* ===================================================================
     * SRAM 区域 - 可读写数据
     * =================================================================== */
    
    /* 已初始化数据 (需从 Flash 拷贝) */
    .data : ALIGN(4)
    {
        __data_start = .;
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
        *(.gnu.linkonce.d.*)
        __data_end = .;
    } > SRAM AT > FLASH
    
    __data_load = LOADADDR(.data);
    
    /* 未初始化数据 */
    .bss (NOLOAD) : ALIGN(4)
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
        __bss_end = .;
    } > SRAM
    
    /* 堆 */
    .heap (NOLOAD) : ALIGN(4)
    {
        __heap_start = .;
        . = . + 64K;  /* 64KB 堆 */
        __heap_end = .;
    } > SRAM
    
    /* 栈 (从 SRAM 顶部向下生长) */
    .stack (NOLOAD) : ALIGN(16)
    {
        . = . + 32K;  /* 32KB 栈 */
        __stack_top = .;
    } > SRAM
    
    /* ===================================================================
     * 调试信息
     * =================================================================== */
    
    .comment 0 : { *(.comment) }
    .debug_info 0 : { *(.debug_info) }
    .debug_abbrev 0 : { *(.debug_abbrev) }
    .debug_line 0 : { *(.debug_line) }
    .debug_frame 0 : { *(.debug_frame) }
    .debug_str 0 : { *(.debug_str) }
    .debug_loc 0 : { *(.debug_loc) }
    .debug_ranges 0 : { *(.debug_ranges) }
    
    /* 丢弃不需要的段 */
    /DISCARD/ : {
        *(.note.GNU-stack)
        *(.gnu_debuglink)
        *(.gnu.lto_*)
    }
}

/* 全局符号 */
PROVIDE(_stack_top = __stack_top);
PROVIDE(_heap_start = __heap_start);
PROVIDE(_heap_end = __heap_end);
