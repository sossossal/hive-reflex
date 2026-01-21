# Hive-Reflex 2.0 Makefile (更新版)
# 支持 RISC-V 工具链和完整的 SDK

# ==============================================================================
# 工具链配置
# ==============================================================================

CROSS_COMPILE ?= riscv32-unknown-elf-
CC      = $(CROSS_COMPILE)gcc
CXX     = $(CROSS_COMPILE)g++
AS      = $(CROSS_COMPILE)as
LD      = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump
SIZE    = $(CROSS_COMPILE)size

# ==============================================================================
# 目标配置
# ==============================================================================

TARGET = hive_reflex
BUILD_DIR = build

# ==============================================================================
# RISC-V 架构参数
# ==============================================================================

ARCH = rv32imac
ABI  = ilp32

# ==============================================================================
# 编译选项
# ==============================================================================

# CPU 和架构
CPU_FLAGS = -march=$(ARCH) -mabi=$(ABI) -mstrict-align

# 优化和调试
OPT_FLAGS = -O2 -g3

# 警告
WARN_FLAGS = -Wall -Wextra -Werror=implicit-function-declaration

# 通用 C 标志
CFLAGS = $(CPU_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS)
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -fno-common -fno-builtin
CFLAGS += -DIMC22_SYSCLK_HZ=100000000  # 100 MHz

# C++ 标志
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti

# 汇编标志
ASFLAGS = $(CPU_FLAGS) -g3

# 链接标志
LDFLAGS = $(CPU_FLAGS) -nostartfiles -nostdlib
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(TARGET).map
LDFLAGS += -T imc22_sdk/linker.ld

# ==============================================================================
# 包含路径
# ==============================================================================

INC_DIRS = \
	imc22_sdk \
	.

INCLUDES = $(addprefix -I,$(INC_DIRS))

# ==============================================================================
# 源文件
# ==============================================================================

# SDK 源文件
SDK_SRCS = \
	imc22_sdk/startup.c \
	imc22_sdk/imc22_can.c \
	imc22_sdk/imc22_npu.c \
	imc22_sdk/imc22_power.c \
	imc22_sdk/imc22_cim.c \
	imc22_sdk/imc22_flash.c \
	imc22_sdk/imc22_nvs.c \
	imc22_sdk/model_loader.c

# 应用源文件 (可通过命令行指定)
APP_SRCS ?= examples/example_reflex_inference.c

# 所有 C 源文件
C_SRCS = $(SDK_SRCS) $(APP_SRCS)

# 对象文件
OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(C_SRCS))

# ==============================================================================
# 构建规则
# ==============================================================================

.PHONY: all clean flash size disasm

all: $(BUILD_DIR)/$(TARGET).bin $(BUILD_DIR)/$(TARGET).hex
	@echo "✓ 构建完成"
	@$(SIZE) $(BUILD_DIR)/$(TARGET).elf

# 链接
$(BUILD_DIR)/$(TARGET).elf: $(OBJS)
	@echo "链接: $@"
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) $^ -o $@ -lm -lc -lgcc
	@echo "✓ 链接完成"

# 生成 BIN 文件
$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	@echo "生成 BIN: $@"
	$(OBJCOPY) -O binary $< $@

# 生成 HEX 文件
$(BUILD_DIR)/$(TARGET).hex: $(BUILD_DIR)/$(TARGET).elf
	@echo "生成 HEX: $@"
	$(OBJCOPY) -O ihex $< $@

# 编译 C 文件
$(BUILD_DIR)/%.o: %.c
	@echo "编译: $<"
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 清理
clean:
	@echo "清理构建目录..."
	rm -rf $(BUILD_DIR)
	@echo "✓ 清理完成"

# 烧录 (使用 OpenOCD)
flash: $(BUILD_DIR)/$(TARGET).bin
	@echo "烧录固件..."
	openocd -f interface/jlink.cfg -f target/riscv.cfg \
		-c "init" \
		-c "reset halt" \
		-c "flash write_image erase $(BUILD_DIR)/$(TARGET).bin 0x08000000" \
		-c "verify_image $(BUILD_DIR)/$(TARGET).bin 0x08000000" \
		-c "reset run" \
		-c "shutdown"

# 显示大小
size: $(BUILD_DIR)/$(TARGET).elf
	@$(SIZE) $<
	@$(SIZE) -A -x $<

# 生成反汇编
disasm: $(BUILD_DIR)/$(TARGET).elf
	@echo "生成反汇编: $(BUILD_DIR)/$(TARGET).dis"
	$(OBJDUMP) -D $< > $(BUILD_DIR)/$(TARGET).dis
	@echo "✓ 反汇编完成"

# 依赖关系
-include $(OBJS:.o=.d)

# 生成依赖文件
$(BUILD_DIR)/%.d: %.c
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) $(INCLUDES) -MM -MT $(@:.d=.o) $< -MF $@

# ==============================================================================
# 帮助信息
# ==============================================================================

help:
	@echo "Hive-Reflex 2.0 构建系统"
	@echo ""
	@echo "目标:"
	@echo "  all      - 构建所有 (默认)"
	@echo "  clean    - 清理构建文件"
	@echo "  flash    - 烧录到芯片"
	@echo "  size     - 显示程序大小"
	@echo "  disasm   - 生成反汇编文件"
	@echo "  help     - 显示此帮助"
	@echo ""
	@echo "变量:"
	@echo "  APP_SRCS - 应用源文件 (如: APP_SRCS=my_app.c)"
	@echo ""
	@echo "示例:"
	@echo "  make APP_SRCS=examples/example_hive2_power.c"
	@echo "  make flash"
