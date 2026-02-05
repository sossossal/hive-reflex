/**
 * Flash 固件加载器头文件
 *
 * @file flash_loader.h
 */

#ifndef FLASH_LOADER_H
#define FLASH_LOADER_H

#include &lt; stdint.h & gt;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 验证 Flash 中的固件
 *
 * @param flash_base_addr Flash 起始地址
 * @return 0 成功, -1 失败
 */
int flash_firmware_validate(uint32_t flash_base_addr);

/**
 * 加载指定切片到 CIM SRAM
 *
 * @param slice_index 切片索引 (0-based)
 * @param cim_sram_base CIM SRAM 基地址
 * @return 0 成功, -1 失败
 */
int flash_load_slice(uint8_t slice_index, uint32_t cim_sram_base);

/**
 * 加载完整模型（所有切片）
 *
 * @param flash_addr Flash 固件地址
 * @param cim_sram_base CIM SRAM 基地址
 * @return 0 成功, -1 失败
 */
int flash_load_full_model(uint32_t flash_addr, uint32_t cim_sram_base);

/**
 * 获取固件中的切片总数
 */
uint16_t flash_get_num_slices(void);

/**
 * 获取当前加载的切片索引
 */
uint8_t flash_get_current_slice(void);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_LOADER_H */
