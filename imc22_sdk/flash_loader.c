/**
 * Hive-Reflex Flash 固件加载器
 * 从 Flash 加载量化模型到 CIM SRAM
 * 
 * @file flash_loader.c
 * @version 2.1.0
 */

#include &lt;stdint.h&gt;
#include &lt;string.h&gt;
#include &lt;stdio.h&gt;

#include "flash_loader.h"
#include "imc22_cim.h"

// ============================================================================
// 固件格式定义
// ============================================================================

#define FIRMWARE_MAGIC 0x32465248  // 'HRF2' (little-endian)
#define FIRMWARE_VERSION_2_1 0x0210

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t version;
    uint16_t num_slices;
    uint32_t total_size;
    uint16_t metadata_length;
    uint16_t reserved;
} firmware_header_t;

typedef struct __attribute__((packed)) {
    uint32_t slice_size;
    uint16_t layer_count;
    uint16_t reserved;
} slice_header_t;

typedef struct {
    char name[64];
    uint8_t num_dims;
    uint32_t shape[4];
    float scale;
    uint32_t weight_size;
    int8_t* weights;
} layer_info_t;

// ============================================================================
// 全局状态
// ============================================================================

static uint32_t current_flash_offset = 0;
static uint8_t current_slice_index = 0;
static uint16_t total_slices = 0;

// ============================================================================
// Flash 读取接口（需要移植）
// ============================================================================

#ifdef PLATFORM_ZCU102
// ZCU102 QSPI Flash 读取
extern void qspi_flash_read(uint32_t addr, uint8_t* buf, uint32_t len);
#define FLASH_READ(addr, buf, len) qspi_flash_read(addr, buf, len)
#else
// 模拟 Flash（从内存读取）
static uint8_t* sim_flash_base = NULL;
#define FLASH_READ(addr, buf, len) memcpy(buf, sim_flash_base + addr, len)
#endif

// ============================================================================
// 固件验证
// ============================================================================

int flash_firmware_validate(uint32_t flash_base_addr) {
    firmware_header_t header;
    
    FLASH_READ(flash_base_addr, (uint8_t*)&amp;header, sizeof(header));
    
    // 检查魔数
    if (header.magic != FIRMWARE_MAGIC) {
        printf("[ERROR] Invalid firmware magic: 0x%08X\n", header.magic);
        return -1;
    }
    
    // 检查版本
    if (header.version != FIRMWARE_VERSION_2_1) {
        printf("[WARN] Firmware version mismatch: 0x%04X (expected 0x%04X)\n",
               header.version, FIRMWARE_VERSION_2_1);
    }
    
    printf("[INFO] Firmware validated:\n");
    printf("       Version: %d.%d.0\n", 
           (header.version &gt;&gt; 8) &amp; 0xF, 
           (header.version &gt;&gt; 4) &amp; 0xF);
    printf("       Slices: %d\n", header.num_slices);
    printf("       Total Size: %d KB\n", header.total_size / 1024);
    
    total_slices = header.num_slices;
    current_flash_offset = flash_base_addr;
    
    return 0;
}

// ============================================================================
// 加载切片
// ============================================================================

int flash_load_slice(uint8_t slice_index, uint32_t cim_sram_base) {
    if (slice_index &gt;= total_slices) {
        printf("[ERROR] Slice index %d out of range (max %d)\n", 
               slice_index, total_slices - 1);
        return -1;
    }
    
    printf("[INFO] Loading slice %d/%d...\n", slice_index + 1, total_slices);
    
    // 跳过固件头部和元数据
    firmware_header_t header;
    FLASH_READ(current_flash_offset, (uint8_t*)&amp;header, sizeof(header));
    
    uint32_t offset = current_flash_offset + sizeof(firmware_header_t);
    
    // 跳过元数据
    uint8_t metadata_buf[header.metadata_length];
    FLASH_READ(offset, metadata_buf, header.metadata_length);
    offset += header.metadata_length;
    
    // 找到目标切片
    for (uint8_t i = 0; i &lt; slice_index; i++) {
        slice_header_t slice_hdr;
        FLASH_READ(offset, (uint8_t*)&amp;slice_hdr, sizeof(slice_hdr));
        offset += sizeof(slice_hdr) + slice_hdr.slice_size;
    }
    
    // 读取切片头部
    slice_header_t slice_hdr;
    FLASH_READ(offset, (uint8_t*)&amp;slice_hdr, sizeof(slice_hdr));
    offset += sizeof(slice_hdr);
    
    printf("       Layers: %d\n", slice_hdr.layer_count);
    printf("       Size: %d KB\n", slice_hdr.slice_size / 1024);
    
    // 加载各层到 CIM SRAM
    uint32_t sram_offset = 0;
    
    for (uint16_t layer_idx = 0; layer_idx &lt; slice_hdr.layer_count; layer_idx++) {
        layer_info_t layer;
        
        // 读取层名称长度
        uint16_t name_len;
        FLASH_READ(offset, (uint8_t*)&amp;name_len, sizeof(name_len));
        offset += sizeof(name_len);
        
        // 读取层名称
        FLASH_READ(offset, (uint8_t*)layer.name, name_len);
        layer.name[name_len] = '\0';
        offset += name_len;
        
        // 读取形状
        FLASH_READ(offset, &amp;layer.num_dims, sizeof(layer.num_dims));
        offset += sizeof(layer.num_dims);
        
        for (uint8_t d = 0; d &lt; layer.num_dims; d++) {
            FLASH_READ(offset, (uint8_t*)&amp;layer.shape[d], sizeof(uint32_t));
            offset += sizeof(uint32_t);
        }
        
        // 读取 scale
        FLASH_READ(offset, (uint8_t*)&amp;layer.scale, sizeof(float));
        offset += sizeof(float);
        
        // 读取权重大小
        FLASH_READ(offset, (uint8_t*)&amp;layer.weight_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // 直接加载权重到 CIM SRAM
        FLASH_READ(offset, (uint8_t*)(cim_sram_base + sram_offset), layer.weight_size);
        offset += layer.weight_size;
        
        printf("       [%d] %s: shape=[", layer_idx, layer.name);
        for (uint8_t d = 0; d &lt; layer.num_dims; d++) {
            printf("%d%s", layer.shape[d], d &lt; layer.num_dims - 1 ? "," : "");
        }
        printf("], scale=%.6f, size=%d\n", layer.scale, layer.weight_size);
        
        sram_offset += layer.weight_size;
    }
    
    printf("[INFO] ✅ Slice %d loaded (%d bytes to SRAM)\n", 
           slice_index, sram_offset);
    
    current_slice_index = slice_index;
    return 0;
}

// ============================================================================
// 便捷函数
// ============================================================================

int flash_load_full_model(uint32_t flash_addr, uint32_t cim_sram_base) {
    /**
     * 加载完整模型（所有切片）
     * 
     * 注意：如果模型切片数 &gt; 1，只会加载第一个切片到 SRAM。
     *       其他切片需要应用层按需加载。
     */
    
    if (flash_firmware_validate(flash_addr) != 0) {
        return -1;
    }
    
    if (total_slices == 1) {
        // 单切片模型：直接加载
        return flash_load_slice(0, cim_sram_base);
    } else {
        // 多切片模型：仅加载第一个切片
        printf("[WARN] Multi-slice model detected (%d slices)\n", total_slices);
        printf("       Loading only slice 0. Use flash_load_slice() for others.\n");
        return flash_load_slice(0, cim_sram_base);
    }
}

uint16_t flash_get_num_slices(void) {
    return total_slices;
}

uint8_t flash_get_current_slice(void) {
    return current_slice_index;
}
