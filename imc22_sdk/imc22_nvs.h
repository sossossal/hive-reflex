/**
 * @file imc22_nvs.h
 * @brief Non-Volatile Storage (NVS) - FLASH 非易失性存储系统
 * @version 2.0
 * @date 2026-01-19
 *
 * 基于 NOR Flash 的键值对存储系统，支持断电数据保护和磨损均衡
 */

#ifndef IMC22_NVS_H
#define IMC22_NVS_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 配置参数                                                                  */
/* ========================================================================= */

#define NVS_FLASH_BASE 0x08190000  // 配置分区起始地址
#define NVS_FLASH_SIZE (64 * 1024) // 配置分区大小 (64KB)
#define NVS_MAX_KEY_LEN 32         // 最大键名长度
#define NVS_MAX_VALUE_LEN 1024     // 最大值长度
#define NVS_SECTOR_SIZE 4096       // Flash 扇区大小

/* ========================================================================= */
/* 数据类型定义                                                              */
/* ========================================================================= */

typedef enum {
  NVS_TYPE_U8 = 0, /**< uint8_t */
  NVS_TYPE_I8,     /**< int8_t */
  NVS_TYPE_U16,    /**< uint16_t */
  NVS_TYPE_I16,    /**< int16_t */
  NVS_TYPE_U32,    /**< uint32_t */
  NVS_TYPE_I32,    /**< int32_t */
  NVS_TYPE_FLOAT,  /**< float */
  NVS_TYPE_DOUBLE, /**< double */
  NVS_TYPE_STR,    /**< 字符串 */
  NVS_TYPE_BLOB    /**< 二进制数据 */
} NVS_Type_t;

typedef struct {
  char key[NVS_MAX_KEY_LEN]; /**< 键名 */
  NVS_Type_t type;           /**< 数据类型 */
  uint32_t size;             /**< 数据大小 */
  uint32_t crc32;            /**< CRC32 校验 */
  uint32_t flash_addr;       /**< Flash 地址 */
} NVS_Entry_t;

typedef struct {
  uint32_t total_entries; /**< 总条目数 */
  uint32_t used_bytes;    /**< 已使用字节数 */
  uint32_t free_bytes;    /**< 剩余字节数 */
  uint32_t erase_count;   /**< 擦除次数 */
} NVS_Stats_t;

/* ========================================================================= */
/* 公共 API                                                                  */
/* ========================================================================= */

/**
 * @brief 初始化 NVS 系统
 * @return 0 成功, -1 失败
 */
int NVS_Init(void);

/**
 * @brief 格式化 NVS 分区 (清除所有数据)
 * @return 0 成功, -1 失败
 */
int NVS_Format(void);

/**
 * @brief 写入数据
 * @param key 键名
 * @param data 数据指针
 * @param size 数据大小
 * @param type 数据类型
 * @return 0 成功, -1 失败
 */
int NVS_Write(const char *key, const void *data, uint32_t size,
              NVS_Type_t type);

/**
 * @brief 读取数据
 * @param key 键名
 * @param data 数据缓冲区
 * @param max_size 缓冲区大小
 * @return 实际读取的字节数, -1 表示失败
 */
int NVS_Read(const char *key, void *data, uint32_t max_size);

/**
 * @brief 删除数据
 * @param key 键名
 * @return 0 成功, -1 失败
 */
int NVS_Erase(const char *key);

/**
 * @brief 检查键是否存在
 * @param key 键名
 * @return true 存在, false 不存在
 */
bool NVS_Exists(const char *key);

/**
 * @brief 获取存储统计信息
 * @param stats 统计结构体指针
 */
void NVS_GetStats(NVS_Stats_t *stats);

/* ========================================================================= */
/* 便捷 API (类型安全)                                                       */
/* ========================================================================= */

/**
 * @brief 写入 uint32_t
 */
static inline int NVS_WriteU32(const char *key, uint32_t value) {
  return NVS_Write(key, &value, sizeof(value), NVS_TYPE_U32);
}

/**
 * @brief 读取 uint32_t
 */
static inline uint32_t NVS_ReadU32(const char *key, uint32_t default_val) {
  uint32_t value = default_val;
  NVS_Read(key, &value, sizeof(value));
  return value;
}

/**
 * @brief 写入 float
 */
static inline int NVS_WriteFloat(const char *key, float value) {
  return NVS_Write(key, &value, sizeof(value), NVS_TYPE_FLOAT);
}

/**
 * @brief 读取 float
 */
static inline float NVS_ReadFloat(const char *key, float default_val) {
  float value = default_val;
  NVS_Read(key, &value, sizeof(value));
  return value;
}

/**
 * @brief 写入字符串
 */
static inline int NVS_WriteString(const char *key, const char *str) {
  return NVS_Write(key, str, strlen(str) + 1, NVS_TYPE_STR);
}

/**
 * @brief 读取字符串
 */
static inline int NVS_ReadString(const char *key, char *buf, uint32_t max_len) {
  return NVS_Read(key, buf, max_len);
}

/* ========================================================================= */
/* 高级功能                                                                  */
/* ========================================================================= */

/**
 * @brief 提交所有挂起的写操作 (确保数据持久化)
 * @return 0 成功, -1 失败
 */
int NVS_Commit(void);

/**
 * @brief 遍历所有条目
 * @param callback 回调函数 (返回 false 停止遍历)
 * @param user_data 用户数据
 */
void NVS_Iterate(bool (*callback)(const NVS_Entry_t *entry, void *user_data),
                 void *user_data);

/**
 * @brief 磨损均衡检查 (自动在后台运行)
 */
void NVS_WearLeveling(void);

/* ========================================================================= */
/* 预定义配置键名                                                            */
/* ========================================================================= */

// 系统配置
#define NVS_KEY_NODE_ID "node.id"
#define NVS_KEY_CAN_BITRATE "can.bitrate"
#define NVS_KEY_BOOT_COUNT "sys.boot_count"

// PID 参数
#define NVS_KEY_PID_KP "pid.kp"
#define NVS_KEY_PID_KI "pid.ki"
#define NVS_KEY_PID_KD "pid.kd"

// IMU 校准
#define NVS_KEY_IMU_GYRO_BIAS_X "imu.gyro_bias_x"
#define NVS_KEY_IMU_GYRO_BIAS_Y "imu.gyro_bias_y"
#define NVS_KEY_IMU_GYRO_BIAS_Z "imu.gyro_bias_z"
#define NVS_KEY_IMU_ACCEL_SCALE "imu.accel_scale"

// 神经网络配置
#define NVS_KEY_NN_MODEL_VER "nn.model_version"
#define NVS_KEY_NN_QUANT_SCALE "nn.quant_scale"
#define NVS_KEY_NN_COMPLIANCE "nn.compliance"

/* ========================================================================= */
/* 内部函数 (供高级用户使用)                                                 */
/* ========================================================================= */

/**
 * @brief 计算 CRC32
 */
uint32_t NVS_CalcCRC32(const void *data, uint32_t size);

/**
 * @brief 直接读取 Flash
 */
int NVS_FlashRead(uint32_t addr, void *buf, uint32_t size);

/**
 * @brief 直接写入 Flash
 */
int NVS_FlashWrite(uint32_t addr, const void *data, uint32_t size);

/**
 * @brief 擦除 Flash 扇区
 */
int NVS_FlashEraseSector(uint32_t addr);

#ifdef __cplusplus
}
#endif

#endif /* IMC22_NVS_H */
