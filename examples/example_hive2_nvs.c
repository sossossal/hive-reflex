/**
 * @file example_hive2_nvs.c
 * @brief Hive-Reflex 2.0 非易失性存储示例
 *
 * 演示配置参数的持久化存储
 */

#include "imc22.h"
#include "imc22_nvs.h"
#include <stdio.h>

// 节点配置结构
typedef struct {
  uint8_t node_id;
  uint32_t can_bitrate;
  float pid_kp;
  float pid_ki;
  float pid_kd;
  float imu_gyro_bias[3];
  float compliance_factor;
} NodeConfig_t;

void print_config(const NodeConfig_t *cfg) {
  printf("当前配置:\n");
  printf("  Node ID: %d\n", cfg->node_id);
  printf("  CAN Bitrate: %lu bps\n", cfg->can_bitrate);
  printf("  PID: Kp=%.3f, Ki=%.3f, Kd=%.3f\n", cfg->pid_kp, cfg->pid_ki,
         cfg->pid_kd);
  printf("  IMU Gyro Bias: [%.3f, %.3f, %.3f]\n", cfg->imu_gyro_bias[0],
         cfg->imu_gyro_bias[1], cfg->imu_gyro_bias[2]);
  printf("  Compliance: %.2f\n", cfg->compliance_factor);
}

int main(void) {
  printf("Hive-Reflex 2.0 非易失性存储示例\n");
  printf("================================\n\n");

  // 1. 初始化 NVS
  if (NVS_Init() != 0) {
    printf("错误: NVS 初始化失败\n");
    return -1;
  }
  printf("✓ NVS 初始化完成\n\n");

  // 2. 尝试读取已保存的配置
  NodeConfig_t config;
  bool config_exists = false;

  if (NVS_Exists(NVS_KEY_NODE_ID)) {
    printf("检测到已保存的配置，正在加载...\n");

    config.node_id = (uint8_t)NVS_ReadU32(NVS_KEY_NODE_ID, 1);
    config.can_bitrate = NVS_ReadU32(NVS_KEY_CAN_BITRATE, 1000000);
    config.pid_kp = NVS_ReadFloat(NVS_KEY_PID_KP, 1.0f);
    config.pid_ki = NVS_ReadFloat(NVS_KEY_PID_KI, 0.1f);
    config.pid_kd = NVS_ReadFloat(NVS_KEY_PID_KD, 0.05f);

    NVS_Read(NVS_KEY_IMU_GYRO_BIAS_X, &config.imu_gyro_bias[0], sizeof(float));
    NVS_Read(NVS_KEY_IMU_GYRO_BIAS_Y, &config.imu_gyro_bias[1], sizeof(float));
    NVS_Read(NVS_KEY_IMU_GYRO_BIAS_Z, &config.imu_gyro_bias[2], sizeof(float));

    config.compliance_factor = NVS_ReadFloat(NVS_KEY_NN_COMPLIANCE, 0.5f);

    printf("✓ 配置加载完成\n\n");
    print_config(&config);
    config_exists = true;

  } else {
    printf("未找到已保存的配置，使用默认值\n");

    // 默认配置
    config.node_id = 1;
    config.can_bitrate = 1000000;
    config.pid_kp = 1.5f;
    config.pid_ki = 0.2f;
    config.pid_kd = 0.08f;
    config.imu_gyro_bias[0] = 0.01f;
    config.imu_gyro_bias[1] = -0.02f;
    config.imu_gyro_bias[2] = 0.005f;
    config.compliance_factor = 0.7f;

    print_config(&config);
  }

  // 3. 演示修改配置
  printf("\n修改配置...\n");
  config.pid_kp = 2.0f;
  config.compliance_factor = 0.8f;
  printf("  PID Kp: %.3f → 2.000\n", config.pid_kp);
  printf("  Compliance: %.2f → 0.80\n", config.compliance_factor);

  // 4. 保存配置到 FLASH
  printf("\n保存配置到 FLASH...\n");

  NVS_WriteU32(NVS_KEY_NODE_ID, config.node_id);
  NVS_WriteU32(NVS_KEY_CAN_BITRATE, config.can_bitrate);
  NVS_WriteFloat(NVS_KEY_PID_KP, config.pid_kp);
  NVS_WriteFloat(NVS_KEY_PID_KI, config.pid_ki);
  NVS_WriteFloat(NVS_KEY_PID_KD, config.pid_kd);

  NVS_Write(NVS_KEY_IMU_GYRO_BIAS_X, &config.imu_gyro_bias[0], sizeof(float),
            NVS_TYPE_FLOAT);
  NVS_Write(NVS_KEY_IMU_GYRO_BIAS_Y, &config.imu_gyro_bias[1], sizeof(float),
            NVS_TYPE_FLOAT);
  NVS_Write(NVS_KEY_IMU_GYRO_BIAS_Z, &config.imu_gyro_bias[2], sizeof(float),
            NVS_TYPE_FLOAT);

  NVS_WriteFloat(NVS_KEY_NN_COMPLIANCE, config.compliance_factor);

  // 提交写操作
  if (NVS_Commit() == 0) {
    printf("✓ 配置已保存到 FLASH\n");
  } else {
    printf("✗ 保存失败\n");
    return -1;
  }

  // 5. 统计信息
  printf("\nNVS 统计信息:\n");
  NVS_Stats_t stats;
  NVS_GetStats(&stats);
  printf("  总条目数: %lu\n", stats.total_entries);
  printf("  已使用: %lu bytes\n", stats.used_bytes);
  printf("  剩余空间: %lu bytes\n", stats.free_bytes);
  printf("  擦除次数: %lu\n", stats.erase_count);

  // 6. 模拟断电重启
  printf("\n模拟断电...\n");
  printf("重启后配置将自动恢复\n");

  printf("\n✅ 示例完成!\n");
  printf("\n提示: 实际使用中，配置会在断电后自动恢复\n");

  return 0;
}
