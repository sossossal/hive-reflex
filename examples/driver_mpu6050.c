/**
 * Example User Driver: MPU6050 IMU
 * Implements HEx API for a Motion Capture Application.
 */

#include "device_interface.h"
#include "imc22_hal.h" // Hypothetical HAL
#include <stdio.h>

// I2C Config
#define MPU_ADDR 0x68

int HEx_Device_Init(void) {
  // 1. Init I2C
  I2C_Config_t cfg = {.freq = 400000};
  HAL_I2C_Init(I2C0, &cfg);

  // 2. Wake up MPU6050
  uint8_t data[] = {0x6B, 0x00};
  HAL_I2C_Write(I2C0, MPU_ADDR, data, 2);

  printf("[HEx] MPU6050 Initialized\\n");
  return 0;
}

int HEx_Device_Read(float *input_tensor, size_t size_bytes) {
  // We expect 3 inputs: Accel X, Y, Z
  int16_t raw_data[3];
  uint8_t reg_addr = 0x3B;

  // Read 6 bytes starting from ACCEL_XOUT_H
  HAL_I2C_Write(I2C0, MPU_ADDR, &reg_addr, 1);
  HAL_I2C_Read(I2C0, MPU_ADDR, (uint8_t *)raw_data, 6);

  // Normalize to g (assuming +/- 2g range)
  // 16384 LSB/g
  input_tensor[0] = (float)raw_data[0] / 16384.0f;
  input_tensor[1] = (float)raw_data[1] / 16384.0f;
  input_tensor[2] = (float)raw_data[2] / 16384.0f;

  // printf("Accel: %.2f %.2f %.2f\\n", input_tensor[0], input_tensor[1],
  // input_tensor[2]);
  return 3;
}

int HEx_Device_Act(const float *output_tensor, size_t size_bytes) {
  // Output is "Stiffness" or "Alert Level"
  float stiffness = output_tensor[0];

  if (stiffness > 0.8f) {
    // High stiffness -> Turn on Red LED
    HAL_GPIO_Set(GPIO_PIN_LED_RED, 1);
  } else {
    HAL_GPIO_Set(GPIO_PIN_LED_RED, 0);
  }

  return 0;
}

void HEx_Device_Sleep(void) {
  // Set MPU to cycle mode (Wake-on-Motion) if supported
}
