/**
 * Hive-Reflex Hardware Extension API (HEx API)
 * ============================================
 *
 * This interface defines how external devices (Sensors, Actuators)
 * interact with the Hive-Reflex core system.
 *
 * Users should implement these functions in their driver code.
 * The system automatically calls these hooks at the appropriate time.
 */

#ifndef __DEVICE_INTERFACE_H__
#define __DEVICE_INTERFACE_H__

#include <stddef.h>
#include <stdint.h>


/**
 * @brief Device Initialization Hook
 * Called once during system startup (Splinal Boot).
 * Users should initialize I2C/SPI/GPIO here.
 *
 * @return 0 on success, <0 on error
 */
int HEx_Device_Init(void);

/**
 * @brief Data Acquisition Hook (Input)
 * Called immediately before Model Inference.
 * Users should read sensors and populate the input tensor buffer.
 *
 * @param input_tensor Pointer to the model's input memory buffer (float32)
 * @param size_bytes Size of the input buffer in bytes
 * @return Number of samples read, or <0 on error
 */
int HEx_Device_Read(float *input_tensor, size_t size_bytes);

/**
 * @brief Action Execution Hook (Output)
 * Called immediately after Model Inference.
 * Users should read the output tensor and drive actuators (e.g., Motors, LEDs).
 *
 * @param output_tensor Pointer to the model's output memory buffer (float32)
 * @param size_bytes Size of the output buffer in bytes
 * @return 0 on success, <0 on error
 */
int HEx_Device_Act(const float *output_tensor, size_t size_bytes);

/**
 * @brief Sleep/Low-Power Hook (Optional)
 * Called when system decides to enter low power mode.
 * Users can configure sensors to Wake-on-Motion mode here.
 */
void HEx_Device_Sleep(void);

#endif // __DEVICE_INTERFACE_H__
