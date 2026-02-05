/**
 * @file can_sync.h
 * @brief CAN 总线同步机制
 *
 * 使用高优先级 SYNC 帧实现:
 * - 全网时间同步 (精度 < 100μs)
 * - 确定性消息调度
 * - 消除总线仲裁抖动
 *
 * 协议设计:
 *   ID 0x000: SYNC 帧 (最高优先级)
 *   周期: 1ms
 *   DLC: 8 bytes (timestamp_us + sequence)
 */

#ifndef CAN_SYNC_H
#define CAN_SYNC_H

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 配置
// ============================================================================

#define CAN_SYNC_ID 0x000        /**< SYNC 帧 CAN ID (最高优先级) */
#define CAN_SYNC_PERIOD_US 1000  /**< SYNC 周期 (1ms) */
#define CAN_SYNC_TIMEOUT_US 5000 /**< SYNC 超时 (5ms) */

#define CAN_ID_EMERGENCY 0x080 /**< 紧急停止 */
#define CAN_ID_CONTROL 0x100   /**< 平衡控制 */
#define CAN_ID_VISION 0x200    /**< 视觉反馈 */
#define CAN_ID_TELEMETRY 0x300 /**< 遥测数据 */

// ============================================================================
// SYNC 帧数据结构
// ============================================================================

/**
 * @brief CAN SYNC 帧 (8 bytes)
 */
typedef struct __attribute__((packed)) {
  uint32_t timestamp_us; /**< 微秒级时间戳 */
  uint16_t sequence;     /**< 序列号 (用于丢包检测) */
  uint8_t reserved[2];   /**< 保留 */
} CANSyncFrame_t;

/**
 * @brief SYNC 管理器状态
 */
typedef struct {
  // 时间同步
  uint32_t local_phase_us;    /**< 本地相位 (0 - SYNC_PERIOD) */
  uint32_t last_sync_time_us; /**< 上次接收 SYNC 的时间 */

  // 序列管理
  uint16_t last_sequence; /**< 上次接收的序列号 */
  uint32_t sync_count;    /**< 接收计数 */
  uint32_t missed_count;  /**< 丢包计数 */

  // 状态标志
  bool is_synchronized; /**< 是否已同步 */
  bool is_master;       /**< 是否为 SYNC 发送节点 */

  // 回调函数
  void (*on_sync_callback)(uint32_t phase_us);

} CANSyncMgr_t;

// ============================================================================
// 初始化
// ============================================================================

/**
 * @brief 初始化 CAN SYNC 管理器
 *
 * @param mgr 管理器指针
 * @param is_master 是否为 SYNC 发送节点 (通常只有一个)
 * @param callback SYNC 到达时的回调函数
 */
void CANSync_Init(CANSyncMgr_t *mgr, bool is_master,
                  void (*callback)(uint32_t phase_us));

// ============================================================================
// Master 节点 API (SYNC 发送者)
// ============================================================================

/**
 * @brief 发送 SYNC 帧 (Master 节点调用)
 *
 * 应在定时器中断中以 1ms 周期调用
 *
 * @param mgr 管理器指针
 */
void CANSync_SendSync(CANSyncMgr_t *mgr);

// ============================================================================
// Slave 节点 API (SYNC 接收者)
// ============================================================================

/**
 * @brief 处理接收到的 SYNC 帧 (Slave 节点调用)
 *
 * 应在 CAN RX 中断中调用
 *
 * @param mgr 管理器指针
 * @param sync SYNC 帧数据
 */
void CANSync_OnSyncReceived(CANSyncMgr_t *mgr, const CANSyncFrame_t *sync);

/**
 * @brief 检查 SYNC 是否超时
 *
 * 应周期性调用 (e.g., 100Hz)
 *
 * @param mgr 管理器指针
 * @return true 已超时, false 正常
 */
bool CANSync_IsTimeout(CANSyncMgr_t *mgr);

// ============================================================================
// 时间同步 API
// ============================================================================

/**
 * @brief 获取当前同步相位 (0 - SYNC_PERIOD)
 *
 * @param mgr 管理器指针
 * @return 相位 (微秒)
 */
uint32_t CANSync_GetPhase(const CANSyncMgr_t *mgr);

/**
 * @brief 等待指定相位 (阻塞)
 *
 * @param mgr 管理器指针
 * @param target_phase_us 目标相位 (微秒)
 */
void CANSync_WaitPhase(CANSyncMgr_t *mgr, uint32_t target_phase_us);

// ============================================================================
// 调度 API (时间触发任务)
// ============================================================================

/**
 * @brief 检查是否到达指定相位
 *
 * @param mgr 管理器指针
 * @param phase_us 目标相位
 * @param tolerance_us 容错范围
 * @return true 到达, false 未到达
 */
bool CANSync_IsPhase(const CANSyncMgr_t *mgr, uint32_t phase_us,
                     uint32_t tolerance_us);

// ============================================================================
// 统计
// ============================================================================

/**
 * @brief 获取 SYNC 同步质量
 *
 * @param mgr 管理器指针
 * @return 同步率 (0.0 - 1.0), 1.0 表示完美同步
 */
float CANSync_GetSyncQuality(const CANSyncMgr_t *mgr);

/**
 * @brief 打印 SYNC 统计信息
 */
void CANSync_PrintStats(const CANSyncMgr_t *mgr);

/**
 * @brief 重置统计信息
 */
void CANSync_ResetStats(CANSyncMgr_t *mgr);

// ============================================================================
// 示例用法
// ============================================================================

/*
// Master 节点 (主控节点)
CANSyncMgr_t sync_mgr;
CANSync_Init(&sync_mgr, true, NULL);  // is_master = true

void Timer_1ms_ISR(void) {
    CANSync_SendSync(&sync_mgr);  // 发送 SYNC
}

// Slave 节点 (从节点)
CANSyncMgr_t sync_mgr;

void on_sync(uint32_t phase_us) {
    // SYNC 到达，执行时间触发任务

    // t = 100μs: 采样传感器
    if (CANSync_IsPhase(&sync_mgr, 100, 10)) {
        SampleIMU();
    }

    // t = 500μs: 运行控制循环
    if (CANSync_IsPhase(&sync_mgr, 500, 10)) {
        RunControlLoop();
    }

    // t = 800μs: 发送 CAN 消息
    if (CANSync_IsPhase(&sync_mgr, 800, 10)) {
        CAN_SendControlCmd();
    }
}

CANSync_Init(&sync_mgr, false, on_sync);  // is_master = false

void CAN_RX_ISR(void) {
    if (can_id == CAN_SYNC_ID) {
        CANSyncFrame_t sync;
        CAN_ReadData(&sync, 8);
        CANSync_OnSyncReceived(&sync_mgr, &sync);
    }
}
*/

#ifdef __cplusplus
}
#endif

#endif // CAN_SYNC_H
