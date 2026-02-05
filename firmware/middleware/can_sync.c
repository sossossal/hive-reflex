/**
 * @file can_sync.c
 * @brief CAN 同步机制实现
 */

#include "can_sync.h"
#include <stdio.h>
#include <string.h>


// 外部函数（需由HAL层提供）
extern uint32_t GetTimestampUs(void);
extern void DelayUs(uint32_t us);
extern void CAN_Transmit(uint16_t id, const void *data, uint8_t dlc);

// ============================================================================
// 初始化
// ============================================================================

void CANSync_Init(CANSyncMgr_t *mgr, bool is_master,
                  void (*callback)(uint32_t phase_us)) {
  if (!mgr)
    return;

  memset(mgr, 0, sizeof(CANSyncMgr_t));
  mgr->is_master = is_master;
  mgr->on_sync_callback = callback;
  mgr->is_synchronized = false;
}

// ============================================================================
// Master 节点实现
// ============================================================================

void CANSync_SendSync(CANSyncMgr_t *mgr) {
  if (!mgr || !mgr->is_master)
    return;

  // 构造 SYNC 帧
  CANSyncFrame_t sync = {.timestamp_us = GetTimestampUs(),
                         .sequence = (uint16_t)(mgr->sync_count & 0xFFFF),
                         .reserved = {0, 0}};

  // 发送 (ID 0x000 最高优先级)
  CAN_Transmit(CAN_SYNC_ID, &sync, sizeof(sync));

  // 更新状态
  mgr->sync_count++;
  mgr->last_sync_time_us = sync.timestamp_us;
  mgr->is_synchronized = true;
}

// ============================================================================
// Slave 节点实现
// ============================================================================

void CANSync_OnSyncReceived(CANSyncMgr_t *mgr, const CANSyncFrame_t *sync) {
  if (!mgr || !sync || mgr->is_master)
    return;

  uint32_t current_time = GetTimestampUs();

  // 检测丢包
  if (mgr->is_synchronized) {
    uint16_t expected_seq = mgr->last_sequence + 1;
    if (sync->sequence != expected_seq) {
      mgr->missed_count += (sync->sequence - expected_seq);
    }
  }

  // 更新相位
  mgr->local_phase_us =
      (current_time - sync->timestamp_us) % CAN_SYNC_PERIOD_US;
  mgr->last_sync_time_us = current_time;
  mgr->last_sequence = sync->sequence;
  mgr->sync_count++;
  mgr->is_synchronized = true;

  // 调用回调
  if (mgr->on_sync_callback) {
    mgr->on_sync_callback(mgr->local_phase_us);
  }
}

bool CANSync_IsTimeout(CANSyncMgr_t *mgr) {
  if (!mgr || !mgr->is_synchronized) {
    return false; // 尚未同步
  }

  uint32_t elapsed = GetTimestampUs() - mgr->last_sync_time_us;

  if (elapsed > CAN_SYNC_TIMEOUT_US) {
    mgr->is_synchronized = false;
    return true;
  }

  return false;
}

// ============================================================================
// 时间同步 API
// ============================================================================

uint32_t CANSync_GetPhase(const CANSyncMgr_t *mgr) {
  if (!mgr || !mgr->is_synchronized) {
    return 0;
  }

  uint32_t elapsed = GetTimestampUs() - mgr->last_sync_time_us;
  return elapsed % CAN_SYNC_PERIOD_US;
}

void CANSync_WaitPhase(CANSyncMgr_t *mgr, uint32_t target_phase_us) {
  if (!mgr || !mgr->is_synchronized)
    return;

  while (1) {
    uint32_t current_phase = CANSync_GetPhase(mgr);

    if (current_phase >= target_phase_us) {
      break;
    }

    // 短延迟避免忙等
    DelayUs(10);
  }
}

// ============================================================================
// 调度 API
// ============================================================================

bool CANSync_IsPhase(const CANSyncMgr_t *mgr, uint32_t phase_us,
                     uint32_t tolerance_us) {
  if (!mgr || !mgr->is_synchronized) {
    return false;
  }

  uint32_t current_phase = CANSync_GetPhase(mgr);

  // 检查是否在容错范围内
  if (current_phase >= phase_us - tolerance_us &&
      current_phase <= phase_us + tolerance_us) {
    return true;
  }

  return false;
}

// ============================================================================
// 统计
// ============================================================================

float CANSync_GetSyncQuality(const CANSyncMgr_t *mgr) {
  if (!mgr || mgr->sync_count == 0) {
    return 0.0f;
  }

  uint32_t total = mgr->sync_count + mgr->missed_count;
  return (float)mgr->sync_count / total;
}

void CANSync_PrintStats(const CANSyncMgr_t *mgr) {
  if (!mgr)
    return;

  printf("\n=== CAN SYNC Statistics ===\n");
  printf("Mode:         %s\n", mgr->is_master ? "Master" : "Slave");
  printf("Synchronized: %s\n", mgr->is_synchronized ? "Yes" : "No");
  printf("SYNC Count:   %lu\n", (unsigned long)mgr->sync_count);
  printf("Missed:       %lu\n", (unsigned long)mgr->missed_count);
  printf("Sync Quality: %.1f%%\n", CANSync_GetSyncQuality(mgr) * 100);

  if (mgr->is_synchronized) {
    printf("Current Phase: %lu us\n", (unsigned long)CANSync_GetPhase(mgr));
  }

  printf("===========================\n");
}

void CANSync_ResetStats(CANSyncMgr_t *mgr) {
  if (!mgr)
    return;

  mgr->sync_count = 0;
  mgr->missed_count = 0;
}
