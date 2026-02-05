/**
 * Hive-Reflex Heterogeneous Scheduler
 * ===================================
 *
 * Manages task distribution between RISC-V CPU and CIM Accelerator.
 *
 * Responsibilities:
 * 1. Parse model execution plan.
 * 2. Configure CIM accelerator for offloaded subgraphs.
 * 3. Fallback to CPU for unsupported operators.
 * 4. Manage data movement (DMA) between system RAM and CIM SRAM.
 */

#include "imc22_cim.h"
#include <stdio.h>
#include <string.h>


// Task Types
typedef enum {
  TASK_TYPE_CPU = 0,
  TASK_TYPE_CIM = 1,
  TASK_TYPE_DMA = 2
} TaskType_t;

// Task Descriptor
typedef struct {
  TaskType_t type;
  uint32_t op_id;
  void *params;
  void *input_addr;
  void *output_addr;
  uint32_t data_size;
} ExecutionTask_t;

// Scheduler Context
typedef struct {
  uint32_t current_task_id;
  uint32_t total_tasks;
  ExecutionTask_t *task_queue;
  uint32_t status;
} SchedulerContext_t;

static SchedulerContext_t g_scheduler_ctx;

/**
 * @brief Initialize the runtime scheduler
 */
void Scheduler_Init(void) {
  g_scheduler_ctx.current_task_id = 0;
  g_scheduler_ctx.total_tasks = 0;
  g_scheduler_ctx.status = 0;
  printf("[Scheduler] Initialized\\n");
}

/**
 * @brief Dispatch a single task to the appropriate execution unit
 */
int Scheduler_Dispatch(ExecutionTask_t *task) {
  int ret = 0;

  switch (task->type) {
  case TASK_TYPE_CIM:
    // Offload to CIM Accelerator
    // Wait for CIM to be ready
    while (IMC22_CIM_BASE->STATUS & CIM_STATUS_BUSY)
      ;

    // In a real scenario, we would trigger the CIM command here
    // using the drivers we wrote earlier
    // ret = CIM_Execute(task->params);
    printf("[Scheduler] Dispatching Task %d to CIM Accelerator\\n",
           task->op_id);
    break;

  case TASK_TYPE_CPU:
    // Execute on CPU (Software Fallback)
    printf("[Scheduler] Dispatching Task %d to RISC-V CPU\\n", task->op_id);
    // ret = CPU_Execute_Op(task->op_id, task->input_addr, task->output_addr);
    break;

  case TASK_TYPE_DMA:
    // Simple memcpy for now, would be DMA config in hardware
    printf("[Scheduler] DMA Transfer: %d bytes\\n", task->data_size);
    memcpy(task->output_addr, task->input_addr, task->data_size);
    break;

  default:
    return -1;
  }

  return ret;
}

/**
 * @brief Run the Full Execution Graph
 */
int Scheduler_RunGraph(ExecutionTask_t *task_list, uint32_t count) {
  g_scheduler_ctx.task_queue = task_list;
  g_scheduler_ctx.total_tasks = count;

  for (uint32_t i = 0; i < count; i++) {
    ExecutionTask_t *task = &task_list[i];

    // Log start
    // printf("[Scheduler] Starting Task %d...\\n", i);

    int ret = Scheduler_Dispatch(task);
    if (ret != 0) {
      printf("[Scheduler] Task %d Failed! Error: %d\\n", i, ret);
      return ret;
    }
  }

  return 0;
}
