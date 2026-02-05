#ifndef __SCHEDULER_H__
#define __SCHEDULER_H__

#include <stdint.h>

void Scheduler_Init(void);

// ... (typedefs for TaskType_t etc would be shared here, simplified for header)

// Simple API for now
int Scheduler_RunGraph(void *task_list, uint32_t count);

#endif // __SCHEDULER_H__
