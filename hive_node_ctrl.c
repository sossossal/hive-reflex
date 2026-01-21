#include <stdint.h>
#include "can_protocol.h"
#include "imc_driver.h" // 假设这是 IMC 加速器的驱动

// --- 全局状态 ---
typedef struct {
    float target_angle;     // 来自总线的目标
    float compliance;       // 柔顺系数 (0.0 = 刚性/无反射, 1.0 = 柔性/全反射)
    float current_angle;    // 本地传感器
    float current_load;     // 本地电流
    uint32_t last_can_rx_time; // 最后一次接收 CAN 消息的时间戳 (ms)
} NodeState;

NodeState g_state;

// LSTM 隐藏状态持久化 (16维)
static float lstm_hidden[16] = {0};
static float lstm_cell[16] = {0};

// 安全配置
#define CAN_TIMEOUT_MS 100  // CAN 通信超时阈值
#define SAFE_COMPLIANCE 0.9 // 安全模式下的柔顺度

// --- 中断服务程序: 1kHz (1ms) ---
// 这是核心控制环路，必须在 1ms 内跑完
void Timer_1kHz_ISR(void) {
    // 1. 读取传感器 (SPI DMA)
    IMU_Data_t imu_raw;
    IMU_Read_NonBlocking(&imu_raw);
    
    // 2. 准备 IMC 输入向量
    // [Gyro, Accel, Error, Current ...]
    int8_t input_tensor[12]; 
    Preprocess_Sensors(imu_raw, g_state.target_angle, input_tensor);

    // 3. 触发 IMC 加速器 (存内计算)
    // 启动 LSTM 推理，传入隐藏状态。预计耗时 < 50us
    IMC_Run_Inference(input_tensor, lstm_hidden, lstm_cell); 
    
    // 4. 等待计算完成 (Polling 或 Wait-for-Interrupt)
    while(IMC_Is_Busy()); 
    
    // 5. 获取反射修正量，并更新隐藏状态
    float reflex_torque = IMC_Get_Output(lstm_hidden, lstm_cell); // 范围 -1.0 到 1.0
    
    // 6. 通信超时检测
    uint32_t current_time = Millis(); // 获取系统时间 (ms)
    bool is_timeout = (current_time - g_state.last_can_rx_time) > CAN_TIMEOUT_MS;
    
    float effective_compliance = g_state.compliance;
    if (is_timeout) {
        // 进入安全模式：提高柔顺度，目标角度锁定为当前角度
        effective_compliance = SAFE_COMPLIANCE;
        g_state.target_angle = g_state.current_angle;
    }
    
    // 7. 叠加控制律 (Superposition Control)
    // 基础指令 (Base Command)
    float pid_out = PID_Controller(g_state.target_angle, g_state.current_angle);
    
    // 最终输出 = PID * (1-柔顺度) + 反射 * 柔顺度
    // compliance = 0: 完全刚性，仅 PID 控制
    // compliance = 1: 完全柔性，仅反射控制
    float final_pwm = pid_out * (1.0f - effective_compliance) + 
                      (reflex_torque * effective_compliance * MAX_TORQUE);
    
    // 8. 写入电机
    Motor_Set_PWM(final_pwm);
    
    // 9. 广播状态 (可选，每 10ms 一次)
    if (tick_count % 10 == 0) {
        CAN_Broadcast_Status(g_state);
    }
}

// --- CAN 接收中断 ---
void CAN_Rx_ISR(void) {
    CanMsg msg;
    CAN_Read(&msg);
    
    if (msg.id == CMD_ID_BASE + MY_NODE_ID) {
        // 更新目标和柔顺度
        g_state.target_angle = msg.data_float[0];
        g_state.compliance   = msg.data_float[1];
        g_state.last_can_rx_time = Millis(); // 更新接收时间戳
    } else if (msg.id == GLOBAL_SYNC_ID) {
        // 时间同步
        Sync_Timer(msg.timestamp);
    }
}

// --- 主函数 ---
int main(void) {
    // 1. 初始化硬件
    System_Init();
    IMC_PowerOn();
    
    // 2. 握手阶段
    // 广播自己的 ID，告诉总控 "我上线了"
    CAN_Send_Handshake(MY_TYPE_LEG, MY_UNIQUE_ID);
    
    // 3. 加载权重到 IMC SRAM
    // 从 Flash 读取 reflex_net.bin 并锁定在 Bank 0
    IMC_Load_Weights("reflex_net.bin");
    
    // 4. 开启 1kHz 定时器
    Timer_Start();
    
    while(1) {
        // 低优先级任务：LED 闪烁，参数保存等
        LowPriority_Tasks();
    }
}
