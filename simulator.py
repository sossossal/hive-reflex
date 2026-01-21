"""
Hive-Reflex 简易仿真器
用于验证控制算法和神经网络反射行为
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimpleJointSimulator:
    """单关节物理仿真器"""
    def __init__(self, mass=1.0, inertia=0.1, damping=0.5):
        self.mass = mass
        self.inertia = inertia  # 转动惯量
        self.damping = damping  # 阻尼系数
        
        # 状态变量
        self.angle = 0.0      # 当前角度 (rad)
        self.velocity = 0.0   # 角速度 (rad/s)
        self.acceleration = 0.0
        
        # 目标
        self.target_angle = 0.0
        
    def apply_torque(self, torque):
        """施加力矩，更新物理状态"""
        # 角加速度 = (力矩 - 阻尼) / 转动惯量
        self.acceleration = (torque - self.damping * self.velocity) / self.inertia
        
    def step(self, dt=0.001):
        """时间步进 (1ms)"""
        self.velocity += self.acceleration * dt
        self.angle += self.velocity * dt
        
    def get_imu_data(self):
        """模拟 IMU 读数 (带噪声)"""
        noise = np.random.normal(0, 0.01, 3)
        return {
            'gyro': np.array([self.velocity, 0, 0]) + noise,
            'accel': np.array([self.acceleration, 0, 0]) + noise,
        }

class PIDController:
    """经典 PID 控制器"""
    def __init__(self, kp=10.0, ki=0.1, kd=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0
        self.prev_error = 0
        
    def compute(self, target, current, dt=0.001):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class DummyReflexNet:
    """虚拟反射网络 (用简单规则模拟)"""
    def __init__(self):
        self.hidden_state = np.zeros(16)
        
    def infer(self, gyro, accel, current_load):
        """
        模拟反射行为:
        - 检测到突然加速 -> 增加阻力 (防止过冲)
        - 检测到外力干扰 -> 产生补偿力矩
        """
        # 简单规则: 如果角加速度过大，产生反向力矩
        if abs(accel[0]) > 5.0:
            reflex_torque = -np.sign(accel[0]) * 0.3
        else:
            reflex_torque = 0.0
            
        return np.clip(reflex_torque, -1.0, 1.0)

def run_simulation(duration=5.0, compliance=0.5, apply_disturbance=True):
    """
    运行仿真
    
    参数:
        duration: 仿真时长 (秒)
        compliance: 柔顺系数 (0=纯PID, 1=纯反射)
        apply_disturbance: 是否在中途施加干扰
    """
    dt = 0.001  # 1kHz
    steps = int(duration / dt)
    
    # 初始化组件
    joint = SimpleJointSimulator()
    pid = PIDController()
    reflex = DummyReflexNet()
    
    # 设置目标
    joint.target_angle = np.pi / 4  # 45度
    
    # 记录数据
    time_data = []
    angle_data = []
    torque_data = []
    reflex_data = []
    
    for i in range(steps):
        t = i * dt
        
        # 读取传感器
        imu = joint.get_imu_data()
        
        # PID 控制
        pid_torque = pid.compute(joint.target_angle, joint.angle, dt)
        
        # 反射网络
        reflex_torque = reflex.infer(imu['gyro'], imu['accel'], 0.0)
        
        # 叠加控制 (实现 compliance 混合)
        final_torque = pid_torque * (1 - compliance) + reflex_torque * compliance * 5.0
        
        # 施加外部干扰 (模拟突然碰撞)
        if apply_disturbance and 2.0 < t < 2.1:
            final_torque += 10.0  # 突然施加大力矩
        
        # 更新物理
        joint.apply_torque(final_torque)
        joint.step(dt)
        
        # 记录数据 (每10ms记录一次)
        if i % 10 == 0:
            time_data.append(t)
            angle_data.append(np.degrees(joint.angle))
            torque_data.append(final_torque)
            reflex_data.append(reflex_torque)
    
    return time_data, angle_data, torque_data, reflex_data, joint.target_angle

if __name__ == "__main__":
    print("Hive-Reflex 仿真器")
    print("=" * 50)
    
    # 对比实验: 不同 compliance 下的响应
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    for compliance, label in [(0.0, "纯PID (刚性)"), (0.5, "混合模式"), (0.9, "高柔顺")]:
        t, angle, torque, reflex, target = run_simulation(
            duration=5.0, 
            compliance=compliance, 
            apply_disturbance=True
        )
        
        axes[0].plot(t, angle, label=f"{label} (γ={compliance})")
        axes[1].plot(t, torque, label=f"{label}")
    
    # 目标线
    axes[0].axhline(y=np.degrees(target), color='r', linestyle='--', label='目标角度')
    axes[0].axvspan(2.0, 2.1, alpha=0.2, color='red', label='干扰时段')
    
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('角度 (度)')
    axes[0].set_title('关节角度响应 (含外部干扰)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel('力矩 (N·m)')
    axes[1].set_title('控制力矩输出')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_result.png', dpi=150)
    print("\n✓ 仿真完成，结果已保存为 simulation_result.png")
    plt.show()
