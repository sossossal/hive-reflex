"""
Demo GIF Generator for Hive-Reflex
==================================

This script generates a visualization of the Spinal Reflex Controller in action.
It simulates a joint response under disturbance and saves the animation as a GIF.
This GIF can be used in the README.md to demonstrate the project visually.

Dependencies:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

def simulate_reflex_system(steps=200):
    """Simulate a mass-spring-damper system with reflex control."""
    # Physics parameters
    m = 1.0  # mass
    k = 10.0 # spring contant (passive)
    b = 0.5  # damping
    dt = 0.01
    
    # State
    x = 0.0
    v = 0.0
    target = 1.0
    
    # History
    x_hist = []
    u_hist = []
    
    # Simulation
    for t in range(steps):
        # Disturbance at t=50
        force_ext = -5.0 if 50 < t < 70 else 0.0
        
        # Controller (Simulated Neural Reflex)
        error = target - x
        
        # Adaptive stiffness (Simulating the Neural Net output)
        # When error is large, increase stiffness (reflex)
        reflex_gain = 5.0 + 15.0 * (1.0 / (1.0 + np.exp(-5.0 * (abs(error) - 0.2))))
        
        u_ctrl = reflex_gain * error
        
        # Dynamics
        acc = (u_ctrl + force_ext - k*x - b*v) / m
        v += acc * dt
        x += v * dt
        
        x_hist.append(x)
        u_hist.append(u_ctrl)
        
    return x_hist, u_hist

def create_demo_gif(filename="reflex_demo.gif"):
    print("Generating simulation data...")
    x_data, u_data = simulate_reflex_system()
    
    print("Setting up animation...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 2]})
    plt.subplots_adjust(hspace=0.3)
    
    # Plot 1: Response Curves
    ax1.set_xlim(0, len(x_data))
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title("Result: Joint Response vs Disturbance")
    ax1.set_ylabel("Position")
    ax1.grid(True, alpha=0.3)
    
    line_x, = ax1.plot([], [], 'b-', lw=2, label='Position')
    line_target, = ax1.plot([0, len(x_data)], [1, 1], 'g--', label='Target')
    ax1.legend(loc='lower right')
    
    # Plot 2: Physical Visualization
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("Visual: Reflex Action")
    
    # Drawing elements
    base = Rectangle((-0.2, -0.2), 0.4, 0.4, color='gray') # Fixed base
    link = val = ax2.plot([], [], 'k-', lw=5)[0] # Arm link
    joint = Circle((0, 0), 0.1, color='r') # Joint
    mass = Circle((0, 0), 0.15, color='b') # End effector
    
    ax2.add_patch(base)
    ax2.add_patch(joint)
    ax2.add_patch(mass)
    
    def init():
        line_x.set_data([], [])
        return line_x, link, mass

    def update(frame):
        # Update curve
        current_x = x_data[:frame]
        line_x.set_data(range(len(current_x)), current_x)
        
        # Update physical arm (mapping linear x to angle for visualization)
        # x=0 -> 0 deg, x=1 -> 45 deg
        pos = x_data[frame]
        angle_rad = (pos) * (np.pi/4) 
        
        # Arm length
        L = 1.5
        end_x = L * np.sin(angle_rad)
        end_y = L * np.cos(angle_rad) - 0.5 # Shift down a bit
        
        link.set_data([0, end_x], [0, end_y])
        mass.set_center((end_x, end_y))
        
        # Color change based on effort (simulated heatmap)
        effort = abs(u_data[frame]) / 20.0
        effort = min(effort, 1.0)
        mass.set_color((effort, 0, 1-effort)) # Blue to Red
        
        return line_x, link, mass

    print(f"Rendering animation to {filename}...")
    ani = animation.FuncAnimation(fig, update, frames=len(x_data),
                                  init_func=init, blit=True, interval=30)
    
    try:
        ani.save(filename, writer='pillow', fps=30)
        print(f"✅ Success! Generated {filename}")
    except Exception as e:
        print(f"❌ Error generating GIF: {e}")
        print("Note: Ensure 'pillow' is installed (pip install pillow)")

if __name__ == "__main__":
    create_demo_gif("tools/demo.gif")
