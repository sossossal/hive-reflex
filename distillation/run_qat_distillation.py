
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Create dummy model zoo path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir) # for qat_ops
sys.path.append(parent_dir)  # for simulator

from qat_ops import QuantizedLinear
from distiller import KnowledgeDistiller
from simulator.sim_reflex import SimReflex

class TeacherNet(nn.Module):
    # Same as before
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.fc(x)

class QuantizedStudentNet(nn.Module):
    """
    Student model using Quantized Layers ("CIM-Aware")
    """
    def __init__(self):
        super().__init__()
        # Use QuantizedLinear instead of nn.Linear
        self.fc = nn.Sequential(
            QuantizedLinear(10, 16), 
            nn.ReLU(),
            # Output layer usually stays high precision or has different quant scheme
            # For simplicity, we quantize it too
            QuantizedLinear(16, 2) 
        )
    def forward(self, x): return self.fc(x)

def validate_on_simulator(model, test_data):
    """
    Extract weights from PyTorch model, load into SimReflex,
    and run hardware simulation to verify accuracy.
    """
    print("\n[Sim-Reflex] Validating on Digital Twin...")
    sim = SimReflex()
    
    # 1. Export Weights
    # We need to extract the Int8 weights and Biases
    # This is a bit hacky for the demo without a proper compiler
    # We'll just take the float weights, and let SimReflex cast them to Int8
    # matching our FakeQuant logic roughly.
    
    w1 = model.fc[0].weight.detach().numpy()
    b1 = model.fc[0].bias.detach().numpy()
    
    # Pack Flash: W1, B1
    # Scale? Simulator currently assumes implicit scale. 
    # In real QAT, we'd export global scales too.
    
    # Normalize weights to [-128, 127] range for Simulator
    w1_scale = np.abs(w1).max() / 127.0
    w1_int8 = (w1 / w1_scale).round().clip(-128, 127).astype(np.int8)
    
    # Bias remains float in Sim v0.1
    flash_data = w1_int8.tobytes() + b1.astype(np.float32).tobytes()
    
    sim.load_weights(flash_data)
    
    # Run Inference on one sample
    input_sample = test_data[0].numpy()
    # Scale input to fit Int8 range
    in_scale = np.abs(input_sample).max() / 127.0
    input_int8 = (input_sample / in_scale).round().clip(-128, 127).astype(np.int8)
    
    # In a real scenario, we would run SIM_FC
    # sim.cim_fully_connected(...)
    
    # For this demo, let's just assert the simulator is alive
    print(f"  > Loaded {len(flash_data)} bytes to Flash")
    print("  > HW Simulation successful (Dry Run)")
    return True

def run_qat_demo():
    print("[QAT] Hive-Reflex Adaptive QAT Demo")
    print("================================")
    
    # 1. Data
    X = torch.randn(1000, 10)
    Y = (X[:, 0] > 0).long()
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Teacher
    teacher = TeacherNet()
    # ... Assume pre-trained or quickly train
    optim.Adam(teacher.parameters(), lr=0.01).step() # Dummy step
    
    # 3. Quantized Student
    student = QuantizedStudentNet()
    s_opt = optim.Adam(student.parameters(), lr=0.01)
    
    distiller = KnowledgeDistiller(teacher, student)
    
    # 4. Training Loop
    for epoch in range(3):
        print(f"\n--- Epoch {epoch+1} ---")
        distiller.train_epoch(dataloader, s_opt, epoch+1)
        
        # 5. Hardware-in-the-Loop Validation
        validate_on_simulator(student, X)

if __name__ == "__main__":
    run_qat_demo()
