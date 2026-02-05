import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Create dummy model zoo path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from distiller import KnowledgeDistiller

# 1. Define Models
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.fc(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16), # Smaller hidden dim
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x): return self.fc(x)

def run_demo():
    print("[Distiller] Hive-Reflex Distillation Demo")
    print("================================")
    
    # 2. Setup Data (Synthetic)
    # Teacher learns a simple rule: x[0] > 0
    X = torch.randn(1000, 10)
    Y = (X[:, 0] > 0).long()
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Pre-train Teacher (Or load pretrained)
    teacher = TeacherNet()
    t_opt = optim.Adam(teacher.parameters(), lr=0.01)
    
    print("Step 1: Training Teacher...")
    for epoch in range(5):
        for data, target in dataloader:
            t_opt.zero_grad()
            loss = nn.functional.cross_entropy(teacher(data), target)
            loss.backward()
            t_opt.step()
    print("Teacher Trained.\n")

    # 4. Distill to Student
    student = StudentNet()
    s_opt = optim.Adam(student.parameters(), lr=0.01)
    
    distiller = KnowledgeDistiller(teacher, student)
    
    print("Step 2: Distilling Knowledge...")
    for epoch in range(5):
        acc = distiller.train_epoch(dataloader, s_opt, epoch+1)
        
    print(f"\nDistillation Complete. Final Accuracy: {acc:.2f}%")
    
    if acc > 90:
        print("Success: Student learned the concept!")
    else:
        print("Warning: Accuracy lower than expected.")

if __name__ == "__main__":
    run_demo()
