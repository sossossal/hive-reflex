import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistiller:
    """
    Generic Knowledge Distillation Trainer.
    Transfers knowledge from a heavy 'Teacher' model to a lightweight 'Student' model.
    """
    def __init__(self, teacher, student, device='cpu'):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device
        self.teacher.eval() # Teacher is always fixed
        self.student.train()
        
    def distillation_loss(self, student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
        """
        Compute Distillation Loss = Alpha * SoftTargetLoss + (1-Alpha) * HardTargetLoss
        
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model
            labels: Ground truth labels
            T: Temperature for softening probability distributions
            alpha: Weighting factor (0.0 = only hard labels, 1.0 = only teacher knowledge)
        """
        # 1. Soft Target Loss (KL Divergence)
        # "Match the teacher's probability distribution"
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(student_logits / T, dim=1)
        
        # Note: KLDivLoss expects log_prob as input and probability as target
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T**2)
        
        # 2. Hard Target Loss (Cross Entropy)
        # "Don't forget the correct answer"
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Weighted Sum
        return alpha * soft_loss + (1 - alpha) * hard_loss

    def train_epoch(self, dataloader, optimizer, epoch_idx):
        self.student.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward Student
            student_out = self.student(data)
            
            # Forward Teacher (No Grad)
            with torch.no_grad():
                teacher_out = self.teacher(data)
            
            # Compute Loss
            loss = self.distillation_loss(student_out, teacher_out, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy (Student)
            pred = student_out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(dataloader)
        acc = 100. * correct / total
        print(f"Epoch {epoch_idx}: Loss={avg_loss:.4f}, Student Acc={acc:.2f}%")
        return acc
