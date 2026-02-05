import torch
import torch.nn as nn

class GestureNet(nn.Module):
    """
    1D-CNN for 6-axis IMU Gesture Recognition
    Input: [Batch, 6, 32] (AccelX,Y,Z, GyroX,Y,Z over 32 ticks)
    """
    def __init__(self, num_classes=4): # Idle, Left, Right, Circle
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv 1: Extract temporal features
            nn.Conv1d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 16
            
            # Conv 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 8
            
            # Conv 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def export_gesture_net():
    model = GestureNet()
    model.eval()
    
    # Dummy Input: Batch=1, 6 Sensors, 32 TimeSteps
    dummy_input = torch.randn(1, 6, 32)
    
    output_path = "model_zoo/gesture_net.onnx"
    
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        print(f"✅ Exported GestureNet to {output_path}")
    except Exception as e:
        print(f"❌ Export Failed (likely missing libraries): {e}")

if __name__ == "__main__":
    export_gesture_net()
