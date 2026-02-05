import torch
import torch.nn as nn
import torch.onnx
import os

class ReflexNet(nn.Module):
    """
    Spinal Reflex Control Network
    Input: [Position Error, Velocity Error, Target Force] (3)
    Output: [Stiffness Gain] (1)
    """
    def __init__(self, hidden_size=64):
        super(ReflexNet, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last time step
        out = self.fc(out)
        return self.sigmoid(out)

def export_reflex_net():
    print("Exporting ReflexNet to ONNX...")
    model = ReflexNet(hidden_size=64)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 10, 3) 
    
    output_path = "model_zoo/reflex_net/reflex_net.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=['input'],
                      output_names=['stiffness'],
                      opset_version=12)
    print(f"✅ Exported to {output_path}")

if __name__ == "__main__":
    export_reflex_net()
