import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MobileNetV1(nn.Module):
    """
    Simplified MobileNet V1 for Hive-Reflex Model Zoo
    Targeting 96x96 grayscale input for high FPS IoT
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial Conv
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Depthwise Separable Blocks
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            # ... Reduced depth for demo purposes ...
            DepthwiseSeparableConv(256, 512, 2),
            DepthwiseSeparableConv(512, 512, 1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def export_mobilenet():
    model = MobileNetV1()
    model.eval()
    
    # Dummy Input: Batch=1, Channel=1 (Grayscale), 96x96
    dummy_input = torch.randn(1, 1, 96, 96)
    
    output_path = "model_zoo/mobilenet_v1_tiny.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"✅ Exported MobileNet V1 Tiny to {output_path}")

if __name__ == "__main__":
    export_mobilenet()
