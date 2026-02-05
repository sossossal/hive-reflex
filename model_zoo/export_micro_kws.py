import torch
import torch.nn as nn

class MicroKWS(nn.Module):
    """
    DS-CNN for Keyword Spotting (KWS)
    Architecture based on Hello Edge / MLPerf Tiny (DS-CNN-S)
    
    Input: [Batch, 1, 49, 10] 
           - 49 time frames (approx 1s audio)
           - 10 MFCC coefficients
    """
    def __init__(self, num_classes=12): # "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Silence", "Unknown"
        super().__init__()
        
        # Initial Conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        # DS-CNN Blocks
        self.ds_blocks = nn.Sequential(
            self._ds_layer(64, 64, stride=1),
            self._ds_layer(64, 64, stride=1),
            self._ds_layer(64, 64, stride=1),
            self._ds_layer(64, 64, stride=1),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(64, num_classes)

    def _ds_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.ds_blocks(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def export_micro_kws():
    model = MicroKWS()
    model.eval()
    
    # Dummy Input: Batch=1, 1 Channel, 49 Time Frames, 10 MFCCs
    dummy_input = torch.randn(1, 1, 49, 10)
    
    output_path = "model_zoo/micro_kws.onnx"
    
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['mfcc_input'],
            output_names=['class_logits']
        )
        print(f"✅ Exported Micro-KWS to {output_path}")
    except Exception as e:
        print(f"❌ Export Failed: {e}")

if __name__ == "__main__":
    export_micro_kws()
