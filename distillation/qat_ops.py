import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Int8 Quantization.
    Forward: Round to Int8
    Backward: Pass gradients as-is (Identity)
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point):
        # Quantize: x_int = round(x / scale + zero_point)
        # We assume symmetric quantization for CIM (zero_point=0) to simplify hardware
        ctx.save_for_backward(input, scale)
        
        x_int = input / scale
        x_int = torch.round(x_int)
        x_int = torch.clamp(x_int, -128, 127)
        
        # Dequantize (Simulate what the hardware effect is in float domain)
        x_dequant = x_int * scale
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Just pass the gradient through
        # Optional: Clip gradients for inputs that were clamped (not implemented for simplicity)
        input, scale = ctx.saved_tensors
        return grad_output, None, None

class QuantizedLinear(nn.Module):
    """
    Linear Layer that simulates Int8 CIM Hardware behavior.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Dynamic Scaling Factors (Learned or Statistically Computed)
        # For simplicity, we use MinMax statistics calculated per batch
        
    def forward(self, x):
        # 1. Quantize Inputs
        # Calculate dynamic scale for input (per tensor)
        in_scale = x.abs().max() / 127.0
        x_q = FakeQuantize.apply(x, in_scale, 0)
        
        # 2. Quantize Weights
        # Calculate dynamic scale for weights (per tensor or per channel)
        w_scale = self.weight.abs().max() / 127.0
        w_q = FakeQuantize.apply(self.weight, w_scale, 0)
        
        # 3. Compute (Float computation simulating Int8 dequantized result)
        out = F.linear(x_q, w_q, self.bias)
        
        return out

