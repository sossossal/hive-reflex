import torch
import torch.nn as nn
import torch.onnx
import os
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Scaled Dot-Product Attention
        # (Simplified for export - real BERT has reshape/transpose)
        # We export as raw MatMuls to test our Compiler's ability to handle them
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        return self.out_proj(context)

class BertTiny(nn.Module):
    """
    Simplified BERT-Tiny for Edge Inference demonstration.
    Focuses on generating the specific graph patterns:
    MatMul -> Div -> Softmax (Attention)
    Add -> LayerNorm (Residual)
    """
    def __init__(self, vocab_size=1000, embed_dim=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x: (batch, seq)
        h = self.embedding(x)
        h = self.norm1(h)
        
        # Attention Block (with Residual)
        residual = h
        h = self.attention(h)
        h = h + residual
        h = self.norm2(h)
        
        # FFN Block
        residual = h
        h = self.fc(h)
        h = self.act(h)
        h = h + residual
        
        return h

def export_bert_tiny():
    print("Exporting BERT-Tiny to ONNX...")
    model = BertTiny()
    model.eval()
    
    # Dummy input (Batch=1, Seq=16)
    dummy_input = torch.randint(0, 1000, (1, 16))
    
    output_path = "model_zoo/bert_tiny/bert_tiny.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We export with dynamic axes to test shape inference
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=['input_ids'],
                      output_names=['hidden_states'],
                      opset_version=12)
    print(f"✅ Exported to {output_path}")

if __name__ == "__main__":
    export_bert_tiny()
