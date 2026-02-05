import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Simplified RoPE for demo (usually computed inside attention)
        # self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention / Scaled Dot Product
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # SwiGLU logic: (gate * silu(gate)) * up -> down? 
        # Llama uses: down(silu(gate) * up)
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

class NanoLlama(nn.Module):
    """
    NanoLlama: A tiny version of Llama2/3 architecture
    for checking compiler compatibility.
    """
    def __init__(self, vocab_size=100, dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CausalSelfAttention(dim, num_heads),
                'mlp': MLP(dim, dim * 4),
                'input_layernorm': RMSNorm(dim),
                'post_attention_layernorm': RMSNorm(dim),
            }) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        # x is input_ids [Batch, SeqLen]
        h = self.embed(x)
        
        for layer in self.layers:
            # Pre-Norm
            h_norm = layer['input_layernorm'](h)
            h = h + layer['attention'](h_norm)
            
            h_norm = layer['post_attention_layernorm'](h)
            h = h + layer['mlp'](h_norm)
            
        return self.lm_head(self.norm(h))

def export_tinyllama():
    # Nano Config
    model = NanoLlama()
    model.eval()
    
    # Dummy Input: Batch=1, SeqLen=8 (Tokens)
    dummy_input = torch.randint(0, 100, (1, 8))
    
    output_path = "model_zoo/tinyllama_nano.onnx"
    
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path,
            export_params=True,
            opset_version=14, # Higher opset for SiLU/RMSNorm if standard
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={'input_ids': {1: 'seq_len'}, 'logits': {1: 'seq_len'}}
        )
        print(f"✅ Exported NanoLlama to {output_path}")
    except Exception as e:
        print(f"❌ Export Failed: {e}")

if __name__ == "__main__":
    export_tinyllama()
