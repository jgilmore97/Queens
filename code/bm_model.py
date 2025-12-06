import torch
import torch.nn as nn

class _LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = torch.var(x, dim = -1, keepdim = True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

class _InitialEmbed(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, p_drop: float):
        super().__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = _LayerNorm(dim = hidden_dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.drop(self.layer_norm(nn.functional.gelu((self.proj(x)))))
        return h
    
class _Encoder(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, p_drop: float):
        super().__init__()

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.project_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.layer_norm1 = _LayerNorm(dim = hidden_dim)
        self.attn_drop = nn.Dropout(p_drop)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.ff_1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ff_2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.layer_norm2 = _LayerNorm(dim = hidden_dim)
        self.ff_drop = nn.Dropout(p_drop)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
    
    def forward(self, x):
        B, C, D = x.shape  # Batch, Cells, Hidden Dim

        qkv = self.project_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1) # Each is (B, C, D)

        q = q.view(B, C, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, C, D_head)
        k = k.view(B, C, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, C, D_head)
        v = v.view(B, C, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, C, D_head)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim ** 0.5)  # (B, n_heads, C, C)
        attn = torch.softmax(attn, dim=-1)  # (B, n_heads, C, C)
        attn = self.attn_drop(attn)
        h_attn = torch.einsum('bhij,bhjd->bhid', attn, v)  # (B, n_heads, C, D_head)
        h_attn = h_attn.transpose(1,2).view(B, C, D)  # (B, C, D)
        h_attn = self.linear_out(h_attn)
        
        h_attn = self.layer_norm1(h_attn + x)
        h_ff = self.ff_2(nn.functional.gelu((self.ff_1(h_attn))))
        h_ff = self.ff_drop(h_ff)
        h_out = self.layer_norm2(h_ff + h_attn)
        return h_out

class BenchmarkComparisonModel(nn.Module):
    """Simple non-graph based transformer benchmark model for comparison."""
    def __init__(
            self, 
            input_dim:int = 14, 
            hidden_dim:int = 128,
            p_drop:float = 0.1,
            layers:int = 4):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        
        self.embedding = _InitialEmbed(self.input_dim, self.hidden_dim, self.p_drop)

        self.layers = nn.ModuleList([
            _Encoder(self.hidden_dim, n_heads=4, p_drop=self.p_drop)
            for _ in range(layers)
        ])
        self.classifier = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        out = self.classifier(h)
        return out

