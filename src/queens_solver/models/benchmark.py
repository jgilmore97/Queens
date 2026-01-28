import torch
import torch.nn as nn


class _LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class _InitialEmbed(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = _LayerNorm(dim=hidden_dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.drop(self.layer_norm(nn.functional.gelu(self.proj(x))))
        return h


class _Encoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.project_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.layer_norm1 = _LayerNorm(dim=hidden_dim)
        self.attn_drop = nn.Dropout(p_drop)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.ff_1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ff_2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.layer_norm2 = _LayerNorm(dim=hidden_dim)
        self.ff_drop = nn.Dropout(p_drop)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

    def forward(self, x):
        B, C, D = x.shape

        qkv = self.project_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        h_attn = torch.einsum('bhij,bhjd->bhid', attn, v)
        h_attn = h_attn.transpose(1, 2).reshape(B, C, D)
        h_attn = self.linear_out(h_attn)

        h_attn = self.layer_norm1(h_attn + x)
        h_ff = self.ff_2(nn.functional.gelu(self.ff_1(h_attn)))
        h_ff = self.ff_drop(h_ff)
        h_out = self.layer_norm2(h_ff + h_attn)
        return h_out


class BenchmarkHRM(nn.Module):
    """HRM-inspired non-graph transformer with hierarchical L/H iteration structure."""
    def __init__(
            self,
            input_dim: int = 14,
            hidden_dim: int = 128,
            p_drop: float = 0.1,
            n_heads: int = 4,
            n_cycles: int = 3,
            t_micro: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.t_micro = t_micro

        self.embedding = _InitialEmbed(input_dim=self.input_dim, hidden_dim=self.hidden_dim, p_drop=self.p_drop)
        self.L_block = _Encoder(hidden_dim=self.hidden_dim, n_heads=self.n_heads, p_drop=self.p_drop)
        self.H_block = _Encoder(hidden_dim=self.hidden_dim, n_heads=self.n_heads, p_drop=self.p_drop)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, 1),
        )

        self.z_L_init = nn.Parameter(torch.randn(121, hidden_dim) * 0.02)
        self.z_H_init = nn.Parameter(torch.randn(121, hidden_dim) * 0.02)

    def forward(self, x):
        z_L = self.z_L_init.clone()
        z_H = self.z_H_init.clone()

        x_embedded = self.embedding(x)

        for cycle_idx in range(self.n_cycles):
            for step in range(self.t_micro):
                z_L = self.L_block(x_embedded + z_L + z_H)
            z_H = self.H_block(z_L + z_H)

        out = self.readout(z_H)
        return out


class BenchmarkSequential(nn.Module):
    """Sequential non-graph transformer baseline for comparison."""
    def __init__(
            self,
            input_dim: int = 14,
            hidden_dim: int = 128,
            p_drop: float = 0.1,
            n_heads: int = 4,
            layers: int = 6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop

        self.embedding = _InitialEmbed(self.input_dim, self.hidden_dim, self.p_drop)
        self.layers = nn.ModuleList([
            _Encoder(self.hidden_dim, n_heads=n_heads, p_drop=self.p_drop)
            for _ in range(layers)
        ])
        self.classifier = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = h + layer(h)
        out = self.classifier(h)
        return out


BENCHMARK_MODELS = {
    'hrm': BenchmarkHRM,
    'sequential': BenchmarkSequential,
}


def get_benchmark_model(model_type: str, **kwargs):
    if model_type not in BENCHMARK_MODELS:
        raise ValueError(f"Unknown benchmark model type: {model_type}. Available: {list(BENCHMARK_MODELS.keys())}")
    return BENCHMARK_MODELS[model_type](**kwargs)