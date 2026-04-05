import torch
import torch.nn as nn
import torch.nn.functional as F

# Use PyTorch's native RMSNorm for performance
RMSNorm = nn.RMSNorm

class SWALayer(nn.Module):
    """Sliding Window Attention (SWA) layer to complement MirasMemory.
    Provides precise local context retrieval with a fixed window size.
    Features independent q, k, v projections to support qTTT query adaptation.
    """
    def __init__(self, dim=128, num_heads=4, window_size=32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.pre_norm = RMSNorm(dim)

        # RoPE parameters (fixed base)
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)),
            persistent=False
        )

    def apply_rope(self, x):
        # x: (B, num_heads, N, head_dim)
        B, n_h, N, d_h = x.shape
        t = torch.arange(N, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # (N, d_h)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        x_rot1 = x[..., : d_h // 2]
        x_rot2 = x[..., d_h // 2 :]
        x_rotated = torch.cat((-x_rot2, x_rot1), dim=-1)

        return (x * cos) + (x_rotated * sin)

    def forward(self, x):
        x = self.pre_norm(x)
        # x: (B, N, D)
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N, head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self.apply_rope(q)
        k = self.apply_rope(k)

        # Calculate scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, num_heads, N, N)

        # Apply sliding window causal mask
        # Create a boolean mask where True means "mask out"
        mask = torch.ones((N, N), dtype=torch.bool, device=x.device)
        mask = torch.triu(mask, diagonal=1) # Causal mask

        # Apply window (mask out elements too far in the past)
        if self.window_size > 0:
            window_mask = torch.tril(torch.ones((N, N), dtype=torch.bool, device=x.device), diagonal=-self.window_size)
            mask = mask | window_mask

        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v) # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.o_proj(out)
