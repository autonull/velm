import torch
import torch.nn as nn
import torch.nn.functional as F
from .calm import SwiGLU
from .swa import RMSNorm

class DepthwiseConv1d(nn.Module):
    """1D depthwise-separable convolution."""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1)

    def forward(self, x):
        # x: (B, N, dim)
        x = x.transpose(1, 2) # (B, dim, N)
        out = self.conv(x) # (B, dim, N + padding)
        out = out[..., :-(self.kernel_size - 1)] # truncate to maintain causality
        return out.transpose(1, 2) # (B, N, dim)

class MirasMemoryLayer(nn.Module):
    """Miras layer: deep associative memory with MLP structure (Memora variant).
    Ported from VELM Full (JAX) to PyTorch.
    """
    def __init__(self, dim=128, num_heads=4, low_rank_dim=32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.low_rank_dim = low_rank_dim

        # QKV projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Depthwise convs
        self.conv_q = DepthwiseConv1d(dim, 4)
        self.conv_k = DepthwiseConv1d(dim, 4)
        self.conv_v = DepthwiseConv1d(dim, 4)

        # Norms
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Low rank projections
        self.eta_down = nn.Linear(dim, low_rank_dim, bias=False)
        self.eta_up = nn.Linear(low_rank_dim, dim, bias=False)

        self.alpha_down = nn.Linear(dim, low_rank_dim, bias=False)
        self.alpha_up = nn.Linear(low_rank_dim, dim, bias=False)

        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def _compute_eta(self, x):
        return torch.sigmoid(self.eta_up(self.eta_down(x)))

    def _compute_alpha(self, x):
        return torch.sigmoid(self.alpha_up(self.alpha_down(x)))

    def forward(self, x, state=None):
        # x: (B, N, dim)
        B, N, D = x.shape

        q = self.conv_q(self.wq(x))
        k = self.conv_k(self.wk(x))
        v = self.conv_v(self.wv(x))

        # Norm heads
        q = q.view(B, N, self.num_heads, self.head_dim)
        q = self.q_norm(q).view(B, N, D)
        k = k.view(B, N, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, N, D)

        if state is None:
            state = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(N):
            k_t = k[:, t, :] # (B, dim)
            v_t = v[:, t, :]
            x_t = x[:, t, :]

            eta = self._compute_eta(x_t) # (B, dim)
            alpha = self._compute_alpha(x_t)

            # state: (B, dim, dim) - batched matrix multiply
            # memory_pred = state @ k_t
            memory_pred = torch.bmm(state, k_t.unsqueeze(2)).squeeze(2) # (B, dim)

            error = memory_pred - v_t # (B, dim)
            # grad = outer(error, k_t) -> (B, dim, 1) @ (B, 1, dim) -> (B, dim, dim)
            grad = torch.bmm(error.unsqueeze(2), k_t.unsqueeze(1))

            new_state = alpha.unsqueeze(2) * state - eta.unsqueeze(2) * grad
            new_state = torch.clamp(new_state, -10.0, 10.0)

            # readout: new_state @ k_t
            out_t = torch.bmm(new_state, k_t.unsqueeze(2)).squeeze(2)
            outputs.append(out_t.unsqueeze(1))
            state = new_state

        outputs = torch.cat(outputs, dim=1) # (B, N, dim)

        gate = F.silu(self.gate_proj(x))
        gated = outputs * gate
        projected = self.wo(gated)

        return projected, state

class MirasMemory(nn.Module):
    """Deep MLP-based associative memory stub with dynamic retention gating.
    Updates state using a learned retention mechanism:
    retention = sigmoid(W_ret * [state, key])
    state = retention * state + (1 - retention) * f(state, key)
    """
    def __init__(self, key_dim=64, state_dim=128):
        super().__init__()
        # State + Key -> State (delta)
        self.update = nn.Sequential(
            nn.LayerNorm(state_dim + key_dim),
            SwiGLU(state_dim + key_dim, state_dim * 2, state_dim)
        )
        # Retention Gate: State + Key -> Retention vector
        self.retention_gate = nn.Sequential(
            nn.Linear(state_dim + key_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.Sigmoid()
        )

    def forward(self, keys):
        # keys: (B, N, key_dim) sequence of latents
        B, N, D = keys.shape
        # Initialize state explicitly
        # SwiGLU does not have out_features exposed directly like Linear, so we access w3.out_features
        state = torch.zeros(B, self.update[-1].w3.out_features, device=keys.device)
        states = []
        for i in range(N):
            # Recurrent update based on previous state and current key
            cat_in = torch.cat([state, keys[:, i, :]], dim=-1)
            delta = self.update(cat_in)
            retention = self.retention_gate(cat_in)

            # Gated update: balance retaining past state vs adding new information
            state = retention * state + (1 - retention) * delta
            states.append(state.unsqueeze(1))
        return torch.cat(states, dim=1)  # (B, N, state_dim)
