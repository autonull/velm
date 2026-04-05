import torch
import torch.nn as nn
from .calm import SwiGLU

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
