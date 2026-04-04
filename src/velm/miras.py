import torch
import torch.nn as nn

class MirasMemory(nn.Module):
    """Simple MLP-based associative memory stub.
    Updates state = f(state, key) by adding a nonlinear transform.
    """
    def __init__(self, key_dim=64, state_dim=128):
        super().__init__()
        # State + Key -> State
        self.update = nn.Sequential(
            nn.Linear(state_dim + key_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim)
        )

    def forward(self, keys):
        # keys: (B, N, key_dim) sequence of latents
        B, N, D = keys.shape
        # Initialize state explicitly
        state = torch.zeros(B, self.update[-1].out_features, device=keys.device)
        states = []
        for i in range(N):
            # Recurrent update based on previous state and current key
            cat_in = torch.cat([state, keys[:, i, :]], dim=-1)
            delta = self.update(cat_in)
            state = state + delta
            states.append(state.unsqueeze(1))
        return torch.cat(states, dim=1)  # (B, N, state_dim)
