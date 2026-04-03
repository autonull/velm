import torch
import torch.nn as nn

class MirasMemory(nn.Module):
    """Simple MLP-based associative memory stub.
    Updates state = f(state, key) by adding a nonlinear transform of key.
    """
    def __init__(self, key_dim=64, state_dim=128):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(key_dim, state_dim), nn.Tanh(), nn.Linear(state_dim, state_dim))

    def forward(self, keys):
        # keys: (B, N, key_dim) sequence of latents
        B, N, D = keys.shape
        state = torch.zeros(B, self.update[-1].out_features, device=keys.device)
        states = []
        for i in range(N):
            delta = self.update(keys[:, i, :])
            state = state + delta
            states.append(state.unsqueeze(1))
        return torch.cat(states, dim=1)  # (B, N, state_dim)
