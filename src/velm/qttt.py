import torch
import torch.nn as nn

class Adapter(nn.Module):
    """Small adapter module used for qTTT (few-step adaptation)."""
    def __init__(self, dim, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(torch.relu(self.down(x)))
