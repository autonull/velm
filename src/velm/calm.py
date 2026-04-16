import torch
import torch.nn as nn
import torch.nn.functional as F

class CALMEncoder(nn.Module):
    """Compress K tokens into one continuous vector (simple proxy).
    Implementation: embed tokens, flatten to preserve order, then MLP projection.
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * block_size, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, tokens):
        # tokens: (B, K)
        B, K = tokens.shape
        emb = self.embed(tokens)  # (B, K, E)
        flat = emb.view(B, K * emb.size(-1)) # (B, K*E)
        lat = self.proj(flat)     # (B, latent)
        return lat

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class CALMDecoder(nn.Module):
    """Energy-based style decoder stub: map latent to K tokens logits via MLP with residual blocks."""
    def __init__(self, latent_dim=64, hidden=128, vocab_size=100, block_size=4, num_blocks=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.blocks = nn.Sequential(*[ResidualBlock(latent_dim, hidden) for _ in range(num_blocks)])
        self.head = nn.Linear(latent_dim, vocab_size * block_size)

    def forward(self, latent):
        # latent: (..., latent_dim)
        # out: (..., block_size, vocab_size)
        h = self.blocks(latent)
        out = self.head(h)
        shape = list(out.shape[:-1]) + [self.block_size, self.vocab_size]
        return out.view(*shape)
