import torch
import torch.nn as nn
from .calm import CALMEncoder, CALMDecoder
from .miras import MirasMemory
from .qttt import Adapter

class VelmFull(nn.Module):
    """Full VELM-like composite model (proxy).

    Flow: tokens -> CALM encoder (blocks) -> Miras memory -> optional adapter -> decoder
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64, state_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.encoder = CALMEncoder(vocab_size, embed_dim=embed_dim, latent_dim=latent_dim)
        self.memory = MirasMemory(key_dim=latent_dim, state_dim=state_dim)
        self.adapter = Adapter(state_dim, bottleneck=max(8, state_dim//4))
        self.decoder = CALMDecoder(latent_dim=state_dim, hidden=state_dim, vocab_size=vocab_size)

    def forward(self, history, use_adapter=False):
        # history: (B, L) where L divisible by block_size
        B, L = history.shape
        K = self.block_size
        assert L % K == 0
        n_blocks = L // K
        blocks = history.view(B, n_blocks, K)
        # encode each block
        latents = []
        for i in range(n_blocks):
            lat = self.encoder(blocks[:, i, :])
            latents.append(lat.unsqueeze(1))
        latents = torch.cat(latents, dim=1)  # (B, n_blocks, latent)
        states = self.memory(latents)        # (B, n_blocks, state_dim)
        # optionally apply adapter to all states (qTTT)
        if use_adapter:
            states = self.adapter(states)
        return states, latents

    def decode_block_state(self, state_block):
        # state_block: (B, state_dim)
        return self.decoder(state_block)
