import torch.nn.functional as F
import torch
import torch.nn as nn
from .calm import CALMEncoder, CALMDecoder, SwiGLU
from .miras import MirasMemoryLayer
from .swa import SWALayer, RMSNorm
from .qttt import Adapter

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate, bias=False)
        self.w_up = nn.Linear(dim, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MirasBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_intermediate):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.miras = MirasMemoryLayer(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_intermediate)

    def forward(self, x, state=None):
        normed = self.norm1(x)
        miras_out, new_state = self.miras(normed, state)
        x = x + miras_out

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        return x, new_state

class SWABlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_intermediate, window_size=32):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SWALayer(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_intermediate)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out = self.attn(normed)
        x = x + attn_out

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        return x

class VelmHybrid(nn.Module):
    """Hybrid VELM model combining best of VELM Full and Lite.
    Uses CALM autoencoder block, interleaved Miras (Memora variant) and SWA blocks.
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64, state_dim=128, num_miras_layers=1, num_swa_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_dim = state_dim

        # CALM Encoder
        self.encoder = CALMEncoder(vocab_size, block_size=block_size, embed_dim=embed_dim, latent_dim=latent_dim)

        # Input compression mapping latent to state_dim (acts like backbone compress)
        self.input_compress = nn.Linear(latent_dim, state_dim, bias=False)
        self.input_ffn = SwiGLUFFN(state_dim, state_dim * 2)

        # Blocks (Samba-style interleaved)
        self.blocks = nn.ModuleList()
        self.block_types = []

        num_heads = max(1, state_dim // 32)
        m_idx, s_idx = 0, 0
        total_layers = num_miras_layers + num_swa_layers

        for i in range(total_layers):
            if i % 2 == 0 and m_idx < num_miras_layers:
                self.blocks.append(MirasBlock(state_dim, num_heads, state_dim * 2))
                self.block_types.append("miras")
                m_idx += 1
            elif s_idx < num_swa_layers:
                self.blocks.append(SWABlock(state_dim, num_heads, state_dim * 2))
                self.block_types.append("swa")
                s_idx += 1
            else:
                if m_idx < num_miras_layers:
                    self.blocks.append(MirasBlock(state_dim, num_heads, state_dim * 2))
                    self.block_types.append("miras")
                    m_idx += 1
                else:
                    self.blocks.append(SWABlock(state_dim, num_heads, state_dim * 2))
                    self.block_types.append("swa")
                    s_idx += 1

        self.final_norm = RMSNorm(state_dim)

        self.adapter = Adapter(state_dim, bottleneck=max(8, state_dim//4))

        # CALM Decoder maps state back to tokens
        self.decoder = CALMDecoder(latent_dim=state_dim, hidden=state_dim, vocab_size=vocab_size, block_size=block_size)

    def forward(self, history, use_adapter=False):
        B, L = history.shape
        K = self.block_size
        assert L % K == 0
        n_blocks = L // K
        blocks = history.view(B, n_blocks, K)

        latents = []
        for i in range(n_blocks):
            lat = self.encoder(blocks[:, i, :])
            latents.append(lat.unsqueeze(1))
        latents = torch.cat(latents, dim=1) # (B, n_blocks, latent)

        # Compress latent to state
        x = self.input_ffn(self.input_compress(latents)) # (B, n_blocks, state_dim)

        # Backbone forward
        states = []
        for i, b_type in enumerate(self.block_types):
            if b_type == "miras":
                x, _ = self.blocks[i](x)
            else:
                x = self.blocks[i](x)

        x = self.final_norm(x)

        if use_adapter:
            x = self.adapter(x)

        return x, latents

    def decode_block_state(self, state_block):
        return self.decoder(state_block)

class VelmCore(nn.Module):
    """Core VELM sequential backbone.

    Flow: continuous latents -> Miras memory -> optional SWA -> optional adapter
    This isolates the continuous sequence modeling from the specific input/output modalities.
    """
    def __init__(self, latent_dim=64, state_dim=128, use_swa=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.use_swa = use_swa

        self.latent_norm = RMSNorm(latent_dim)
        from .miras import MirasMemory
        self.memory = MirasMemory(key_dim=latent_dim, state_dim=state_dim)

        if self.use_swa:
            self.swa_norm = RMSNorm(state_dim)
            self.swa = SWALayer(dim=state_dim, num_heads=max(1, state_dim // 32), window_size=32)

        self.adapter = Adapter(state_dim, bottleneck=max(8, state_dim//4))

    def forward(self, latents, use_adapter=False):
        # latents: (B, n_blocks, latent_dim)
        norm_latents = self.latent_norm(latents)
        states = self.memory(norm_latents)        # (B, n_blocks, state_dim)

        # Hybrid configuration: SWA provides local context, Miras provides long-range state
        if self.use_swa:
            # Pre-LN Residual connection around SWA for stability
            norm_states = self.swa_norm(states)
            states = states + self.swa(norm_states)

        # optionally apply adapter to all states (qTTT)
        if use_adapter:
            states = self.adapter(states)
        return states

class VelmFull(nn.Module):
    """Full VELM-like composite model for Language Modeling (proxy).

    Flow: tokens -> CALM encoder (blocks) -> VelmCore -> decoder
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64, state_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.encoder = CALMEncoder(vocab_size, block_size=block_size, embed_dim=embed_dim, latent_dim=latent_dim)
        self.core = VelmCore(latent_dim=latent_dim, state_dim=state_dim)
        self.decoder = CALMDecoder(latent_dim=state_dim, hidden=state_dim, vocab_size=vocab_size, block_size=block_size)

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

        states = self.core(latents, use_adapter=use_adapter)
        return states, latents

    def decode_block_state(self, state_block):
        # state_block: (B, state_dim)
        return self.decoder(state_block)

    @property
    def adapter(self):
        return self.core.adapter
