"""PyTorch lightweight implementation of VELM components.

This subpackage provides a simplified but fully functional PyTorch
implementation of VELM intended for fast experiments, debugging, and
visualization. It uses the same high-level API as the canonical JAX
implementation under ``velm.jax.*``, making the two drop-in replacements
for each other.

Key simplifications vs. the JAX implementation:
  - Block compression uses mean-pooling instead of the full CALM encoder
  - Memory uses a simple MLP instead of the Miras deep associative memory
  - Decoding uses a linear head instead of the energy-based generative head
"""

from .calm import CALMEncoder, CALMDecoder
from .miras import MirasMemoryLayer as MirasMemory # keep alias for compatibility if needed elsewhere
from .cib import CIBLoss
from .qttt import Adapter
from .eggroll_stub import Eggroll
from .model import VelmHybrid, VelmFull

__all__ = [
    "CALMEncoder",
    "CALMDecoder",
    "MirasMemory",
    "CIBLoss",
    "Adapter",
    "Eggroll",
    "VelmHybrid",
    "VelmFull",
]
