"""VELM: Vector-Evolution Language Model.

A self-evolving, continuous-latent, gradient-free language model architecture.

Two implementations are provided:
  - velm.jax.*:       JAX/Equinox research implementation (canonical, high-fidelity)
  - velm.lite.*:      PyTorch implementation (lightweight, for fast experiments and debugging)

Quick start:
  from velm.jax.model import VELM, CONFIGS          # research-scale model
  from velm.lite import VelmFull                     # fast iteration model
  from velm.jax.training import eggroll_step         # gradient-free optimizer
  from velm.jax.inference import apply_qttt          # test-time adaptation
  from velm.jax.evolution import GroupEvolver        # self-improvement loop
"""

__version__ = "0.1.0"

# JAX research implementation (canonical)
try:
    from .jax.model import (
        VELM,
        CALMAutoencoder,
        VELMBackbone,
        EnergyHead,
        energy_score,
        CONFIGS,
        QWEN35_VOCAB_SIZE,
    )
except Exception:
    pass

# PyTorch lightweight implementation
try:
    from .lite import (
        CALMEncoder,
        CALMDecoder,
        MirasMemory,
        CIBLoss,
        Adapter,
        Eggroll,
        VelmFull,
    )
except Exception:
    pass

__all__ = [
    # JAX
    "VELM",
    "CALMAutoencoder",
    "VELMBackbone",
    "EnergyHead",
    "energy_score",
    "CONFIGS",
    "QWEN35_VOCAB_SIZE",
    # Lite
    "CALMEncoder",
    "CALMDecoder",
    "MirasMemory",
    "CIBLoss",
    "Adapter",
    "Eggroll",
    "VelmFull",
]
