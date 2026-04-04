"""
VELM: Vector-Evolution Language Model

A self-evolving, continuous-latent, gradient-free language model architecture.

This package exposes two implementations:
- model.VELM: JAX/Equinox research implementation (high-fidelity)
- velm.VelmFull: PyTorch proxy for fast experiments and debugging

Users should import the appropriate backend explicitly to avoid ambiguity.
"""

__version__ = "0.1.0"

# Expose canonical entrypoints for convenience
try:
    from .model import VELM as VELM_JAX
except Exception:  # pragma: no cover - optional dependency
    VELM_JAX = None

try:
    from .velm import VelmFull as VELM_TORCH
except Exception:  # pragma: no cover - optional dependency
    VELM_TORCH = None

__all__ = ["VELM_JAX", "VELM_TORCH", "__version__"]
