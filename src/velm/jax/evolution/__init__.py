"""VELM evolution — JAX/Equinox implementation.

- gea_eggroll: Group-Evolving Agents integrated with EGGROLL populations
"""

from .gea_eggroll import (
    EvolutionTrace,
    compute_novelty,
    performance_novelty_selection,
    GroupEvolver,
    run_evolution,
)

__all__ = [
    "EvolutionTrace",
    "compute_novelty",
    "performance_novelty_selection",
    "GroupEvolver",
    "run_evolution",
]
