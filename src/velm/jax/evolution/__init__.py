"""VELM evolution — JAX/Equinox implementation.

- gea_eggroll: Group-Evolving Agents integrated with EGGROLL populations
  Includes: single-island (run_evolution) and multi-island evolution,
  diversity bonus, experience migration.
"""

from .gea_eggroll import (
    EvolutionTrace,
    compute_novelty,
    performance_novelty_selection,
    _es_gradient_from_traces,
    apply_es_update,
    GroupEvolver,
    run_evolution,
    IslandState,
    create_island,
    migrate_top_k,
    run_multi_island_evolution,
)

__all__ = [
    "EvolutionTrace",
    "compute_novelty",
    "performance_novelty_selection",
    "_es_gradient_from_traces",
    "apply_es_update",
    "GroupEvolver",
    "run_evolution",
    "IslandState",
    "create_island",
    "migrate_top_k",
    "run_multi_island_evolution",
]
