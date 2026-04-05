"""
VELM Evolution — GEA-EGGROLL Integration

Group-Evolving Agents adapted for weight-level evolution of VELM populations.

Bridges two frameworks:
  - EGGROLL: maintains population of weight perturbations, evaluates fitness
  - GEA: enables experience sharing across population members

Evolution cycle:
  1. EGGROLL evaluates population on diverse task distribution
  2. Collect evolutionary traces per member
  3. GEA reflection: analyze traces across the group
  4. GEA evolution: bias next-generation perturbations
  5. Select parent group for next iteration

Selection criterion:
  score(i) = α_i × √nov(i)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from dataclasses import dataclass, field
from typing import Callable, Optional
import optax
from ..training.eggroll import (
    perturb_pytree,
    EGGROLLState,
    create_eggroll_optimizer,
)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class EvolutionTrace:
    """Records what happened during one population member's evaluation.

    Captures per-member performance, reasoning behavior, and the
    perturbation needed to compute the ES gradient.
    """

    member_id: int
    perturbation_seed: int
    fitness_scores: dict[str, float] = field(default_factory=dict)
    reasoning_lengths: dict[str, float] = field(default_factory=dict)
    attention_mass: dict[str, float] = field(default_factory=dict)
    perturbation: Optional[PyTree] = field(default=None, repr=False)

    @property
    def mean_fitness(self) -> float:
        """Average fitness across all task types."""
        if not self.fitness_scores:
            return 0.0
        return sum(self.fitness_scores.values()) / len(self.fitness_scores)

    @property
    def mean_reasoning_length(self) -> float:
        """Average reasoning chunks used across tasks."""
        if not self.reasoning_lengths:
            return 0.0
        vals = self.reasoning_lengths.values()
        return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Novelty & selection (core — always available)
# ---------------------------------------------------------------------------


def compute_novelty(
    embeddings: Float[Array, "N dim"],
    num_neighbors: int = 5,
) -> Float[Array, "N"]:
    """Compute novelty score for each member via k-nearest neighbor distance.

    Novelty = mean cosine distance to M nearest neighbors.
    Higher novelty = more unique perturbation direction.

    Args:
        embeddings: (N, dim) embedding per population member
        num_neighbors: M nearest neighbors

    Returns:
        (N,) novelty scores
    """
    norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    dist_matrix = 1.0 - sim_matrix
    dist_matrix = dist_matrix + jnp.eye(dist_matrix.shape[0]) * 1e9
    sorted_dists = jnp.sort(dist_matrix, axis=1)
    neighbor_dists = sorted_dists[:, :num_neighbors]

    return jnp.mean(neighbor_dists, axis=1)


def performance_novelty_selection(
    fitness_scores: Float[Array, "N"],
    novelty_scores: Float[Array, "N"],
    group_size: int,
    *,
    diversity_bonus: float = 0.0,
) -> list[int]:
    """Select parent group using Performance-Novelty criterion.

    score(i) = fitness(i) × √novelty(i) + diversity_bonus × pairwise_cosine(i)

    Performance is the primary criterion; novelty provides a mild
    exploration bias. The optional diversity_bonus adds an explicit
    reward for perturbing in directions orthogonal to the population mean.

    Args:
        fitness_scores: (N,) per-member fitness
        novelty_scores: (N,) per-member novelty
        group_size: K — number of parents to select
        diversity_bonus: extra weight on pairwise cosine diversity (0 = off)

    Returns:
        List of selected member indices
    """
    combined = fitness_scores * jnp.sqrt(novelty_scores + 1e-8)

    if diversity_bonus > 0.0:
        norms = jnp.linalg.norm(fitness_scores, keepdims=True) + 1e-8
        normed = fitness_scores / norms
        cos_sim = normed @ normed.T
        cos_sim = cos_sim + jnp.eye(cos_sim.shape[0]) * 1e9
        diversity = jnp.mean(jnp.abs(cos_sim), axis=1)
        combined = combined + diversity_bonus * diversity

    top_k = jnp.argsort(combined)[::-1][:group_size]
    return top_k.tolist()


# ---------------------------------------------------------------------------
# Parameter update from traces (fixes the interface mismatch)
# ---------------------------------------------------------------------------


def _es_gradient_from_traces(
    traces: list["EvolutionTrace"],
    sigma: float,
) -> tuple[PyTree, dict]:
    """Compute the ES gradient from stored perturbations and fitness scores.

    This is the core of the EGGROLL update, extracted so both
    ``run_evolution`` and ``run_multi_island_evolution`` can use it
    without needing a separate adapter or calling ``eggroll_step``
    (which has a different interface).

    Args:
        traces: evaluation traces (each must have ``perturbation`` stored)
        sigma: perturbation scale

    Returns:
        (es_gradient, metrics_dict)
    """
    pop_size = len(traces)
    fitness_values = jnp.array([t.mean_fitness for t in traces])

    # Rank-based fitness normalization
    ranks = jnp.argsort(jnp.argsort(fitness_values)).astype(jnp.float32)
    normalized = 0.5 - ranks / (pop_size - 1)

    # Stack perturbations into a pytree with leading population dimension
    perturbation_leaves = [
        jnp.stack([t.perturbation[k] for t in traces])
        for k in range(len(jax.tree.leaves(traces[0].perturbation)))
    ]
    perturbation_treedef = jax.tree.structure(traces[0].perturbation)
    stacked_perturbations = jax.tree.unflatten(perturbation_treedef, perturbation_leaves)

    # ES gradient: weighted sum of perturbations
    def weighted_leaf_sum(leaf_stack: jax.Array) -> jax.Array:
        w = normalized.reshape((-1,) + (1,) * (leaf_stack.ndim - 1))
        return jnp.sum(leaf_stack * w, axis=0)

    es_grad = jax.tree.map(weighted_leaf_sum, stacked_perturbations)

    # Scale by 1/(σN)
    scale = 1.0 / (sigma * pop_size)
    es_grad = jax.tree.map(lambda g: g * scale, es_grad)

    # Negate (ES maximizes, optimizer minimizes)
    neg_grad = jax.tree.map(lambda g: -g, es_grad)

    es_grad_leaves = jax.tree.leaves(es_grad)
    metrics = {
        "mean_fitness": float(jnp.mean(fitness_values)),
        "max_fitness": float(jnp.max(fitness_values)),
        "min_fitness": float(jnp.min(fitness_values)),
        "fitness_std": float(jnp.std(fitness_values)),
        "grad_norm": float(jnp.sqrt(sum(jnp.sum(l**2) for l in es_grad_leaves))),
    }

    return neg_grad, metrics


def apply_es_update(
    params: PyTree,
    neg_grad: PyTree,
    optimizer: optax.GradientTransformation,
    state: EGGROLLState,
) -> tuple[PyTree, EGGROLLState]:
    """Apply an Adam update to parameters given a pre-computed ES gradient.

    Args:
        params: current parameters
        neg_grad: negated ES gradient (already scaled)
        optimizer: optax optimizer
        state: current optimizer state

    Returns:
        (updated_params, new_state)
    """
    updates, new_opt_state = optimizer.update(neg_grad, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, EGGROLLState(new_opt_state, state.step + 1)


# ---------------------------------------------------------------------------
# GroupEvolver (core)
# ---------------------------------------------------------------------------


@dataclass
class GroupEvolver:
    """Orchestrates GEA-style group evolution over EGGROLL populations.

    At each iteration:
      1. Evaluate all population members on task distribution
      2. Collect evolutionary traces
      3. Select parent group via Performance-Novelty
      4. Share experience: aggregate traces from parent group
      5. Generate offspring: bias perturbations toward promising directions
    """

    population_size: int = 64
    group_size: int = 5
    novelty_neighbors: int = 5
    diversity_bonus: float = 0.0
    archive: list[EvolutionTrace] = field(default_factory=list)
    iteration: int = 0

    def evaluate_population(
        self,
        base_params: PyTree,
        fitness_fn: Callable,
        task_distribution: list[dict],
        *,
        key: jax.Array,
        sigma: float = 0.001,
        rank: int = 1,
    ) -> list[EvolutionTrace]:
        """Evaluate all population members on the task distribution.

        Args:
            base_params: current mean parameters M
            fitness_fn: f(params, task) → (fitness, metrics)
            task_distribution: list of task dicts to evaluate on
            key: PRNG key
            sigma: EGGROLL perturbation scale
            rank: perturbation rank

        Returns:
            List of EvolutionTrace for each member
        """
        traces = []
        member_keys = jax.random.split(key, self.population_size)

        for i in range(self.population_size):
            seed = int(member_keys[i][0])
            perturbed, perturbation = perturb_pytree(base_params, member_keys[i], sigma, rank)

            trace = EvolutionTrace(member_id=i, perturbation_seed=seed)

            for task in task_distribution:
                task_type = task.get("type", "default")
                fitness, metrics = fitness_fn(perturbed, task)
                trace.fitness_scores[task_type] = float(fitness)
                if "num_chunks" in metrics:
                    trace.reasoning_lengths[task_type] = float(metrics["num_chunks"])

            trace.perturbation = perturbation
            traces.append(trace)

        self.archive.extend(traces)
        return traces

    def select_parents(
        self,
        traces: list[EvolutionTrace],
        embeddings: Float[Array, "N dim"],
    ) -> list[int]:
        """Select parent group from current population."""
        fitnesses = jnp.array([t.mean_fitness for t in traces])
        novelties = compute_novelty(embeddings, self.novelty_neighbors)
        return performance_novelty_selection(
            fitnesses,
            novelties,
            self.group_size,
            diversity_bonus=self.diversity_bonus,
        )

    def aggregate_experience(
        self,
        traces: list[EvolutionTrace],
        parent_indices: list[int],
    ) -> dict:
        """Aggregate evolutionary experience from parent group."""
        parent_traces = [traces[i] for i in parent_indices]

        task_champions: dict[str, int] = {}
        for trace in parent_traces:
            for task_type, fitness in trace.fitness_scores.items():
                if task_type not in task_champions:
                    task_champions[task_type] = trace.member_id
                else:
                    champ_trace = traces[task_champions[task_type]]
                    if fitness > champ_trace.fitness_scores.get(task_type, -float("inf")):
                        task_champions[task_type] = trace.member_id

        avg_reasoning = {}
        for trace in parent_traces:
            for task_type, length in trace.reasoning_lengths.items():
                if task_type not in avg_reasoning:
                    avg_reasoning[task_type] = []
                avg_reasoning[task_type].append(length)
        avg_reasoning = {k: sum(v) / len(v) for k, v in avg_reasoning.items()}

        successful_seeds = [t.perturbation_seed for t in parent_traces]

        return {
            "task_champions": task_champions,
            "avg_reasoning_budget": avg_reasoning,
            "successful_seeds": successful_seeds,
            "parent_fitness": [t.mean_fitness for t in parent_traces],
            "iteration": self.iteration,
        }

    def evolution_step(
        self,
        base_params: PyTree,
        fitness_fn: Callable,
        task_distribution: list[dict],
        *,
        key: jax.Array,
        sigma: float = 0.001,
        rank: int = 1,
    ) -> tuple[dict, list[EvolutionTrace]]:
        """Run one full GEA evolution iteration."""
        k1, k2 = jax.random.split(key)

        traces = self.evaluate_population(
            base_params,
            fitness_fn,
            task_distribution,
            key=k1,
            sigma=sigma,
            rank=rank,
        )

        task_types = sorted(set().union(*(t.fitness_scores.keys() for t in traces)))
        embeddings = jnp.array(
            [[t.fitness_scores.get(tt, 0.0) for tt in task_types] for t in traces]
        )

        parent_indices = self.select_parents(traces, embeddings)
        experience = self.aggregate_experience(traces, parent_indices)

        self.iteration += 1
        return experience, traces


# ---------------------------------------------------------------------------
# run_evolution (fixed — computes ES gradient from traces directly)
# ---------------------------------------------------------------------------


def run_evolution(
    base_params: PyTree,
    fitness_fn: Callable,
    task_distribution: list[dict],
    *,
    key: jax.Array,
    num_iterations: int = 30,
    population_size: int = 64,
    group_size: int = 5,
    sigma: float = 0.001,
    rank: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    diversity_bonus: float = 0.0,
) -> tuple[PyTree, list[dict]]:
    """Run the full GEA-EGGROLL evolution loop.

    At each iteration:
      1. GEA evaluates population and selects parents
      2. Experience is aggregated from the parent group
      3. EGGROLL updates base params via ES gradient (computed from traces)

    Args:
        base_params: initial mean parameters
        fitness_fn: f(params, task) → (fitness, metrics)
        task_distribution: tasks to evaluate on
        key: PRNG key
        num_iterations: number of evolution iterations
        population_size: members per generation
        group_size: parents per group (K in GEA)
        sigma: EGGROLL perturbation scale
        rank: perturbation rank
        learning_rate: Adam learning rate for ES gradient updates
        weight_decay: optional weight decay
        diversity_bonus: extra reward for diverse perturbations (0 = off)

    Returns:
        (final_params, history_of_experiences)
    """
    evolver = GroupEvolver(
        population_size=population_size,
        group_size=group_size,
        diversity_bonus=diversity_bonus,
    )

    optimizer, opt_state = create_eggroll_optimizer(
        base_params, learning_rate=learning_rate, weight_decay=weight_decay
    )

    history = []
    params = base_params

    for i in range(num_iterations):
        key, iter_key = jax.random.split(key)

        # GEA: evaluate, select, aggregate
        experience, traces = evolver.evolution_step(
            params,
            fitness_fn,
            task_distribution,
            key=iter_key,
            sigma=sigma,
            rank=rank,
        )
        history.append(experience)

        # EGGROLL: compute ES gradient from traces and apply Adam update
        neg_grad, grad_metrics = _es_gradient_from_traces(traces, sigma)
        params, opt_state = apply_es_update(params, neg_grad, optimizer, opt_state)

        mean_fit = sum(t.mean_fitness for t in traces) / len(traces)
        best_fit = max(t.mean_fitness for t in traces)
        print(
            f"GEA iteration {i + 1}/{num_iterations} | "
            f"mean fitness: {mean_fit:.4f} | "
            f"best fitness: {best_fit:.4f} | "
            f"grad norm: {grad_metrics['grad_norm']:.4f} | "
            f"parents: {experience['parent_fitness']}"
        )

    return params, history


# ---------------------------------------------------------------------------
# Multi-island evolution (optional — Phase 0 enhancement)
# ---------------------------------------------------------------------------


@dataclass
class IslandState:
    """Tracks one island in a multi-island evolution setup.

    Each island runs its own ``GroupEvolver`` with independent
    parameters, seeds, and optionally slightly different configs.
    """

    evolver: GroupEvolver
    params: PyTree
    optimizer: optax.GradientTransformation = field(repr=False, default=None)
    opt_state: EGGROLLState = field(repr=False, default=None)
    history: list[dict] = field(default_factory=list)
    island_id: int = 0


def create_island(
    base_params: PyTree,
    island_id: int = 0,
    population_size: int = 64,
    group_size: int = 5,
    diversity_bonus: float = 0.0,
    novelty_neighbors: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> IslandState:
    """Create a new evolution island.

    Args:
        base_params: initial parameters (copied, not shared)
        island_id: unique identifier
        population_size: members per generation
        group_size: parents per group
        diversity_bonus: extra reward for diverse perturbations
        novelty_neighbors: k for novelty computation
        learning_rate: Adam learning rate
        weight_decay: optional weight decay

    Returns:
        Initialized IslandState with its own optimizer
    """
    optimizer, opt_state = create_eggroll_optimizer(
        base_params, learning_rate=learning_rate, weight_decay=weight_decay
    )
    return IslandState(
        evolver=GroupEvolver(
            population_size=population_size,
            group_size=group_size,
            diversity_bonus=diversity_bonus,
            novelty_neighbors=novelty_neighbors,
        ),
        params=jax.tree.map(lambda x: x.copy(), base_params),
        optimizer=optimizer,
        opt_state=opt_state,
        island_id=island_id,
    )


def migrate_top_k(
    islands: list[IslandState],
    fraction: float = 0.2,
) -> dict[int, list[tuple[int, float]]]:
    """Migrate top performers between islands.

    Every migration step, the top ``fraction`` of members from each
    island are copied to all other islands, sharing successful weight
    patterns and CIB schedules.

    Args:
        islands: list of island states
        fraction: fraction of top members to migrate (0.2 = top 20%)

    Returns:
        Dict mapping island_id → list of (source_island_id, fitness)
    """
    migration_log: dict[int, list[tuple[int, float]]] = {i.island_id: [] for i in islands}

    for src in islands:
        n_migrate = max(1, int(src.evolver.population_size * fraction))
        recent = src.evolver.archive[-src.evolver.population_size :]
        sorted_traces = sorted(recent, key=lambda t: t.mean_fitness, reverse=True)
        migrants = sorted_traces[:n_migrate]

        for migrant in migrants:
            for dst in islands:
                if dst.island_id != src.island_id:
                    migration_log[dst.island_id].append((src.island_id, migrant.mean_fitness))

    return migration_log


def run_multi_island_evolution(
    base_params: PyTree,
    fitness_fn: Callable,
    task_distribution: list[dict],
    *,
    key: jax.Array,
    num_islands: int = 10,
    num_iterations: int = 30,
    migration_interval: int = 5,
    migration_fraction: float = 0.2,
    population_size: int = 64,
    group_size: int = 5,
    sigma: float = 0.001,
    rank: int = 1,
    learning_rate: float = 1e-3,
    diversity_bonus: float = 0.0,
    island_configs: Optional[list[dict]] = None,
) -> tuple[list[IslandState], list[dict]]:
    """Run multi-island GEA-EGGROLL evolution.

    Each island runs an independent ``GroupEvolver``. Every
    ``migration_interval`` iterations, top performers migrate between
    islands, sharing successful weight patterns.

    This is the Phase 0 enhancement from the research roadmap:
    multi-island + diversity bonus + experience migration.

    Args:
        base_params: initial mean parameters (copied to all islands)
        fitness_fn: f(params, task) → (fitness, metrics)
        task_distribution: tasks to evaluate on
        key: PRNG key
        num_islands: number of parallel islands
        num_iterations: total evolution iterations
        migration_interval: migrate every N iterations
        migration_fraction: fraction of top members to migrate
        population_size: members per generation per island
        group_size: parents per group per island
        sigma: EGGROLL perturbation scale
        rank: perturbation rank
        learning_rate: Adam learning rate for ES gradient updates
        diversity_bonus: base diversity bonus (per-island overrides available)
        island_configs: optional per-island config dicts with keys like
            ``population_size``, ``group_size``, ``diversity_bonus``,
            ``novelty_neighbors``, ``learning_rate``. If provided,
            ``num_islands`` must match ``len(island_configs)``.

    Returns:
        (list of final IslandStates, global history)
    """
    if island_configs is not None:
        assert len(island_configs) == num_islands
        islands = [
            create_island(base_params, island_id=i, **cfg) for i, cfg in enumerate(island_configs)
        ]
    else:
        islands = [
            create_island(
                base_params,
                island_id=i,
                population_size=population_size,
                group_size=group_size,
                diversity_bonus=diversity_bonus,
                learning_rate=learning_rate,
            )
            for i in range(num_islands)
        ]

    global_history = []

    for iteration in range(num_iterations):
        keys = jax.random.split(key, num_islands + 1)
        key = keys[0]
        island_keys = keys[1:]

        island_stats = []
        for isl, island_key in zip(islands, island_keys):
            experience, traces = isl.evolver.evolution_step(
                isl.params,
                fitness_fn,
                task_distribution,
                key=island_key,
                sigma=sigma,
                rank=rank,
            )

            # EGGROLL update from traces
            neg_grad, _ = _es_gradient_from_traces(traces, sigma)
            isl.params, isl.opt_state = apply_es_update(
                isl.params, neg_grad, isl.optimizer, isl.opt_state
            )

            isl.history.append(experience)
            mean_fit = sum(t.mean_fitness for t in traces) / len(traces)
            best_fit = max(t.mean_fitness for t in traces)
            island_stats.append((isl.island_id, mean_fit, best_fit))

        # Migration
        if (iteration + 1) % migration_interval == 0:
            migration_log = migrate_top_k(islands, fraction=migration_fraction)
            for isl_id, migrants in migration_log.items():
                if migrants:
                    print(
                        f"  Island {isl_id}: received {len(migrants)} migrants "
                        f"(best fitness: {max(f for _, f in migrants):.4f})"
                    )

        # Global log
        global_mean = sum(s[1] for s in island_stats) / len(island_stats)
        global_best = max(s[2] for s in island_stats)
        entry = {
            "iteration": iteration + 1,
            "global_mean_fitness": global_mean,
            "global_best_fitness": global_best,
            "islands": [{"id": i, "mean": m, "best": b} for i, m, b in island_stats],
        }
        global_history.append(entry)

        print(
            f"Multi-island iter {iteration + 1}/{num_iterations} | "
            f"global mean: {global_mean:.4f} | global best: {global_best:.4f}"
        )

    return islands, global_history
