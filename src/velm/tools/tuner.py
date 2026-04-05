"""Hyperparameter tuner for VELM using Optuna and a Rich TUI.

Designed to run on commodity hardware using the PyTorch proxy (velm.model.VelmFull)
for fast iterations. Optimizes for accuracy on a small synthetic LM task while
encouraging small parameter counts via a weighted objective.

Run with: python -m velm.tools.tuner --trials 30
Or: velm-tune (console script)
"""

from __future__ import annotations

import argparse
import threading
import time
import math
import os
from dataclasses import dataclass

try:
    import optuna
except Exception as e:  # pragma: no cover - optional
    raise RuntimeError("Optuna is required for the tuner. Install via 'pip install optuna'.")

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console
    from rich.progress import SpinnerColumn, TimeElapsedColumn, Progress
except Exception:  # pragma: no cover - optional
    raise RuntimeError("Rich is required for the TUI. Install via 'pip install rich'.")

import torch
import torch.nn.functional as F
import numpy as np

from velm.lite import VelmFull

import warnings
from collections import deque

console = Console()

# Capture warnings globally into an in-memory deque so the TUI can display them
warnings_log: deque[str] = deque(maxlen=200)
_original_showwarning = warnings.showwarning


def _capture_showwarning(message, category, filename, lineno, file=None, line=None):
    text = warnings.formatwarning(message, category, filename, lineno, line)
    warnings_log.append(text)
    try:
        _original_showwarning(message, category, filename, lineno, file=file, line=line)
    except Exception:
        pass


warnings.showwarning = _capture_showwarning


def generate_dataset(n_samples: int, seq_len: int, vocab_size: int, max_gap: int | None = None):
    QID = 1
    assert seq_len >= 8
    L = seq_len - 1
    if max_gap is None:
        max_gap = max(2, L // 2)
    gaps = np.random.randint(1, max_gap + 1, size=n_samples)
    data = np.random.randint(3, vocab_size, size=(n_samples, seq_len))
    labels = np.random.randint(3, vocab_size, size=n_samples)
    for i in range(n_samples):
        g = int(gaps[i])
        Lbl = int(labels[i])
        data[i, 0] = Lbl
        if g >= seq_len - 1:
            g = seq_len - 2
            gaps[i] = g
        data[i, g] = QID
        data[i, g + 1] = Lbl
    return (
        torch.tensor(data, dtype=torch.long),
        torch.tensor(gaps, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class TunerConfig:
    trials: int = 20
    steps: int = 100
    device: str = "auto"
    out: str = "outputs_tune"
    seed: int = 42
    # Reduce parameter-count penalty so small models are preferred but not at the expense of accuracy
    weight_param_millions: float = 0.002  # penalty per million params in combined score (was 0.02)
    # Allow disabling UMAP when noisy or unavailable
    use_umap: bool = True


def train_and_eval(cfg: TunerConfig, hparams: dict) -> tuple[float, int, float]:
    """Train the VelmFull proxy for a few steps and return (accuracy, param_count, time_elapsed)."""
    # dataset small enough for commodity hardware
    train_n = 800
    test_n = 200
    seq_len = 33
    vocab = 256

    train_x, train_gaps, _ = generate_dataset(train_n, seq_len, vocab, max_gap=12)
    test_x, test_gaps, _ = generate_dataset(test_n, seq_len, vocab, max_gap=12)

    device = torch.device(
        "cuda"
        if (cfg.device == "auto" and torch.cuda.is_available()) or cfg.device == "cuda"
        else "cpu"
    )

    model = VelmFull(
        vocab_size=vocab,
        block_size=hparams["block_size"],
        embed_dim=hparams["embed_dim"],
        latent_dim=hparams["latent_dim"],
        state_dim=hparams["state_dim"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    batch_size = hparams["batch_size"]
    steps = cfg.steps

    N = train_x.shape[0]

    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = train_x[idx].to(device)
        hist = batch[:, :-1]
        q_pos = (hist == 1).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
        states, latents = model(hist)
        b_idx = (q_pos // hparams["block_size"]).long()
        logits_at = model.decode_block_state(states[torch.arange(states.shape[0]), b_idx, :])
        # decode_block_state returns (B, K, vocab_size).
        # We need the logits corresponding to the q_pos relative to its block.
        rel_q_pos = q_pos % hparams["block_size"]
        logits_at = logits_at[torch.arange(logits_at.shape[0]), rel_q_pos, :]
        loss = F.cross_entropy(logits_at, targets)
        # CIB proxy
        lat_q = latents[torch.arange(latents.shape[0]), b_idx, :]
        cib = lat_q.norm(p=2, dim=1).mean()
        loss = loss + hparams.get("cib_lambda", 1e-3) * cib
        opt.zero_grad()
        loss.backward()
        opt.step()
    t1 = time.time()
    elapsed_time = t1 - t0

    # Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        B = 128
        for i in range(0, test_x.shape[0], B):
            batch = test_x[i : i + B].to(device)
            hist = batch[:, :-1]
            q_pos = (hist == 1).int().argmax(dim=1)
            states, latents = model(hist)
            b_idx = (q_pos // hparams["block_size"]).long()
            logits_at = model.decode_block_state(states[torch.arange(states.shape[0]), b_idx, :])
            rel_q_pos = q_pos % hparams["block_size"]
            logits_at = logits_at[torch.arange(logits_at.shape[0]), rel_q_pos, :]
            targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
            pred = logits_at.argmax(dim=1)
            correct += (pred == targets).sum().item()
    acc = correct / test_x.shape[0]
    param_count = count_parameters(model)
    return acc, param_count, elapsed_time


def objective(trial: optuna.trial.Trial, cfg: TunerConfig) -> float:
    # hyperparameter search space
    # Expanded hyperparameter ranges to allow larger models capable of solving the task
    embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64, 128, 256])
    # latent_dim should be between a fraction of embed_dim and up to embed_dim*2 (but bounded reasonably)
    latent_min = max(8, embed_dim // 2)
    latent_max = max(embed_dim, min(256, embed_dim * 2))
    latent_dim = trial.suggest_int("latent_dim", latent_min, latent_max, step=8)
    # allow larger state dimensions for expressivity
    # State dim must be divisible by num_heads = max(1, state_dim // 32)
    state_dim = trial.suggest_categorical("state_dim", [32, 64, 128, 256, 512])
    block_size = trial.suggest_categorical("block_size", [1, 2, 4, 8])
    # wider LR search and allow smaller rates
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    # allow stronger and weaker CIB penalties
    cib_lambda = trial.suggest_float("cib_lambda", 1e-5, 1e-1, log=True)

    hparams = {
        "embed_dim": embed_dim,
        "latent_dim": latent_dim,
        "state_dim": state_dim,
        "block_size": block_size,
        "lr": lr,
        "batch_size": batch_size,
        "cib_lambda": cib_lambda,
    }

    # train and evaluate
    acc, param_count, elapsed_time = train_and_eval(cfg, hparams)

    # combined score (maximize): accuracy - penalty * params_in_millions
    params_m = param_count / 1e6
    score = float(acc) - cfg.weight_param_millions * float(params_m)

    # report to optuna (higher is better)
    trial.set_user_attr("accuracy", float(acc))
    trial.set_user_attr("param_count", int(param_count))
    trial.set_user_attr("params_millions", float(params_m))
    trial.set_user_attr("time_elapsed", float(elapsed_time))
    trial.report(score, step=0)

    return score


def ui_thread_fn(
    study: optuna.study.Study, stop_event: threading.Event, cfg: TunerConfig, refresh: float = 0.5
):
    """Full-screen Rich live TUI showing trials, hyperparameters, progress, captured warnings, and search-space map."""
    from rich.layout import Layout
    from rich.align import Align

    # Rich Image may not be available in older rich versions; import dynamically later
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start_time = time.time()

    tmp_plot = os.path.join(cfg.out, "__tuner_umap.png")

    with console.screen(), Live(console=console, refresh_per_second=8) as live:
        while not stop_event.is_set():
            trials = study.trials
            # Summary / progress
            total = cfg.trials
            # optuna TrialState comparisons
            completed = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)
            running = sum(1 for t in trials if t.state == optuna.trial.TrialState.RUNNING)
            prct = (completed / total) * 100 if total > 0 else 0.0
            elapsed = time.time() - start_time
            eta = None
            if completed > 0:
                rate = completed / max(1e-6, elapsed)
                eta = (total - completed) / rate

            # Build hyperparameter embedding for visualization
            hp_rows = []
            hp_vals = []
            hp_labels = []
            for t in trials:
                if t.params:
                    p = t.params
                    # vectorize params consistently: embed_dim, latent_dim, state_dim, block_size, batch_size, lr, cib_lambda
                    vec = [
                        float(p.get("embed_dim", 0)),
                        float(p.get("latent_dim", 0)),
                        float(p.get("state_dim", 0)),
                        float(p.get("block_size", 0)),
                        float(p.get("batch_size", 0)),
                        float(p.get("lr", 0.0)),
                        float(p.get("cib_lambda", 0.0)),
                    ]
                    hp_rows.append(vec)
                    hp_vals.append(t.value if t.value is not None else 0.0)
                    hp_labels.append(t.number)

            scatter_img = None
            if len(hp_rows) >= 2:
                try:
                    import numpy as _np

                    X = _np.array(hp_rows, dtype=_np.float32)

                    # Try UMAP first for richer embeddings (if available) and silence noisy warnings
                    try:
                        if cfg.use_umap:
                            import logging

                            with warnings.catch_warnings():
                                # suppress common noisy warnings from numba/umap
                                warnings.filterwarnings(
                                    "ignore", category=UserWarning, module="numba"
                                )
                                warnings.filterwarnings(
                                    "ignore", category=UserWarning, module="umap"
                                )
                                try:
                                    # Prefer stable public API: from umap import UMAP
                                    try:
                                        from umap import UMAP
                                    except Exception:
                                        import umap as _umap

                                        UMAP = getattr(_umap, "UMAP", None)
                                    if UMAP is None:
                                        raise ImportError("UMAP API not found in umap package")

                                    n_samples = X.shape[0]
                                    if n_samples < 3:
                                        raise RuntimeError(
                                            "Too few samples for UMAP; falling back to PCA"
                                        )

                                    n_neighbors = min(15, max(2, n_samples - 1))
                                    # configure UMAP with conservative defaults to avoid warnings on small data
                                    reducer = UMAP(
                                        n_components=2,
                                        random_state=cfg.seed,
                                        n_neighbors=n_neighbors,
                                        min_dist=0.1,
                                        metric="euclidean",
                                        verbose=False,
                                    )
                                    # silence warnings during fit
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        proj = reducer.fit_transform(X)
                                    xs = proj[:, 0]
                                    ys = proj[:, 1]
                                except Exception as e_umap:
                                    # record once and fallback to PCA
                                    warnings_log.append(
                                        f"UMAP disabled due to import/fit error: {e_umap}"
                                    )
                                    Xc = X - X.mean(axis=0)
                                    U, S, Vt = _np.linalg.svd(Xc, full_matrices=False)
                                    proj = Xc @ Vt.T[:, :2]
                                    xs = proj[:, 0]
                                    ys = proj[:, 1]
                        else:
                            # forced PCA
                            Xc = X - X.mean(axis=0)
                            U, S, Vt = _np.linalg.svd(Xc, full_matrices=False)
                            proj = Xc @ Vt.T[:, :2]
                            xs = proj[:, 0]
                            ys = proj[:, 1]
                    except Exception as e:
                        warnings_log.append(f"2D projection failed: {e}")
                        xs = np.zeros(0)
                        ys = np.zeros(0)

                    # color by value (higher better)
                    vals = _np.array(hp_vals)
                    # normalize values for colormap
                    if vals.max() - vals.min() > 1e-8:
                        norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
                    else:
                        norm_vals = _np.zeros_like(vals)

                    plt.figure(figsize=(6, 4), dpi=100)
                    plt.scatter(xs, ys, c=norm_vals, cmap="viridis", s=60, edgecolor="k")
                    # annotate top 3 points
                    order = _np.argsort(vals)[::-1]
                    for idx in order[:3]:
                        plt.annotate(
                            str(hp_labels[idx]), (xs[idx], ys[idx]), fontsize=8, weight="bold"
                        )
                    plt.title("Search space (2D) — colored by trial value")
                    plt.xlabel("Dim1")
                    plt.ylabel("Dim2")
                    plt.colorbar(label="Normalized score")
                    plt.tight_layout()
                    plt.savefig(tmp_plot)
                    plt.close()
                    try:
                        # Try to render with Rich image support if available
                        from rich.image import Image as RichImage

                        scatter_img = RichImage(tmp_plot, width=48)
                    except Exception:
                        # Fallback: render a colorful ASCII scatter using the projected coords
                        try:
                            from rich.text import Text

                            # build small grid
                            W, H = 48, 18
                            xs_norm = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
                            ys_norm = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
                            grid = [[None for _ in range(W)] for _ in range(H)]
                            # colormap (viridis-like hex colors)
                            cmap = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]

                            for i_pt in range(len(xs)):
                                gx = int(xs_norm[i_pt] * (W - 1))
                                gy = int((1.0 - ys_norm[i_pt]) * (H - 1))
                                color_idx = (
                                    int(norm_vals[i_pt] * (len(cmap) - 1))
                                    if len(norm_vals) == len(xs)
                                    else 0
                                )
                                grid[gy][gx] = (i_pt, cmap[color_idx])

                            lines = []
                            for row in grid:
                                txt = Text()
                                for cell in row:
                                    if cell is None:
                                        txt.append(" ")
                                    else:
                                        idx_cell, color = cell
                                        txt.append("●", style=color)
                                lines.append(txt)

                            # annotate top 3 by value on a separate small legend
                            legend = Text()
                            order = np.argsort(vals)[::-1]
                            for idx in order[:3]:
                                legend.append(f"#{hp_labels[idx]} ", style="bold")
                                legend.append(
                                    "■ ", style=f"{cmap[int(norm_vals[idx] * (len(cmap) - 1))]}"
                                )
                                legend.append(f" val={hp_vals[idx]:.4f}  ")

                            # combine legend and lines into a Panel-compatible Text
                            combined = Text("\n")
                            combined.append(legend)
                            combined.append("\n")
                            for tline in lines:
                                combined.append(tline)
                                combined.append("\n")

                            scatter_img = combined
                        except Exception as e2:
                            warnings_log.append(f"ASCII plot failed: {e2}")
                            scatter_img = None
                except Exception as e:
                    warnings_log.append(f"2D projection failed: {e}")
                    scatter_img = None

            # Top trials table
            table = Table(expand=True)
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Trial", style="magenta", width=6)
            table.add_column("Value", style="green", width=9)
            table.add_column("Acc", style="yellow", width=7)
            table.add_column("Time(s)", style="blue", width=7)
            table.add_column("Params(M)", style="cyan", width=10)
            table.add_column("Block", style="white", width=6)
            table.add_column("Embed", style="white", width=7)
            table.add_column("Latent", style="white", width=7)
            table.add_column("StateDim", style="white", width=8)
            table.add_column("Batch", style="white", width=6)
            table.add_column("LR", style="white", width=9)
            table.add_column("CIB", style="white", width=8)
            table.add_column("Status", style="bright_white", width=10)

            sorted_trials = sorted(
                [t for t in trials if t.value is not None], key=lambda t: t.value, reverse=True
            )
            for i, t in enumerate(sorted_trials[: min(20, len(sorted_trials))]):
                acc = t.user_attrs.get("accuracy", "-")
                pm = t.user_attrs.get("params_millions", "-")
                tm = t.user_attrs.get("time_elapsed", "-")
                params = t.params
                table.add_row(
                    str(i + 1),
                    str(t.number),
                    f"{t.value:.4f}",
                    f"{acc:.4f}" if isinstance(acc, float) else str(acc),
                    f"{tm:.2f}" if isinstance(tm, float) else str(tm),
                    f"{pm:.3f}" if isinstance(pm, float) else str(pm),
                    str(params.get("block_size", "-")),
                    str(params.get("embed_dim", "-")),
                    str(params.get("latent_dim", "-")),
                    str(params.get("state_dim", "-")),
                    str(params.get("batch_size", "-")),
                    f"{params.get('lr', '-'):.4g}"
                    if isinstance(params.get("lr", None), float)
                    else str(params.get("lr", "-")),
                    f"{params.get('cib_lambda', '-'):.2g}"
                    if isinstance(params.get("cib_lambda", None), float)
                    else str(params.get("cib_lambda", "-")),
                    str(t.state),
                )

            # Details panel for best trial
            best_panel = Table.grid(expand=True)
            best_panel.add_column(justify="left")
            best_panel.add_column(justify="left")
            # study.best_trial can raise ValueError if storage has no completed trials yet (RDB corner case)
            try:
                bt = study.best_trial
            except Exception:
                bt = None

            if bt is not None:
                best_panel.add_row("Best value:", f"{bt.value:.6f}")
                best_panel.add_row("Params:", str(bt.user_attrs.get("param_count", "-")))
                best_panel.add_row("Hyperparameters:", "")
                for k, v in bt.params.items():
                    best_panel.add_row(f"  {k}", str(v))
            else:
                best_panel.add_row("Best trial:", "N/A yet")

            # Warnings panel
            warnings_table = Table.grid(expand=True)
            warnings_table.add_column()
            if len(warnings_log) == 0:
                warnings_table.add_row("No warnings captured so far.")
            else:
                # show last 8 warnings
                for w in list(warnings_log)[-8:]:
                    warnings_table.add_row(w.strip())

            # Progress summary
            prog = Table.grid(expand=True)
            prog.add_column()
            prog.add_row(f"Completed: {completed}/{total} ({prct:.1f}%) | Running: {running}")
            prog.add_row(f"Elapsed: {elapsed:.1f}s")
            if eta is not None:
                prog.add_row(f"ETA: {eta:.1f}s")

            # Layout
            layout = Layout()
            layout.split_column(
                Layout(name="top", ratio=3),
                Layout(name="bottom", ratio=1),
            )
            layout["top"].split_row(
                Layout(name="left"),
                Layout(name="right", size=60),
            )
            # left: table + plot
            left_col = Layout(name="left_col")
            left_col.split_column(
                Layout(Panel(table, title="Top Trials"), ratio=3), Layout(name="plot", ratio=2)
            )

            # populate the inner plot layout before attaching
            if scatter_img is not None:
                left_col["plot"].update(Panel(Align.center(scatter_img), title="Search map (PCA)"))
            else:
                left_col["plot"].update(Panel("(no plot available)", title="Search map (PCA)"))

            layout["top"]["left"].update(left_col)

            layout["top"]["right"].update(Panel(best_panel, title="Best Trial"))
            layout["bottom"].split_row(
                Layout(Panel(prog, title="Progress")),
                Layout(Panel(warnings_table, title="Captured Warnings")),
            )

            live.update(layout)
            time.sleep(refresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="outputs_tune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-param-millions",
        type=float,
        default=None,
        help="Penalty weight per million parameters (overrides default in config). Lower prefers accuracy over compactness.",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Disable UMAP and force PCA-only 2D projection (suppresses umap/numba warnings).",
    )
    args = parser.parse_args(argv)

    # Allow CLI override for the parameter-count penalty
    penalty = (
        args.weight_param_millions
        if args.weight_param_millions is not None
        else TunerConfig.weight_param_millions
    )
    cfg = TunerConfig(
        trials=args.trials,
        steps=args.steps,
        device=args.device,
        out=args.out,
        seed=args.seed,
        weight_param_millions=penalty,
        use_umap=not args.no_umap,
    )
    os.makedirs(cfg.out, exist_ok=True)

    storage = f"sqlite:///{os.path.join(cfg.out, 'optuna.db')}"
    study_name = "velm_tuning"
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage, load_if_exists=True
    )

    stop_event = threading.Event()
    ui_t = threading.Thread(target=ui_thread_fn, args=(study, stop_event, cfg), daemon=True)
    ui_t.start()

    try:
        try:
            study.optimize(lambda t: objective(t, cfg), n_trials=cfg.trials)
        except ValueError as e:
            # Handle incompatible distribution changes in a persisted study (Optuna RDB stores distributions)
            msg = str(e)
            if (
                "CategoricalDistribution does not support dynamic value space" in msg
                or "dynamic value space" in msg
            ):
                console.print(
                    "Incompatible prior study distributions detected — creating a fresh study to continue tuning."
                )
                ts = int(time.time())
                new_db = os.path.join(cfg.out, f"optuna_{ts}.db")
                new_storage = f"sqlite:///{new_db}"
                study_name = f"velm_tuning_{ts}"
                study = optuna.create_study(
                    direction="maximize",
                    study_name=study_name,
                    storage=new_storage,
                    load_if_exists=False,
                )
                # restart UI thread with new study reference
                stop_event.set()
                ui_t.join(timeout=1.0)
                stop_event = threading.Event()
                ui_t = threading.Thread(
                    target=ui_thread_fn, args=(study, stop_event, cfg), daemon=True
                )
                ui_t.start()
                study.optimize(lambda t: objective(t, cfg), n_trials=cfg.trials)
            else:
                raise
    except KeyboardInterrupt:
        console.print("Keyboard interrupt — stopping optimization")
    finally:
        stop_event.set()
        ui_t.join(timeout=2.0)

    console.print("Best trial:")
    if study.best_trial is not None:
        bt = study.best_trial
        console.print(f"  Value: {bt.value:.4f}")
        console.print(f"  Params: {bt.user_attrs.get('param_count', '-')}")
        console.print("  Hyperparameters:")
        for k, v in bt.params.items():
            console.print(f"    {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
