"""Smoke test runner for the velm package.

Runs small, dependency-guarded forward passes for available backends
(JAX/Equinox research implementation and PyTorch proxy). Exits with
0 on success, non-zero on failure. Designed for fast CI smoke tests.
"""

from __future__ import annotations


def main() -> int:
    ok = True
    msgs = []

    # Basic import of the package
    try:
        import importlib

        importlib.import_module("velm")
        msgs.append("Imported velm package")
    except Exception as e:  # pragma: no cover - environment dependent
        msgs.append(f"Import velm failed: {e}")
        ok = False

    # Try importing the high-fidelity JAX implementation (may be optional)
    try:
        from velm.model import VELM  # type: ignore

        msgs.append("VELM (JAX) importable")
    except Exception as e:  # pragma: no cover - optional dependency
        msgs.append(f"VELM (JAX) unavailable: {e}")

    # Try importing and running a very small PyTorch proxy forward pass
    try:
        from velm.model import VelmFull  # type: ignore
        import torch

        if VelmFull is None:
            msgs.append("VelmFull is not available")
            ok = False
        else:
            # instantiate an extremely small model and run a forward pass
            try:
                model = VelmFull(vocab_size=32, block_size=4, embed_dim=8, latent_dim=8, state_dim=16)
                model.eval()
                x = torch.randint(3, 32, (2, 8), dtype=torch.long)
                with torch.no_grad():
                    states, latents = model(x, use_adapter=False)
                msgs.append("VelmFull forward OK")
            except Exception as e:  # pragma: no cover - runtime dependent
                msgs.append(f"VelmFull forward failed: {e}")
                ok = False
    except Exception as e:  # pragma: no cover - environment dependent
        msgs.append(f"VelmFull import failed: {e}")
        ok = False

    for m in msgs:
        print(m)

    return 0 if ok else 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
