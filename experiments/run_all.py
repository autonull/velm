#!/usr/bin/env python3
"""Turnkey runner for all VELM experiments.

Usage:
  python experiments/run_all.py              # Run all experiments sequentially
  python experiments/run_all.py --smoketest  # Quick end-to-end smoke test (seconds)
  python experiments/run_all.py --select velm_vs_transformer qttt_demo  # Run specific experiments

Each experiment writes outputs to results/<experiment_name>/.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "results"

EXPERIMENTS = {
    "velm_vs_transformer": {
        "script": "velm_vs_transformer.py",
        "description": "VELM proxy vs Transformer baseline (basic/tuned/extended modes)",
    },
    "train_velm_full": {
        "script": "train_velm_full.py",
        "description": "Full VelmFull training + optional EGGROLL tuning",
    },
    "benchmark_lm": {
        "script": "benchmark_lm.py",
        "description": "VELM vs Vanilla Transformer on Tiny Shakespeare",
    },
    "qttt_demo": {
        "script": "qttt_demo.py",
        "description": "qTTT adapter adaptation pre/post accuracy demo",
        "supports_device": False,
    },
}

SMOKETEST_OVERRIDES = {
    "velm_vs_transformer": ["--smoketest"],
    "train_velm_full": ["--steps", "5", "--eggroll"],
    "benchmark_lm": ["--smoketest"],
    "qttt_demo": ["--smoketest"],
}


def run_experiment(name, out_dir, device, smoketest=False, extra_args=None):
    """Run a single experiment script. Returns (success, elapsed_time)."""
    config = EXPERIMENTS[name]
    script = EXPERIMENTS_DIR / config["script"]
    if not script.exists():
        print(f"  SKIP: {script.name} not found")
        return False, 0

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(script), "--out", str(out_dir)]
    cmd.extend(extra_args or [])

    if smoketest and name in SMOKETEST_OVERRIDES:
        cmd.extend(SMOKETEST_OVERRIDES[name])

    # Only pass --device if the script supports it
    if config.get("supports_device", True):
        cmd.extend(["--device", device])

    print(f"\n{'=' * 60}")
    print(f"  {name}: {config['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(EXPERIMENTS_DIR.parent), capture_output=False)
        elapsed = time.time() - t0
        success = result.returncode == 0
        status = "PASS" if success else "FAIL"
        print(f"\n  [{status}] {name} completed in {elapsed:.1f}s")
        return success, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [FAIL] {name} crashed: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all VELM experiments")
    parser.add_argument(
        "--smoketest", action="store_true", help="Quick end-to-end test with minimal data/steps"
    )
    parser.add_argument(
        "--select",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        help="Run only selected experiments",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for all experiments")
    args = parser.parse_args()

    mode = "SMOKETEST" if args.smoketest else "FULL"
    selected = args.select or list(EXPERIMENTS.keys())

    print(f"VELM Experiment Runner — {mode} mode")
    print(f"Selected experiments: {', '.join(selected)}")
    print(f"Results directory: {RESULTS_DIR}")

    results = {}
    total_start = time.time()

    for name in selected:
        out_dir = RESULTS_DIR / name
        success, elapsed = run_experiment(
            name, out_dir, device=args.device, smoketest=args.smoketest
        )
        results[name] = {"success": success, "time": elapsed}

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {mode} mode ({total_elapsed:.1f}s total)")
    print(f"{'=' * 60}")
    all_pass = True
    for name, r in results.items():
        status = "PASS" if r["success"] else "FAIL"
        if not r["success"]:
            all_pass = False
        print(f"  [{status}] {name}: {r['time']:.1f}s")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"  Results: {RESULTS_DIR}/")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
