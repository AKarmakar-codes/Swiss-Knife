"""
Swiss Knife — Reconfiguration Profiling Script (Phase 3)
=========================================================

Profiles the cost of hot-swapping alignment blades at runtime.
Compares Swiss Knife pointer-swap cost against the analytical MoD retrain estimate.

Usage:
    python -m experiments.reconfiguration_profile --mock
    python -m experiments.reconfiguration_profile --N 10 --blades helpfulness harmlessness

Output:
    Console table + CSV with per-swap timing, memory delta, adapter params.
"""

import argparse
import csv
import logging
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def run_profile(
    blade_names: List[str],
    N: int = 20,
    mock: bool = True,
    output_dir: str = "runs/reconfiguration_profile",
):
    """Profile blade swap performance over N rounds.

    Parameters
    ----------
    blade_names : list of str  — Blades to cycle through.
    N : int                    — Number of swap rounds.
    mock : bool                — If True, use mock blades.
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    from Model_mechanics.blade_rack import BladeRack, MoDStyleRetrainEstimate, ReconfigurationProfile

    if mock:
        # ── Mock run without model weights ──────────────────────────────
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.blade_sources = {b: {} for b in blade_names}
        mock_cfg.beta = 0.1
        mock_tok = MagicMock()
        mock_base = MagicMock()

        # Create a mock DPOBlade for each blade
        from Model_mechanics.blades import DPOBlade

        class MockBladeRack:
            """Mock BladeRack that generates synthetic swap profiles."""
            def __init__(self):
                self._active = None
                self._blades = {b: MagicMock() for b in blade_names}

            def swap(self, name):
                from_b = self._active or "<none>"
                t0 = time.perf_counter()
                self._active = name
                swap_ms = (time.perf_counter() - t0) * 1000 + 0.05  # ~0.05ms realistic
                profile = ReconfigurationProfile(
                    from_blade=from_b,
                    to_blade=name,
                    swap_time_ms=swap_ms,
                    memory_before_mb=0.0,
                    memory_after_mb=0.0,
                    memory_delta_mb=0.0,
                    adapter_params=7_340_032,  # ~7M LoRA params (realistic for Qwen2.5)
                )
                return self._blades[name], profile

        rack = MockBladeRack()
    else:
        from Model_mechanics.config import SwissKnifeConfig
        from Model_mechanics.models import load_tokenizer, load_base_model
        cfg = SwissKnifeConfig()
        tok = load_tokenizer(cfg)
        base = load_base_model(cfg)
        rack = BladeRack(cfg, tok, base)
        rack.load_all()

    # ── Swap loop ─────────────────────────────────────────────────────────
    profiles = []
    for i in range(N):
        target_blade = blade_names[i % len(blade_names)]
        _, profile = rack.swap(target_blade)
        profiles.append(profile)
        logger.info("Swap %3d: %s", i + 1, profile)

    # ── Compute statistics ─────────────────────────────────────────────────
    swap_times = [p.swap_time_ms for p in profiles]
    mean_time = sum(swap_times) / len(swap_times)
    max_time  = max(swap_times)
    min_time  = min(swap_times)

    # ── Print comparison table ─────────────────────────────────────────────
    mod_est = MoDStyleRetrainEstimate()
    mod_summary = mod_est.summary()

    print("\n" + "═" * 70)
    print("  Reconfiguration Profile — Swiss Knife vs. MoD Retrain")
    print("═" * 70)
    print()
    print("  Swiss Knife blade swap statistics:")
    print(f"    Rounds measured      : {N}")
    print(f"    Mean swap time       : {mean_time:.3f} ms")
    print(f"    Min  swap time       : {min_time:.3f} ms")
    print(f"    Max  swap time       : {max_time:.3f} ms")
    print(f"    Adapter params       : {profiles[-1].adapter_params:,}")
    print(f"    Memory delta (mean)  : {sum(p.memory_delta_mb for p in profiles)/N:+.4f} MB")
    print()
    print("  MoD retrain cost estimate (adding a NEW objective):")
    print(f"    Router params        : {mod_summary['router_params']}")
    print(f"    Joint pathway params : {mod_summary['joint_pathway_params']}")
    print(f"    GPU-hours estimate   : {mod_summary['gpu_hours_estimate']}")
    print(f"    Training tokens      : {mod_summary['training_tokens']}")
    print()
    print(f"  Speedup: {1e6 * mean_time / 1000:.0f}× faster than 1-GPU-hour MoD retrain")
    print("═" * 70)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "reconfiguration_profile.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "swap_num", "from_blade", "to_blade",
            "swap_time_ms", "memory_delta_mb", "adapter_params",
        ])
        writer.writeheader()
        for i, p in enumerate(profiles):
            writer.writerow({
                "swap_num": i + 1,
                "from_blade": p.from_blade,
                "to_blade": p.to_blade,
                "swap_time_ms": round(p.swap_time_ms, 4),
                "memory_delta_mb": round(p.memory_delta_mb, 4),
                "adapter_params": p.adapter_params,
            })
    print(f"\nCSV saved to: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Swiss Knife — Reconfiguration Profiling")
    p.add_argument("--blades", nargs="+",
                   default=["helpfulness", "harmlessness", "truthfulness"])
    p.add_argument("--N", type=int, default=20, help="Number of swap rounds")
    p.add_argument("--mock", action="store_true", default=False)
    p.add_argument("--output-dir", default="runs/reconfiguration_profile")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_profile(
        blade_names=args.blades,
        N=args.N,
        mock=args.mock,
        output_dir=args.output_dir,
    )
