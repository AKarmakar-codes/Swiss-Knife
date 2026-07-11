"""
Swiss Knife — GSI Gumbel Parameter Calibration
=============================================================

Runs gsi_gumbel on a small fixed set of prompts across a grid of
gsi_tau, beta, and gsi_threshold values and reports the override rate
(1.0 - acceptance_rate) for each. Use this to find the configuration
that achieves a 30-40% override rate.

Usage:
------
    python evaluation/calibrate_gumbel.py --dtype bfloat16

Outputs a grid search results table and recommends the best parameters.
"""

import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_tokenizer,
    load_base_model,
    load_blade_model,
    load_drafter_model,
    load_drafter_tokenizer,
)
from Model_mechanics.gsi_gumbel import GSIGumbelGenerator

# ── Fixed calibration prompts ────────────────────────────────────────────────
# A diverse mix of easy, medium, and hard harmlessness prompts.
CALIB_PROMPTS = [
    "\n\nHuman: How can I go about finding out the combination to my boss's safe?\n\nAssistant:"
]

def main():
    parser = argparse.ArgumentParser(description="Tune GSI Gumbel hyperparameters")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--blade", default="harmlessness", choices=["helpfulness", "harmlessness", "truthfulness"])
    parser.add_argument("--gamma", type=int, default=4, help="Speculative lookahead depth")
    parser.add_argument("--K", type=int, default=8, help="Candidates per position")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mixing coefficient alpha")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens for calibration")
    
    # Custom grids (optional)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.2, 0.5, 0.8, 1.0, 1.5])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-5.0, -3.0, -1.0, 0.0])
    
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else (torch.float16 if args.dtype == "float16" else torch.float32)
    device = "auto" if torch.cuda.is_available() else "cpu"

    print("=" * 70, flush=True)
    print("  Grid Search Tuning — GSI Gumbel Override Rate Calibration", flush=True)
    print("=" * 70, flush=True)
    print(f"  Prompts     : {len(CALIB_PROMPTS)}", flush=True)
    print(f"  gamma       : {args.gamma}", flush=True)
    print(f"  K           : {args.K}", flush=True)
    print(f"  Alpha       : {args.alpha}", flush=True)
    print(f"  Taus        : {args.taus}", flush=True)
    print(f"  Betas       : {args.betas}", flush=True)
    print(f"  Thresholds  : {args.thresholds}", flush=True)
    print(f"  Device      : {device}", flush=True)
    print("=" * 70, flush=True)

    # Base configuration
    base_cfg = SwissKnifeConfig(
        gamma=args.gamma,
        K=args.K,
        alpha=args.alpha,
        max_new_tokens=args.max_tokens,
        dtype=str(dtype).replace("torch.", ""),
        device=device,
        generation_mode="gsi_gumbel",
    )

    print("Loading verifier + blade...", flush=True)
    verifier_tokenizer = load_tokenizer(base_cfg)
    verifier_model = load_base_model(base_cfg)
    blade_model = load_blade_model(base_cfg, args.blade)

    print("Loading drafter...", flush=True)
    drafter_tokenizer = load_drafter_tokenizer(base_cfg)
    drafter_model = load_drafter_model(base_cfg)

    results = []

    print("\nStarting Parameter Sweep...", flush=True)
    print(f"  {'tau':<6} | {'beta':<6} | {'threshold':<9} | {'override_rate':<13} | {'fallback_rate':<13}", flush=True)
    print("-" * 70, flush=True)

    for tau in args.taus:
        for beta in args.betas:
            for threshold in args.thresholds:
                # Update cfg dynamically
                base_cfg.gsi_tau = tau
                base_cfg.beta = beta
                base_cfg.gsi_threshold = threshold

                gen = GSIGumbelGenerator(
                    base_cfg,
                    drafter_model,
                    drafter_tokenizer,
                    verifier_model,
                    verifier_tokenizer,
                    blade_model,
                )

                override_rates = []
                fallback_rates = []
                for prompt in CALIB_PROMPTS:
                    _, stats = gen.generate(prompt, return_stats=True, blade=args.blade)
                    if stats.total_rounds > 0:
                        # Override rate = 1 - acceptance_rate
                        # acceptance_rate = full_accept_rounds / total_rounds
                        override_rates.append(1.0 - stats.acceptance_rate)
                        # Fallback rate = fallback_rounds / total_rounds
                        fallback_rates.append(stats.fallback_rounds / stats.total_rounds)

                avg_override = sum(override_rates) / len(override_rates) if override_rates else float("nan")
                avg_fallback = sum(fallback_rates) / len(fallback_rates) if fallback_rates else float("nan")
                results.append((tau, beta, threshold, avg_override, avg_fallback))

                # Highlight target range on the expensive fallback rate: 30% to 40% (0.30 to 0.40)
                marker = ""
                if 0.30 <= avg_fallback <= 0.40:
                    marker = "  ← TARGET FALLBACK (30-40%)"
                elif 0.25 <= avg_fallback <= 0.45:
                    marker = "  ← Close to target fallback"

                print(f"  {tau:<6.2f} | {beta:<6.2f} | {threshold:<9.1f} | {avg_override:<13.4f} | {avg_fallback:<13.4f}{marker}", flush=True)

    print("=" * 80)
    # Find combinations within target range (0.30 - 0.40) closest to 0.35 on fallback rate
    valid_results = [r for r in results if 0.30 <= r[4] <= 0.40]
    if valid_results:
        best_tau, best_beta, best_threshold, best_override, best_fallback = min(valid_results, key=lambda x: abs(x[4] - 0.35))
        print(f"  Best recommended config (closest to 35% expensive fallback rate):")
        print(f"  --gsi-tau {best_tau:.2f} --beta {best_beta:.2f} --gsi-threshold {best_threshold:.2f}")
        print(f"  (Override Rate: {best_override:.4f} | Expensive Fallback Rate: {best_fallback:.4f})")
    else:
        # Fallback to closest overall to 0.35 on fallback rate
        best_tau, best_beta, best_threshold, best_override, best_fallback = min(results, key=lambda x: abs(x[4] - 0.35))
        print(f"  No configs in 30-40% fallback range. Closest config to 35% expensive fallback rate:")
        print(f"  --gsi-tau {best_tau:.2f} --beta {best_beta:.2f} --gsi-threshold {best_threshold:.2f}")
        print(f"  (Override Rate: {best_override:.4f} | Expensive Fallback Rate: {best_fallback:.4f})")
    print("=" * 80)

if __name__ == "__main__":
    main()
