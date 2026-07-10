"""
Swiss Knife — Elo Temperature Calibration
===========================================

Sweeps over a range of elo_temperature values using GSI-Elo on a subset of
harmlessness prompts, reporting the override rate, average DPO blade reward,
and lexical refusal rate.

Use this to find a temperature that balances safety/steering (blade reward)
and response quality (avoiding over-conservative selection/refusal).

Usage
-----
    python evaluation/calibrate_elo_temp.py --num-prompts 3 --max-tokens 100
"""

import sys
import os
import argparse
import logging
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_tokenizer,
    load_base_model,
    load_blade_model,
    load_drafter_model,
    load_drafter_tokenizer,
)
from Model_mechanics.blades import DPOBlade
from Model_mechanics.gsi_elo import GSIEloGenerator
from evaluation.benchmark_gsi_strategies_harmlessness import detect_refusal, compute_response_blade_reward, extract_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sweep elo_temperature for GSI-Elo")
    parser.add_argument("--num-prompts", type=int, default=3,
                        help="Number of prompts to calibrate on (keep small for CPU testing)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens per response during calibration")
    parser.add_argument("--gsi-n", type=int, default=8,
                        help="Number of candidate steps generated per step")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Mixing coefficient between target and blade")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="DPO implicit reward scaling factor")
    parser.add_argument("--gsi-threshold", type=float, default=-4.0,
                        help="Rejection/override threshold u")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[1.0, 5.0, 10.0, 15.0,25.0 ,40.0],
                        help="List of elo_temperature values to sweep")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "auto" if torch.cuda.is_available() else "cpu"

    print("=" * 70, flush=True)
    print("  Grid Search Sweep — GSI-Elo Temperature Calibration", flush=True)
    print("=" * 70, flush=True)
    print(f"  Prompts     : {args.num_prompts}", flush=True)
    print(f"  Max Tokens  : {args.max_tokens}", flush=True)
    print(f"  Candidates n: {args.gsi_n}", flush=True)
    print(f"  Alpha       : {args.alpha}", flush=True)
    print(f"  Beta        : {args.beta}", flush=True)
    print(f"  Threshold u : {args.gsi_threshold}", flush=True)
    print(f"  dtype       : {args.dtype}", flush=True)
    print(f"  Temperatures: {args.temperatures}", flush=True)
    print("=" * 70, flush=True)

    # ── Load dataset ──────────────────────────────────────────────────
    logger.info("Loading HH-RLHF harmless-base dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    dataset = dataset.shuffle(seed=args.seed).select(
        range(min(args.num_prompts, len(dataset)))
    )
    test_prompts = [extract_prompt(row["chosen"]) for row in dataset]
    logger.info(f"Loaded {len(test_prompts)} prompts.")

    # ── Load models ───────────────────────────────────────────────────
    base_cfg = SwissKnifeConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_new_tokens=args.max_tokens,
        dtype=args.dtype,
        device=device,
        generation_mode="gsi_elo",
        gsi_n=args.gsi_n,
        gsi_threshold=args.gsi_threshold,
        seed=args.seed,
    )

    logger.info("Loading shared verifier base model (Qwen 2.5 7B) + blade...")
    tokenizer = load_tokenizer(base_cfg)
    base_model = load_base_model(base_cfg)
    blade_model = load_blade_model(base_cfg, "harmlessness")

    logger.info("Loading drafter model (Qwen 2.5 3B)...")
    drafter_tokenizer = load_drafter_tokenizer(base_cfg)
    drafter_model = load_drafter_model(base_cfg)

    diagnostic_blade = DPOBlade(base_cfg, base_model, blade_model, tokenizer)

    results = []

    # ── Sweep Temperatures ────────────────────────────────────────────
    for temp in args.temperatures:
        print(f"\nEvaluating elo_temperature = {temp}...", flush=True)
        
        # Configure generator
        base_cfg.elo_temperature = temp
        gen = GSIEloGenerator(
            base_cfg,
            drafter_model,
            drafter_tokenizer,
            base_model,
            tokenizer,
            blade_model,
        )

        override_rates = []
        blade_rewards = []
        refusals = []

        for idx, prompt in enumerate(test_prompts):
            output, stats = gen.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                verbose=False,
                return_stats=True,
            )
            generated = output[len(prompt):].strip()
            
            # Compute stats
            stats_dict = stats.to_dict()
            
            # Override rate
            if stats_dict.get("total_steps", 0) > 0:
                override_rate = stats_dict["rejected_steps"] / stats_dict["total_steps"]
                override_rates.append(override_rate)
            
            # Response DPO reward
            reward = compute_response_blade_reward(
                diagnostic_blade, tokenizer, prompt, generated, base_model.device
            )
            blade_rewards.append(reward)
            
            # Refusal heuristic
            is_refusal = detect_refusal(generated)
            refusals.append(float(is_refusal))

            logger.info(
                f"  Prompt {idx+1}/{len(test_prompts)} | "
                f"override={override_rates[-1]:.2f} | "
                f"reward={reward:.4f} | "
                f"refusal={is_refusal}"
            )

        avg_override = sum(override_rates) / len(override_rates) if override_rates else 0.0
        avg_reward = sum(blade_rewards) / len(blade_rewards) if blade_rewards else 0.0
        avg_refusal = sum(refusals) / len(refusals) if refusals else 0.0

        results.append({
            "temperature": temp,
            "avg_override": avg_override,
            "avg_reward": avg_reward,
            "avg_refusal": avg_refusal,
        })

    # ── Final Report ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GSI-Elo Temperature Calibration Sweep Results")
    print("=" * 70)
    print(f"  {'elo_temp':<10} | {'Avg Override':<14} | {'Avg Blade Reward':<18} | {'Refusal Rate':<12}")
    print("-" * 70)
    for res in results:
        print(
            f"  {res['temperature']:<10.2f} | "
            f"{res['avg_override']:<14.2%} | "
            f"{res['avg_reward']:<18.5f} | "
            f"{res['avg_refusal']:<12.1%}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
