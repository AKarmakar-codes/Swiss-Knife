"""
Swiss Knife — GSI Helpfulness Benchmark
=======================================

Helpfulness-focused GSI benchmark that:
    - Uses the HELPFULNESS blade instead of harmlessness.
    - Samples prompts from Anthropic/hh-rlhf's helpful-base subset
      instead of the harmless-base subset.
    - Does NOT score with a reward model. This script's only job is to
      generate + save raw outputs for all strategies on the same prompts,
      so a SEPARATE script can run GEval (and any other judge)
      on identical generations afterward.
    - Adds steering-verification instrumentation: per-step blade
      reward stats and draft-vs-blade override rate.

Strategies tested:
    1. gsi_softmax              — Standard GSI: softmax(β·r̃) selection
    2. swiss                    — Swiss-system → points table → softmax
    3. elo_swiss                — Elo-system tournament selection
    4. swiss_mode_b             — Swiss-system in Mode B (no fallback)
    5. elo_swiss_mode_b         — Elo-system tournament in Mode B (no fallback)
    6. baseline_adapter         — Greedy decoding on base model + helpfulness adapter
    7. baseline_adapter_softmax — Softmax sampling on base model + helpfulness adapter

Run:
    python evaluation/benchmark_gsi_strategies_helpfulness.py \
        --strategies gsi_softmax swiss elo_swiss swiss_mode_b elo_swiss_mode_b baseline_adapter baseline_adapter_softmax \
        --num-prompts 100 \
        --gsi-n 8 \
        --beta 0.1 \
        --max-tokens 200 \
        --blade helpfulness
"""

import sys
import os
import json
import time
import argparse
import logging
import statistics
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model, load_drafter_model, load_drafter_tokenizer
from Model_mechanics.blades import DPOBlade
from Model_mechanics.gsi_softmax import GSISoftmaxGenerator
from Model_mechanics.swiss import SwissGenerator
from Model_mechanics.elo_swiss import EloSwissGenerator
from Model_mechanics.swiss_mode_b import SwissModeBGenerator
from Model_mechanics.elo_swiss_mode_b import EloSwissModeBGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Greedy Generator
# ─────────────────────────────────────────────────────────────────────────────
class BaselineGreedyGenerator:
    def __init__(self, tokenizer, model, name="baseline"):
        self.tokenizer = tokenizer
        self.model = model
        self.name = name
    def generate(self, prompt: str, max_new_tokens: int, return_stats: bool = False, **kwargs):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        class EmptyStats:
            def to_dict(self):
                return {}
        
        if return_stats:
            return generated_text, EmptyStats()
        return generated_text


class BaselineSoftmaxGenerator:
    def __init__(self, tokenizer, model, name="baseline_softmax", cfg=None):
        self.tokenizer = tokenizer
        self.model = model
        self.name = name
        self.cfg = cfg
    def generate(self, prompt: str, max_new_tokens: int, return_stats: bool = False, **kwargs):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        temp = self.cfg.temperature if self.cfg else 1.0
        top_k = self.cfg.top_k if self.cfg else 50
        top_p = self.cfg.top_p if self.cfg else 0.95
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        class EmptyStats:
            def to_dict(self):
                return {}
        
        if return_stats:
            return generated_text, EmptyStats()
        return generated_text



# ─────────────────────────────────────────────────────────────────────────────
# Steering verification — is the blade actually doing anything?
# ─────────────────────────────────────────────────────────────────────────────

def compute_response_blade_reward(
    dpo_blade: DPOBlade,
    tokenizer,
    prompt: str,
    generated_text: str,
    device,
) -> float:
    """Score a FULL finished response with the same DPO implicit reward."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_ids = tokenizer(
        generated_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    if gen_ids.shape[1] == 0:
        return 0.0
    prefix_mask = torch.ones_like(prompt_ids)
    reward_tensor = dpo_blade.score_reasoning_steps(
        prefix_ids=prompt_ids,
        step_token_ids_list=[gen_ids.squeeze(0)],
    )
    return float(reward_tensor[0].item())


def extract_override_rate(stats_dict: dict) -> float:
    """Pull an override/rejection rate out of whatever stats dict a given generator returned."""
    if "acceptance_rate" in stats_dict:
        accept_rate = stats_dict["acceptance_rate"]
        return round(1.0 - accept_rate, 4) if accept_rate is not None else None
    if "rejected_steps" in stats_dict and "total_steps" in stats_dict and stats_dict["total_steps"]:
        return round(stats_dict["rejected_steps"] / stats_dict["total_steps"], 4)
    return None  # baseline generators have no tournament, no override concept


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def extract_prompt(text: str) -> str:
    """Extracts everything up to and including the last 'Assistant:'"""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:"
    return text


def run_single_strategy(
    strategy_name: str,
    generator,
    test_prompts: list,
    dpo_blade: DPOBlade,
    tokenizer,
    device,
    max_new_tokens: int,
    verbose: bool = False,
) -> dict:
    """Run a single strategy across all prompts, save raw generations, and compute steering diagnostics."""

    print(f"\n{'━' * 70}")
    print(f"  Strategy: {strategy_name}")
    print(f"{'━' * 70}")

    all_responses = []
    all_stats = []
    t_start = time.perf_counter()

    for idx, prompt in enumerate(test_prompts):
        output, stats = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            return_stats=True,
        )
        generated = output[len(prompt):].strip()
        stats_dict = stats.to_dict()

        blade_reward = compute_response_blade_reward(
            dpo_blade, tokenizer, prompt, generated, device,
        )
        override_rate = extract_override_rate(stats_dict)

        all_responses.append({
            "prompt_idx": idx,
            "prompt": prompt[:200],
            "generated": generated,
            "blade_reward": round(blade_reward, 6),
            "override_rate": override_rate,
        })
        all_stats.append(stats_dict)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (idx + 1) % 10 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Log progress after every prompt
        rewards_so_far = [r["blade_reward"] for r in all_responses]
        avg_reward = sum(rewards_so_far) / len(rewards_so_far)
        
        override_rates_so_far = [r["override_rate"] for r in all_responses if r["override_rate"] is not None]
        avg_override = sum(override_rates_so_far) / len(override_rates_so_far) if override_rates_so_far else 0.0
        
        logger.info(
            "[%s] %d/%d | avg_blade_reward=%.5f | last_blade_reward=%.5f | avg_override=%.4f",
            strategy_name, idx + 1, len(test_prompts), avg_reward, blade_reward, avg_override,
        )

    elapsed = time.perf_counter() - t_start

    all_rewards = [r["blade_reward"] for r in all_responses]
    avg_blade_reward = sum(all_rewards) / len(all_rewards)
    std_blade_reward = statistics.pstdev(all_rewards) if len(all_rewards) > 1 else 0.0
    override_rates = [r["override_rate"] for r in all_responses if r["override_rate"] is not None]
    avg_override_rate = sum(override_rates) / len(override_rates) if override_rates else None

    return {
        "strategy": strategy_name,
        "avg_blade_reward": round(avg_blade_reward, 6),
        "std_blade_reward": round(std_blade_reward, 6),
        "avg_override_rate": round(avg_override_rate, 4) if avg_override_rate is not None else None,
        "num_prompts": len(test_prompts),
        "elapsed_s": round(elapsed, 1),
        "responses": all_responses,
        "stats": all_stats,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark GSI strategies head-to-head for Helpfulness",
    )
    p.add_argument("--num-prompts", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--blade", type=str, default="helpfulness",
                    choices=["helpfulness", "harmlessness", "truthfulness"])
    p.add_argument("--gsi-n", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gsi-threshold", type=float, default=-5.0)
    p.add_argument("--swiss-rounds", type=int, default=6)
    p.add_argument("--elo-rounds", type=int, default=6)
    p.add_argument("--elo-temperature", type=float, default=15.0)
    p.add_argument("--gsi-max-step-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-dir", type=str, default="runs/gsi_helpfulness_benchmark/qwen3B")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                    help="If a strategy's *_results.json already exists, skip regenerating.")
    p.add_argument("--use-tilted-elo", action="store_true",
                    help="Use the tilted reward instead of the blended match score inside GSI-Elo.")
    p.add_argument(
        "--strategies", type=str, nargs="+",
        default=["gsi_softmax", "swiss", "elo_swiss", "swiss_mode_b", "elo_swiss_mode_b", "baseline_adapter", "baseline_adapter_softmax"],
        choices=["gsi_softmax", "swiss", "elo_swiss", "swiss_mode_b", "elo_swiss_mode_b", "baseline_adapter", "baseline_adapter_softmax"],
        help="Which strategies to benchmark.",
    )
    # Phase 1 & 2 arguments
    p.add_argument("--no-fallback", action="store_true", help="Disable verifier fallback, running in Mode B.")
    p.add_argument("--sigma-mode", type=str, default="none", choices=["none", "mc_dropout", "log_ratio_proxy"])
    p.add_argument("--sigma-mc-samples", type=int, default=10)
    p.add_argument("--sigma-dropout-p", type=float, default=0.1)
    p.add_argument("--w-tournament", type=float, default=1.0)
    p.add_argument("--w-blade", type=float, default=1.0)
    p.add_argument("--uwo-lambda", type=float, default=0.5)
    p.add_argument("--adaptive-threshold", action="store_true")
    p.add_argument("--threshold-percentile", type=float, default=10.0)
    p.add_argument("--threshold-buffer-size", type=int, default=100)
    p.add_argument("--hard-draw", action="store_true")
    p.add_argument(
        "--probabilistic", action="store_true",
        help="Force Thurstonian CDF for all Elo matches (enables stochastic upsets).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Swiss Knife — GSI Helpfulness Benchmark (no judge)")
    print("=" * 70)
    print(f"  Strategies    : {args.strategies}")
    print(f"  Blade         : {args.blade}")
    print(f"  n (candidates): {args.gsi_n}")
    print(f"  α (mix)       : {args.alpha}")
    print(f"  β (DPO)       : {args.beta}")
    print(f"  threshold     : {args.gsi_threshold}")
    print(f"  # prompts     : {args.num_prompts}")
    print(f"  max tokens    : {args.max_tokens}")
    print(f"  dtype         : {args.dtype}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "auto" if torch.cuda.is_available() else "cpu"

    # ── Load dataset ──────────────────────────────────────────────────
    logger.info("Loading HH-RLHF helpful-base dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")
    dataset = dataset.shuffle(seed=args.seed).select(
        range(min(args.num_prompts, len(dataset)))
    )

    test_prompts = [extract_prompt(row["chosen"]) for row in dataset]
    logger.info(f"Sampled {len(test_prompts)} prompts.")

    # ── Load base model + blade (shared) ──────────────────────────────
    base_cfg = SwissKnifeConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_new_tokens=args.max_tokens,
        dtype=args.dtype,
        device=device,
        generation_mode="gsi_softmax",  # any GSI mode to pass validation
        gsi_n=args.gsi_n,
        gsi_threshold=args.gsi_threshold,
        gsi_max_step_tokens=args.gsi_max_step_tokens,
        swiss_rounds=args.swiss_rounds,
        elo_rounds=args.elo_rounds,
        elo_temperature=args.elo_temperature,
        seed=args.seed,
        use_tilted_elo=args.use_tilted_elo,
        with_fallback=not args.no_fallback,
        sigma_mode=args.sigma_mode,
        sigma_mc_samples=args.sigma_mc_samples,
        sigma_dropout_p=args.sigma_dropout_p,
        w_tournament=args.w_tournament,
        w_blade=args.w_blade,
        uwo_lambda=args.uwo_lambda,
        adaptive_threshold=args.adaptive_threshold,
        threshold_percentile=args.threshold_percentile,
        threshold_buffer_size=args.threshold_buffer_size,
        hard_draw=args.hard_draw,
        probabilistic=args.probabilistic,
    )

    logger.info("Loading shared verifier base model (Qwen 2.5 7B) + blade...")
    tokenizer = load_tokenizer(base_cfg)
    base_model = load_base_model(base_cfg)
    blade_model = load_blade_model(base_cfg, args.blade)

    # ── Load drafter model ────────────────────
    logger.info("Loading drafter...")
    drafter_tokenizer = load_drafter_tokenizer(base_cfg)
    drafter_model = load_drafter_model(base_cfg)

    # Standalone DPOBlade used ONLY for the post-hoc steering diagnostic
    diagnostic_blade = DPOBlade(base_cfg, base_model, blade_model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # ── Run each strategy ─────────────────────────────────────────────
    strategy_generators = {
        "gsi_softmax": lambda cfg: GSISoftmaxGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
        "swiss": lambda cfg: SwissGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
        "elo_swiss": lambda cfg: EloSwissGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
        "swiss_mode_b": lambda cfg: SwissModeBGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
        "elo_swiss_mode_b": lambda cfg: EloSwissModeBGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
        "baseline_adapter": lambda cfg: BaselineGreedyGenerator(tokenizer, blade_model, "baseline_adapter"),
        "baseline_adapter_softmax": lambda cfg: BaselineSoftmaxGenerator(tokenizer, blade_model, "baseline_adapter_softmax", cfg),
    }

    for strat_name in args.strategies:
        out_file_check = os.path.join(args.output_dir, f"{strat_name}_results.json")
        if args.skip_existing and os.path.exists(out_file_check):
            try:
                with open(out_file_check) as f:
                    existing = json.load(f)
                if existing.get("num_prompts") == len(test_prompts):
                    logger.info(
                        "[%s] Found existing results with %d prompts at %s — skipping (--skip-existing).",
                        strat_name, existing["num_prompts"], out_file_check,
                    )
                    all_results[strat_name] = {
                        "avg_blade_reward": existing["avg_blade_reward"],
                        "std_blade_reward": existing["std_blade_reward"],
                        "avg_override_rate": existing["avg_override_rate"],
                        "num_prompts": existing["num_prompts"],
                        "elapsed_s": existing["elapsed_s"],
                    }
                    continue
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("[%s] Existing file unreadable (%s) — regenerating.", strat_name, e)

        # Build strategy-specific config
        cfg = SwissKnifeConfig(
            alpha=args.alpha,
            beta=args.beta,
            max_new_tokens=args.max_tokens,
            dtype=args.dtype,
            device=device,
            generation_mode="gsi_softmax" if strat_name.startswith("baseline") else strat_name,
            gsi_n=args.gsi_n,
            gsi_threshold=args.gsi_threshold,
            gsi_max_step_tokens=args.gsi_max_step_tokens,
            swiss_rounds=args.swiss_rounds,
            elo_rounds=args.elo_rounds,
            elo_temperature=args.elo_temperature,
            seed=args.seed,
            use_tilted_elo=args.use_tilted_elo,
            with_fallback=not args.no_fallback,
            sigma_mode=args.sigma_mode,
            sigma_mc_samples=args.sigma_mc_samples,
            sigma_dropout_p=args.sigma_dropout_p,
            w_tournament=args.w_tournament,
            w_blade=args.w_blade,
            uwo_lambda=args.uwo_lambda,
            adaptive_threshold=args.adaptive_threshold,
            threshold_percentile=args.threshold_percentile,
            threshold_buffer_size=args.threshold_buffer_size,
            hard_draw=args.hard_draw,
            probabilistic=args.probabilistic,
        )

        generator = strategy_generators[strat_name](cfg)
        result = run_single_strategy(
            strategy_name=strat_name,
            generator=generator,
            test_prompts=test_prompts,
            dpo_blade=diagnostic_blade,
            tokenizer=tokenizer,
            device=base_model.device,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose,
        )

        all_results[strat_name] = {
            "avg_blade_reward": result["avg_blade_reward"],
            "std_blade_reward": result["std_blade_reward"],
            "avg_override_rate": result["avg_override_rate"],
            "num_prompts": result["num_prompts"],
            "elapsed_s": result["elapsed_s"],
        }

        # Save per-strategy detailed results
        out_file = os.path.join(args.output_dir, f"{strat_name}_results.json")
        with open(out_file, "w") as f:
            json.dump({
                "strategy": strat_name,
                "config": {
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "blade": args.blade,
                    "gsi_n": args.gsi_n,
                    "gsi_threshold": args.gsi_threshold,
                    "swiss_rounds": args.swiss_rounds,
                    "elo_rounds": args.elo_rounds,
                    "elo_temperature": args.elo_temperature,
                    "max_tokens": args.max_tokens,
                },
                "avg_blade_reward": result["avg_blade_reward"],
                "std_blade_reward": result["std_blade_reward"],
                "avg_override_rate": result["avg_override_rate"],
                "num_prompts": result["num_prompts"],
                "elapsed_s": result["elapsed_s"],
                "responses": result["responses"],
                "stats": result["stats"],
            }, f, indent=2)
        logger.info("Saved %s results → %s", strat_name, out_file)

    # ── Results comparison table ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GSI Helpfulness Benchmark — Steering Diagnostics")
    print("=" * 70)
    print(f"\n  {'Strategy':<22} {'BladeReward':>12} {'Std':>8} {'OverrideRate':>13} {'Time':>8}")
    print(f"  {'─' * 22} {'─' * 12} {'─' * 8} {'─' * 13} {'─' * 8}")

    for strat_name in args.strategies:
        r = all_results[strat_name]
        override_str = f"{r['avg_override_rate']:.4f}" if r["avg_override_rate"] is not None else "n/a"
        print(
            f"  {strat_name:<22} {r['avg_blade_reward']:>12.5f} {r['std_blade_reward']:>8.4f} "
            f"{override_str:>13} "
            f"{r['elapsed_s']:>7.1f}s"
        )

    best = max(all_results.items(), key=lambda x: x[1]["avg_blade_reward"])
    print(f"\n  Highest avg blade reward: {best[0]} ({best[1]['avg_blade_reward']:.5f})")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": args.num_prompts,
            "seed": args.seed,
            "blade": args.blade,
            "alpha": args.alpha,
            "beta": args.beta,
            "gsi_n": args.gsi_n,
            "gsi_threshold": args.gsi_threshold,
            "swiss_rounds": args.swiss_rounds,
            "elo_rounds": args.elo_rounds,
            "elo_temperature": args.elo_temperature,
            "max_tokens": args.max_tokens,
            "dataset": "Anthropic/hh-rlhf:helpful-base",
        },
        "results": all_results,
        "highest_blade_reward_strategy": best[0],
        "note": "No GEval/AQI scoring done here. Run run_geval_only.py against "
                 "this output directory to score these exact saved generations.",
    }
    summary_file = os.path.join(args.output_dir, "gsi_helpfulness_benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
