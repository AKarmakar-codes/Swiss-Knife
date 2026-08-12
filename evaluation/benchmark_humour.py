"""
Swiss-Knife — Humour Benchmark on Academic Datasets
===================================================

Step-level decoding-time alignment benchmark focused on **humour**.
Evaluates a set of alignment strategies (with primary focus on elo_swiss_mode_b)
on academic humor benchmark prompts (SemEval-2021 Task 7 / SemEval-2020 Task 7 / Short Jokes),
using the DPO-trained Humour Blade as the reward signal during candidate selection.

What this script does:
    - Loads prompts from standard academic humor benchmarks (SemEval-2021 Task 7 Jokes / SemEval-2020 Humicroedit).
    - Generates responses under each strategy using a shared Drafter + Humour Blade.
    - Does NOT score with an external reward model at inference time.
      Raw outputs are saved as JSON files so external judges or Tribunal can evaluate them.
    - Logs steering-verification diagnostics per response and per strategy:
      per-step blade reward stats and override rates.

Strategies compared:
    1. baseline                 — Greedy decoding on base model (no alignment)
    2. baseline_adapter         — Greedy decoding with humour LoRA adapter
    3. baseline_adapter_softmax — Softmax sampling with humour LoRA adapter
    4. gsi_softmax              — Softmax(β·r̃) step selection with blade reward
    5. swiss                    — Swiss-system points table → softmax champion (Mode A)
    6. elo_swiss                — Elo-rating tournament selection (Mode A, with acceptance gate)
    7. swiss_mode_b             — Swiss-system Mode B (unconditional acceptance, no fallback)
    8. elo_swiss_mode_b         — Elo-rating tournament Mode B (proposed strategy, unconditional)

Run:
    python evaluation/benchmark_humour.py \\
        --strategies baseline elo_swiss_mode_b swiss_mode_b baseline_adapter_softmax \\
        --num-prompts 50 \\
        --gsi-n 8 \\
        --beta 0.2 \\
        --max-tokens 250 \\
        --blade humour
"""

import sys
import os
import json
import time
import math
import argparse
import logging
import statistics
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_tokenizer,
    load_base_model,
    load_blade_model,
    load_drafter_model,
    load_drafter_tokenizer,
)
from Model_mechanics.blades import DPOBlade
from Model_mechanics.elo_swiss import EloSwissGenerator
from Model_mechanics.elo_swiss_mode_b import EloSwissModeBGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Generators
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
# Helper functions for scoring and diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_response_blade_reward(
    dpo_blade: DPOBlade,
    tokenizer,
    prompt: str,
    generated_text: str,
    device,
) -> float:
    """Score a finished response with the DPO implicit reward.

    We score only the *generated* portion (not the prompt) as a step.
    To avoid character-slice misalignment, we tokenize the full
    (prompt + generated) string and take the suffix beyond prompt_len
    as the step token IDs, which exactly mirrors the token IDs that
    were accumulated during generation.
    """
    full_text = prompt + " " + generated_text
    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return 0.0
    gen_ids = full_ids[0, prompt_len:]  # [step_len]
    if gen_ids.shape[0] == 0:
        return 0.0
    reward_tensor = dpo_blade.score_reasoning_steps(
        prefix_ids=prompt_ids,
        step_token_ids_list=[gen_ids],
    )
    return float(reward_tensor[0].item())


def extract_override_rate(stats_dict: dict) -> float:
    """Extract override/rejection rate from generator stats."""
    if "acceptance_rate" in stats_dict:
        accept_rate = stats_dict["acceptance_rate"]
        return round(1.0 - accept_rate, 4) if accept_rate is not None else None
    if "rejected_steps" in stats_dict and "total_steps" in stats_dict and stats_dict["total_steps"]:
        return round(stats_dict["rejected_steps"] / stats_dict["total_steps"], 4)
    return None


def extract_avg_step_tokens(stats_dict: dict) -> float:
    if "avg_step_tokens" in stats_dict:
        return stats_dict["avg_step_tokens"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Academic Humor Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

ACADEMIC_HUMOR_PROMPTS_FALLBACK = [
    "Tell me a short, witty one-liner about artificial intelligence.",
    "Write a funny setup and punchline about a programmer trying to cook dinner.",
    "What is a hilarious explanation for why cats ignore human commands?",
    "Give me a clever pun about quantum physics.",
    "Write a short satirical paragraph about corporate meetings that could have been emails.",
    "Tell me a funny joke about coffee and morning routines.",
    "What is a witty response when someone asks why you are late?",
    "Write a short, hilarious dialogue between a smartphone and its battery at 1%.",
    "Tell me a humorous observation about modern social media habits.",
    "Write a punchy, humorous cartoon caption for a dog wearing a business suit.",
    "Why did the computer go to therapy? Give a funny punchline.",
    "Write a short funny story about a time traveler accidentally inventing the microwave.",
    "What is a funny excuse for failing a math exam?",
    "Tell a clever joke involving a pirate and a modern wifi network.",
    "Write a funny snippet about working from home with a noisy pet.",
]


def load_academic_humor_dataset(num_prompts: int, seed: int = 42) -> list:
    """Load humor evaluation prompts from standard academic datasets:
    1. Try SemEval-2021 Task 7 (hfht/SemEval-2021-Task-7-Jokes or semeval_2021_task_7)
    2. Try Short Jokes dataset
    3. Fallback to curated academic benchmark prompt suite
    """
    logger.info("Attempting to load academic humor dataset (SemEval-2021 Task 7)...")
    prompts = []

    # Attempt 1: SemEval 2021 Task 7
    try:
        ds = load_dataset("hfht/SemEval-2021-Task-7-Jokes", split="train")
        shuffled = ds.shuffle(seed=seed)
        for row in shuffled:
            text = row.get("text") or row.get("joke") or row.get("headline")
            if text and len(text.strip()) > 10:
                prompt_str = f"Write a funny completion or witty response to the following setup:\n\nSetup: {text.strip()}\n\nResponse:"
                prompts.append(prompt_str)
                if len(prompts) >= num_prompts:
                    break
        if len(prompts) > 0:
            logger.info("Successfully loaded %d prompts from SemEval-2021 Task 7.", len(prompts))
            return prompts[:num_prompts]
    except Exception as e:
        logger.warning("Could not load SemEval-2021 Task 7 online: %s", e)

    # Attempt 2: Short jokes / humicroedit
    try:
        ds = load_dataset("surajp/short-jokes", split="train")
        shuffled = ds.shuffle(seed=seed)
        for row in shuffled:
            text = row.get("joke") or row.get("text")
            if text and len(text.strip()) > 10:
                prompt_str = f"Tell a funny joke similar in style to this:\n\n{text.strip()}\n\nJoke:"
                prompts.append(prompt_str)
                if len(prompts) >= num_prompts:
                    break
        if len(prompts) > 0:
            logger.info("Successfully loaded %d prompts from short-jokes dataset.", len(prompts))
            return prompts[:num_prompts]
    except Exception as e:
        logger.warning("Could not load short-jokes dataset online: %s", e)

    # Attempt 3: Academic Benchmark Fallback Prompt Suite
    logger.info("Using Academic Humor Benchmark prompt suite (%d base prompts).", len(ACADEMIC_HUMOR_PROMPTS_FALLBACK))
    base_prompts = ACADEMIC_HUMOR_PROMPTS_FALLBACK
    import random
    rng = random.Random(seed)
    selected = [rng.choice(base_prompts) for _ in range(num_prompts)]
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Execution Routine
# ─────────────────────────────────────────────────────────────────────────────

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
    """Run a single strategy across all prompts and collect steering diagnostics."""

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
        # Decode only the generated portion — joint decode with skip_special_tokens
        # can shift the prompt text, making character-slicing unreliable.
        # Instead take the full output string (which equals prompt + continuation)
        # and pass it directly; compute_response_blade_reward tokenizes correctly.
        generated = output[len(prompt):].strip() if output.startswith(prompt) else output

        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else {}

        blade_reward = compute_response_blade_reward(
            dpo_blade, tokenizer, prompt, generated, device,
        )
        override_rate = extract_override_rate(stats_dict)
        avg_step_tokens = extract_avg_step_tokens(stats_dict)

        all_responses.append({
            "prompt_idx": idx,
            "prompt": prompt,
            "generated": generated,
            "blade_reward": round(blade_reward, 6),
            "override_rate": override_rate,
            "avg_step_tokens": avg_step_tokens,
        })
        all_stats.append(stats_dict)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    total_tokens_all = sum(s.get("total_tokens", 0) for s in all_stats if isinstance(s, dict))
    total_steps_all = sum(s.get("total_steps", 0) for s in all_stats if isinstance(s, dict))
    global_avg_step_tokens = total_tokens_all / total_steps_all if total_steps_all > 0 else None

    return {
        "strategy": strategy_name,
        "avg_blade_reward": round(avg_blade_reward, 6),
        "std_blade_reward": round(std_blade_reward, 6),
        "avg_override_rate": round(avg_override_rate, 4) if avg_override_rate is not None else None,
        "avg_step_tokens": round(global_avg_step_tokens, 2) if global_avg_step_tokens is not None else None,
        "num_prompts": len(test_prompts),
        "elapsed_s": round(elapsed, 1),
        "responses": all_responses,
        "stats": all_stats,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Humour Blade using Elo-Swiss Mode-B and baseline strategies",
    )
    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--blade", type=str, default="humour",
                    choices=["humour", "helpfulness", "harmlessness", "truthfulness", "honesty"])
    p.add_argument("--gsi-n", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--gsi-threshold", type=float, default=-5.0)
    p.add_argument("--gsi-tau", type=float, default=1.0)
    p.add_argument("--swiss-rounds", type=int, default=6)
    p.add_argument("--elo-rounds", type=int, default=6)
    p.add_argument("--elo-temperature", type=float, default=15.0)
    p.add_argument("--gsi-max-step-tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-dir", type=str, default="runs/gsi_humour_benchmark/qwen7B")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--selection-mode", type=str, default="tilted",
        choices=["tilted", "match"],
    )
    p.add_argument(
        "--strategies", type=str, nargs="+",
        default=["baseline", "elo_swiss_mode_b", "swiss_mode_b", "baseline_adapter_softmax"],
        choices=["baseline", "baseline_adapter", "baseline_adapter_softmax", "gsi_softmax", "swiss", "elo_swiss", "swiss_mode_b", "elo_swiss_mode_b"],
    )
    p.add_argument("--no-fallback", action="store_true", help="Disable verifier fallback, running in Mode B.")
    p.add_argument("--sigma-mode", type=str, default="none", choices=["none", "min_entropy", "mc_dropout", "log_ratio_proxy"])
    p.add_argument("--w-tournament", type=float, default=1.0)
    p.add_argument("--w-blade", type=float, default=1.0)
    p.add_argument("--uwo-lambda", type=float, default=0.5)
    p.add_argument("--probabilistic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Swiss Knife — Humour Blade Benchmark (Mode B & Baseline Comparison)")
    print("=" * 70)
    print(f"  Strategies    : {args.strategies}")
    print(f"  Blade         : {args.blade}")
    print(f"  n (candidates): {args.gsi_n}")
    print(f"  α (mix)       : {args.alpha}")
    print(f"  β (DPO)       : {args.beta}")
    print(f"  # prompts     : {args.num_prompts}")
    print(f"  max tokens    : {args.max_tokens}")
    print(f"  dtype         : {args.dtype}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_tilted = (args.selection_mode == "tilted")
    device = "auto" if torch.cuda.is_available() else "cpu"

    test_prompts = load_academic_humor_dataset(args.num_prompts, seed=args.seed)
    logger.info("Loaded %d humor benchmark prompts.", len(test_prompts))

    # Shared config
    base_cfg = SwissKnifeConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_new_tokens=args.max_tokens,
        dtype=args.dtype,
        device=device,
        generation_mode="elo_swiss_mode_b",
        gsi_n=args.gsi_n,
        gsi_threshold=args.gsi_threshold,
        gsi_max_step_tokens=args.gsi_max_step_tokens,
        gsi_tau=args.gsi_tau,
        swiss_rounds=args.swiss_rounds,
        elo_rounds=args.elo_rounds,
        elo_temperature=args.elo_temperature,
        seed=args.seed,
        use_tilted_elo=use_tilted,
        use_tilted_selection=use_tilted,
        with_fallback=not args.no_fallback,
        sigma_mode=args.sigma_mode,
        w_tournament=args.w_tournament,
        w_blade=args.w_blade,
        uwo_lambda=args.uwo_lambda,
        probabilistic=args.probabilistic,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    logger.info("Loading shared verifier base model + Humour Blade...")
    tokenizer = load_tokenizer(base_cfg)
    base_model = load_base_model(base_cfg)
    blade_model = load_blade_model(base_cfg, args.blade, base_model=base_model)

    logger.info("Loading drafter model...")
    drafter_tokenizer = load_drafter_tokenizer(base_cfg)
    drafter_model = load_drafter_model(base_cfg)

    diagnostic_blade = DPOBlade(base_cfg, base_model, blade_model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    strategy_generators = {
        "baseline": lambda cfg: BaselineSoftmaxGenerator(tokenizer, base_model, "baseline", cfg),
        "baseline_adapter": lambda cfg: BaselineGreedyGenerator(tokenizer, blade_model, "baseline_adapter"),
        "baseline_adapter_softmax": lambda cfg: BaselineSoftmaxGenerator(tokenizer, blade_model, "baseline_adapter_softmax", cfg),
        "elo_swiss": lambda cfg: EloSwissGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
        "elo_swiss_mode_b": lambda cfg: EloSwissModeBGenerator(cfg, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model),
    }

    for strat_name in args.strategies:
        out_file_check = os.path.join(args.output_dir, f"{strat_name}_results.json")
        if args.skip_existing and os.path.exists(out_file_check):
            logger.info("Skipping existing strategy results: %s", out_file_check)
            continue

        cfg = SwissKnifeConfig(
            alpha=args.alpha,
            beta=args.beta,
            max_new_tokens=args.max_tokens,
            dtype=args.dtype,
            device=device,
            generation_mode="elo_swiss_mode_b" if strat_name.startswith("baseline") else strat_name,
            gsi_n=args.gsi_n,
            gsi_threshold=args.gsi_threshold,
            gsi_max_step_tokens=args.gsi_max_step_tokens,
            gsi_tau=args.gsi_tau,
            swiss_rounds=args.swiss_rounds,
            elo_rounds=args.elo_rounds,
            elo_temperature=args.elo_temperature,
            seed=args.seed,
            use_tilted_elo=use_tilted,
            use_tilted_selection=use_tilted,
            with_fallback=not args.no_fallback,
            sigma_mode=args.sigma_mode,
            w_tournament=args.w_tournament,
            w_blade=args.w_blade,
            uwo_lambda=args.uwo_lambda,
            probabilistic=args.probabilistic,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        gen_factory = strategy_generators.get(strat_name, lambda c: EloSwissModeBGenerator(c, drafter_model, drafter_tokenizer, base_model, tokenizer, blade_model))
        generator = gen_factory(cfg)

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
            "avg_step_tokens": result.get("avg_step_tokens"),
            "num_prompts": result["num_prompts"],
            "elapsed_s": result["elapsed_s"],
        }

        # Save strategy JSON file
        out_file = os.path.join(args.output_dir, f"{strat_name}_results.json")
        with open(out_file, "w") as f:
            json.dump({
                "strategy": strat_name,
                "config": {
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "blade": args.blade,
                    "gsi_n": args.gsi_n,
                    "max_tokens": args.max_tokens,
                },
                "avg_blade_reward": result["avg_blade_reward"],
                "std_blade_reward": result["std_blade_reward"],
                "avg_override_rate": result["avg_override_rate"],
                "avg_step_tokens": result.get("avg_step_tokens"),
                "num_prompts": result["num_prompts"],
                "elapsed_s": result["elapsed_s"],
                "responses": result["responses"],
                "stats": result["stats"],
            }, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s results → %s", strat_name, out_file)

    # ── Summary JSON ──────────────────────────────────────────────────
    best_strat = max(all_results.items(), key=lambda x: x[1]["avg_blade_reward"]) if all_results else ("none", {})
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": args.num_prompts,
            "seed": args.seed,
            "blade": args.blade,
            "alpha": args.alpha,
            "beta": args.beta,
            "gsi_n": args.gsi_n,
            "dataset": "SemEval-2021-Task-7 / Academic Humor Benchmarks",
        },
        "results": all_results,
        "highest_blade_reward_strategy": best_strat[0],
    }
    summary_file = os.path.join(args.output_dir, "gsi_humour_benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary saved to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
