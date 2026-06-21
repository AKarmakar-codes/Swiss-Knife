"""
Swiss Knife — GSI Six-Strategy Helpfulness Benchmark
=======================================================

Runs the 3 GSI inference strategies plus 3 baselines head-to-head on the
Anthropic HH-RLHF dataset, evaluating only the helpfulness objective (using 100
sampled prompts). Outputs are scored with an external helpfulness reward model.

Strategies tested:
    1. baseline_base_model  — Greedy Qwen 2.5 7B without any adapter
    2. baseline_adapter     — Greedy Qwen 2.5 7B with helpfulness adapter
    3. original_swiss_knife — Original Swiss Knife (Token-level Swiss system)
    4. gsi_softmax          — Standard GSI: softmax(β·r̃) selection
    5. gsi_pairwise         — Bradley-Terry: P(A wins) = σ(MATCH(A,B)/τ)
    6. gsi_swiss            — Swiss-system → points table → softmax

Methodology:
    1. Load Anthropic/hh-rlhf helpfulness subsets.
    2. Sample 100 prompts (seed=42).
    3. Generate with each strategy.
    4. Score with Ray2333/gpt2-large-helpful-reward_model.

Run on Vast.ai:
    python evaluation/benchmark_gsi_strategies.py \
        --strategies baseline_base_model baseline_adapter original_swiss_knife gsi_softmax gsi_pairwise gsi_swiss \
        --num-prompts 100 \
        --gsi-n 8 \
        --beta 0.1 \
        --max-tokens 200
"""

import sys
import os
import json
import time
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model
from Model_mechanics.gsi_softmax import GSISoftmaxGenerator
from Model_mechanics.gsi_pairwise import GSIPairwiseGenerator
from Model_mechanics.gsi_swiss import GSISwissGenerator
from Model_mechanics.speculative_generator import SwissKnifeSpeculativeGenerator

class BaselineGreedyGenerator:
    """Standard greedy generation from a given model."""
    def __init__(self, tokenizer, target_model, strategy_name="baseline"):
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.strategy_name = strategy_name

    def generate(self, prompt: str, max_new_tokens: int, verbose: bool = False, return_stats: bool = False):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.target_model.device)
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Slice the token tensor to extract only new tokens before decoding
        new_tokens = outputs[0][input_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        class MockStats:
            def __init__(self, name, length):
                self.name = name
                self.length = length
            def to_dict(self):
                return {"strategy": self.name, "total_tokens": self.length}
        
        # Construct full string to preserve downstream script contract
        full_output_string = prompt + generated_text
        
        if return_stats:
            return full_output_string, MockStats(self.strategy_name, outputs.shape[1])
        return full_output_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Model
# ─────────────────────────────────────────────────────────────────────────────

REWARD_MODEL_ID = "Ray2333/gpt2-large-helpful-reward_model"


def get_reward_scorer(device):
    logger.info(f"Loading helpfulness reward model: {REWARD_MODEL_ID}")
    rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID).to(device)
    rm_model.eval()

    def score_helpfulness(prompt: str, response: str) -> float:
        text = prompt + response
        inputs = rm_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.no_grad():
            outputs = rm_model(**inputs)
            score = outputs.logits[0, 0].item()
        return score

    return score_helpfulness


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
    scorer,
    max_new_tokens: int,
    verbose: bool = False,
) -> dict:
    """Run a single strategy across all prompts and return results."""

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

        score = scorer(
            prompt,
            " " + generated if not generated.startswith(" ") else generated,
        )

        all_responses.append({
            "prompt_idx": idx,
            "prompt": prompt[:200],
            "generated": generated,
            "reward_score": score,
        })
        all_stats.append(stats.to_dict())

        if (idx + 1) % 10 == 0 or idx == 0:
            scores_so_far = [r["reward_score"] for r in all_responses]
            avg = sum(scores_so_far) / len(scores_so_far)
            
            # Calculate average for the recent batch of up to 10 questions
            recent_10 = scores_so_far[-10:] if len(scores_so_far) >= 10 else scores_so_far
            avg_10 = sum(recent_10) / len(recent_10)
            
            logger.info(
                "[%s] %d/%d | avg_reward=%.4f | recent_10_avg=%.4f | last_score=%.4f",
                strategy_name, idx + 1, len(test_prompts), avg, avg_10, score,
            )

    elapsed = time.perf_counter() - t_start
    all_scores = [r["reward_score"] for r in all_responses]
    avg_reward = sum(all_scores) / len(all_scores)

    return {
        "strategy": strategy_name,
        "avg_reward_score": round(avg_reward, 4),
        "num_prompts": len(test_prompts),
        "elapsed_s": round(elapsed, 1),
        "responses": all_responses,
        "stats": all_stats,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark all 6 strategies head-to-head",
    )
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--blade", type=str, default="helpfulness",
                    choices=["helpfulness"])
    p.add_argument("--gsi-n", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gsi-threshold", type=float, default=0.0)
    p.add_argument("--gsi-tau", type=float, default=1.0)
    p.add_argument("--swiss-rounds", type=int, default=0)
    p.add_argument("--gsi-max-step-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-dir", type=str, default="runs/gsi_benchmark")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--strategies", type=str, nargs="+",
        default=["baseline_base_model", "baseline_adapter", "original_swiss_knife", "gsi_softmax", "gsi_pairwise", "gsi_swiss"],
        choices=["baseline_base_model", "baseline_adapter", "original_swiss_knife", "gsi_softmax", "gsi_pairwise", "gsi_swiss"],
        help="Which strategies to benchmark (default: all six).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Swiss Knife — GSI Six-Strategy Benchmark")
    print("=" * 70)
    print(f"  Strategies    : {args.strategies}")
    print(f"  Blade         : {args.blade}")
    print(f"  n (candidates): {args.gsi_n}")
    print(f"  α (mix)       : {args.alpha}")
    print(f"  β (DPO)       : {args.beta}")
    print(f"  τ (BT temp)   : {args.gsi_tau}")
    print(f"  threshold     : {args.gsi_threshold}")
    print(f"  # prompts     : {args.num_prompts}")
    print(f"  max tokens    : {args.max_tokens}")
    print(f"  dtype         : {args.dtype}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "auto" if torch.cuda.is_available() else "cpu"
    rm_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load dataset ──────────────────────────────────────────────────
    logger.info("Loading HH-RLHF helpfulness datasets...")
    subsets = ["helpful-base", "helpful-online", "helpful-rejection-sampled"]
    dsets = [load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="test") for subset in subsets]
    dataset = concatenate_datasets(dsets)
    dataset = dataset.shuffle(seed=args.seed).select(
        range(min(args.num_prompts, len(dataset)))
    )

    test_prompts = [extract_prompt(row["chosen"]) for row in dataset]
    logger.info(f"Sampled {len(test_prompts)} prompts.")

    # ── Load reward model ─────────────────────────────────────────────
    scorer = get_reward_scorer(rm_device)

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
        gsi_tau=args.gsi_tau,
        swiss_rounds=args.swiss_rounds,
        seed=args.seed,
    )

    logger.info("Loading shared base model + blade...")
    tokenizer = load_tokenizer(base_cfg)
    base_model = load_base_model(base_cfg)
    blade_model = load_blade_model(base_cfg, args.blade)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # ── Run each strategy ─────────────────────────────────────────────
    strategy_generators = {
        "baseline_base_model": lambda cfg: BaselineGreedyGenerator(tokenizer, base_model, "baseline_base_model"),
        "baseline_adapter": lambda cfg: BaselineGreedyGenerator(tokenizer, blade_model, "baseline_adapter"),
        "original_swiss_knife": lambda cfg: SwissKnifeSpeculativeGenerator(cfg, tokenizer, base_model, blade_model),
        "gsi_softmax": lambda cfg: GSISoftmaxGenerator(cfg, tokenizer, base_model, blade_model),
        "gsi_pairwise": lambda cfg: GSIPairwiseGenerator(cfg, tokenizer, base_model, blade_model),
        "gsi_swiss": lambda cfg: GSISwissGenerator(cfg, tokenizer, base_model, blade_model),
    }

    for strat_name in args.strategies:
        # Build strategy-specific config
        actual_gen_mode = strat_name
        if strat_name == "original_swiss_knife" or strat_name.startswith("baseline_"):
            actual_gen_mode = "option_b"  # to pass config validation
            
        cfg = SwissKnifeConfig(
            alpha=args.alpha,
            beta=args.beta,
            max_new_tokens=args.max_tokens,
            dtype=args.dtype,
            device=device,
            generation_mode=actual_gen_mode,
            K=8,
            gamma=4,
            tournament_mode="swiss",
            gsi_n=args.gsi_n,
            gsi_threshold=args.gsi_threshold,
            gsi_max_step_tokens=args.gsi_max_step_tokens,
            gsi_tau=args.gsi_tau,
            swiss_rounds=args.swiss_rounds,
            seed=args.seed,
        )

        generator = strategy_generators[strat_name](cfg)
        result = run_single_strategy(
            strategy_name=strat_name,
            generator=generator,
            test_prompts=test_prompts,
            scorer=scorer,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose,
        )

        all_results[strat_name] = {
            "avg_reward_score": result["avg_reward_score"],
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
                    "gsi_n": args.gsi_n,
                    "gsi_tau": args.gsi_tau,
                    "gsi_threshold": args.gsi_threshold,
                    "swiss_rounds": args.swiss_rounds,
                    "max_tokens": args.max_tokens,
                },
                "avg_reward_score": result["avg_reward_score"],
                "num_prompts": result["num_prompts"],
                "elapsed_s": result["elapsed_s"],
                "responses": result["responses"],
                "stats": result["stats"],
            }, f, indent=2)
        logger.info("Saved %s results → %s", strat_name, out_file)

    # ── Results comparison table ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GSI Three-Strategy Benchmark Results")
    print("=" * 70)
    print(f"\n  {'Strategy':<25} {'Avg Reward':>12} {'Time':>10}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 10}")

    for strat_name in args.strategies:
        r = all_results[strat_name]
        print(
            f"  {strat_name:<25} {r['avg_reward_score']:>12.4f} "
            f"{r['elapsed_s']:>9.1f}s"
        )

    # Best strategy
    best = max(all_results.items(), key=lambda x: x[1]["avg_reward_score"])
    print(f"\n  🏆 Best: {best[0]} (avg reward = {best[1]['avg_reward_score']:.4f})")
    print(f"  MOD Target (PPO Helpfulness): 1.91")

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
            "gsi_tau": args.gsi_tau,
            "gsi_threshold": args.gsi_threshold,
            "swiss_rounds": args.swiss_rounds,
            "max_tokens": args.max_tokens,
            "reward_model": REWARD_MODEL_ID,
        },
        "results": all_results,
        "best_strategy": best[0],
    }
    summary_file = os.path.join(args.output_dir, "gsi_benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
