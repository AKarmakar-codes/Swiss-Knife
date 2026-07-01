"""
Benchmark: Stochastic Token-Level Speculative Tournament Decoding
=================================================================
LLaMA 3.2 3B (drafter) + Qwen 2.5 7B (verifier + DPO blade).

Runs 4 strategies on the Anthropic HH-RLHF harmlessness test split:
  1. baseline_argmax_harmlessness  (mode=None  — no stochastic perturbations)
  2. stochastic_knockout_dropout   (mode='mc_dropout')
  3. stochastic_knockout_proj      (mode='random_proj')
  4. stochastic_knockout_subsample (mode='head_subsample')

Output JSON format is identical to benchmark_stochastic_strategies_knockout_bracket.py
so that Tribunal and prepare_tribunal_eval.py can consume it unchanged.

Usage example
-------------
python evaluation/benchmark_stochastic_llama\\&qwen.py \\
    --num-prompts 15 \\
    --gamma 4 \\
    --gsi-n 8 \\
    --alpha 0.5 \\
    --beta 0.1 \\
    --gsi-threshold 0.0 \\
    --output-dir runs/stochastic_llama_qwen_benchmark

Threshold selection note
------------------------
The threshold u (--gsi-threshold) determines how often the draft winner is
accepted vs. rejected in favour of a Qwen fallback.

  u = 0.0   (default): accept whenever the tilted reward is non-negative.
             Permissive — biases toward accepting drafter candidates.
             Good starting point for harmlessness tasks.

  u < 0.0 : even more permissive (accept almost everything from the drafter).
  u > 0.0 : more conservative (require positively-tilted reward to accept).
             Increases fallback rate and Qwen influence.

  Empirical tuning: start at 0.0, observe `rejected_positions / total` in stats.
  If rejection rate is <5%  →  try u = 0.5.
  If rejection rate is >50% →  try u = -0.5 or reduce beta.

  The GSI paper (Algorithm 1) treats u as a hyperparameter tuned per task.
"""

import sys
import os
import gc
import json
import time
import argparse
import logging
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model
from Model_mechanics.stochastic_auditor import StochasticAuditorConfig

# The '&' in the filename requires dynamic import
spec = importlib.util.spec_from_file_location(
    "stoch_lq",
    os.path.join(
        os.path.dirname(__file__), "..", "Model_mechanics", "stochastic_llama&qwen.py"
    ),
)
stoch_lq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stoch_lq)
StochasticLlamaQwenGenerator = stoch_lq.StochasticLlamaQwenGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy registry  (name, stoch_mode)
# stoch_mode = None  → deterministic baseline (no auditor hooks)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    ("baseline_argmax_harmlessness",  None),
    ("stochastic_knockout_dropout",   "mc_dropout"),
    ("stochastic_knockout_proj",      "random_proj"),
    ("stochastic_knockout_subsample", "head_subsample"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_prompt(text: str) -> str:
    """Strip the chosen response, keeping only the conversation prompt."""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark stochastic LLaMA+Qwen tournament decoding."
    )
    p.add_argument("--num-prompts",    type=int,   default=15,
                   help="Number of prompts per strategy (default: 15)")
    p.add_argument("--max-tokens",     type=int,   default=200,
                   help="Max new Qwen tokens per prompt (default: 200)")
    p.add_argument("--blade",          type=str,   default="harmlessness",
                   choices=["helpfulness", "harmlessness", "truthfulness"])
    p.add_argument("--gamma",          type=int,   default=4,
                   help="Draft lookahead depth γ (default: 4)")
    p.add_argument("--gsi-n",          type=int,   default=8,
                   help="Top-K candidates per draft position (default: 8)")
    p.add_argument("--alpha",          type=float, default=0.5,
                   help="Tournament mixing α: 0→blade only, 1→verifier only (default: 0.5)")
    p.add_argument("--beta",           type=float, default=0.1,
                   help="DPO reward scaling β (default: 0.1)")
    p.add_argument("--gsi-threshold",  type=float, default=0.0,
                   help=(
                       "Tilted-reward acceptance threshold u. "
                       "Winner accepted iff r̃(w) >= u. "
                       "0.0 = accept non-negative tilted reward. "
                       "See module docstring for tuning guidance."
                   ))
    p.add_argument("--dtype",          type=str,   default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-dir",     type=str,   default="runs/stochastic_llama_qwen_benchmark")
    p.add_argument("--skip-existing",  action="store_true",
                   help="Skip a strategy if its output JSON already exists.")
    p.add_argument("--strategies", nargs="+",
                   default=[s for s, _ in STRATEGIES],
                   choices=[s for s, _ in STRATEGIES],
                   help="Subset of strategies to run (default: all).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("Loading HH-RLHF harmlessness test split...")
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    test_prompts = [
        extract_prompt(ds[i]["chosen"]) for i in range(args.num_prompts)
    ]
    logger.info("Loaded %d prompts.", len(test_prompts))

    # ── Config ───────────────────────────────────────────────────────────────
    torch_dtype = getattr(torch, args.dtype)

    cfg = SwissKnifeConfig(
        max_new_tokens=args.max_tokens,
        temperature=1.0,
        gsi_n=args.gsi_n,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        gsi_threshold=args.gsi_threshold,
    )

    # ── Models — load once, reuse across strategies ───────────────────────────
    logger.info("Loading Qwen 2.5 7B verifier (SFT-merged base)...")
    qwen_tok   = load_tokenizer(cfg)
    qwen_model = load_base_model(cfg)
    # The config already moves it to cfg.device (default cuda:0) but we can ensure it's on cuda:0
    qwen_model.to(torch_dtype).to("cuda:0")

    logger.info("Loading LLaMA 3.2 3B drafter...")
    llama_tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    if llama_tok.pad_token is None:
        llama_tok.pad_token = llama_tok.eos_token
        llama_tok.pad_token_id = llama_tok.eos_token_id
    llama_tok.padding_side = "left"
    
    llama_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", 
        torch_dtype=torch_dtype, 
        device_map="cuda:0"
    )

    logger.info("Loading %s blade adapter...", args.blade)
    blade_model = load_blade_model(cfg, args.blade)

    # ── Strategy loop ─────────────────────────────────────────────────────────
    active_strategies = [(n, m) for n, m in STRATEGIES if n in args.strategies]

    for strategy_name, stoch_mode in active_strategies:
        out_path = os.path.join(args.output_dir, f"{strategy_name}_results.json")

        if args.skip_existing and os.path.exists(out_path):
            logger.info("Skipping %s (output exists).", strategy_name)
            continue

        logger.info("=" * 70)
        logger.info("  Strategy: %s", strategy_name)
        logger.info("  Threshold u = %.3f", args.gsi_threshold)
        logger.info("=" * 70)

        auditor_cfg = StochasticAuditorConfig(
            mode=stoch_mode,
            harmlessness_only=True,
        )

        generator = StochasticLlamaQwenGenerator(
            cfg=cfg,
            drafter_model=llama_model,
            drafter_tokenizer=llama_tok,
            verifier_model=qwen_model,
            verifier_tokenizer=qwen_tok,
            blade_model=blade_model,
            auditor_cfg=auditor_cfg,
        )

        all_responses = []
        t_start = time.perf_counter()

        for idx, prompt in enumerate(test_prompts):
            try:
                output, stats = generator.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    verbose=args.verbose,
                    return_stats=True,
                )
                generated = (
                    output[len(prompt):].strip()
                    if output.startswith(prompt)
                    else output.strip()
                )

                stats_dict = stats.to_dict()
                override_rate = round(1.0 - stats_dict["acceptance_rate"], 4)

                all_responses.append({
                    "prompt_idx":   idx,
                    "prompt":       prompt,
                    "generated":    generated,
                    "override_rate": override_rate,
                    "tilted_reward_mean": stats_dict["mean_tilted_reward"],
                    "acceptance_rate":    stats_dict["acceptance_rate"],
                })

                logger.info(
                    "[%s] %d/%d | tokens=%d | acc=%.2f | mean_r̃=%.4f",
                    strategy_name, idx + 1, len(test_prompts),
                    stats_dict["total_tokens"],
                    stats_dict["acceptance_rate"],
                    stats_dict["mean_tilted_reward"],
                )

            except Exception as exc:
                logger.error("[%s] Prompt %d failed: %s", strategy_name, idx, exc, exc_info=True)
                all_responses.append({
                    "prompt_idx": idx,
                    "prompt":     prompt,
                    "generated":  "",
                    "override_rate": None,
                    "error": str(exc),
                })

            if (idx + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t_start
        logger.info("[%s] Finished in %.1fs", strategy_name, elapsed)

        result_obj = {
            "strategy":    strategy_name,
            "num_prompts": len(test_prompts),
            "elapsed_s":   round(elapsed, 1),
            "hyperparams": {
                "gamma":         args.gamma,
                "K":             args.gsi_n,
                "alpha":         args.alpha,
                "beta":          args.beta,
                "threshold_u":   args.gsi_threshold,
                "stoch_mode":    stoch_mode,
            },
            "responses": all_responses,
        }

        with open(out_path, "w") as f:
            json.dump(result_obj, f, indent=2)
        logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
