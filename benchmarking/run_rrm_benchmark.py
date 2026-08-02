"""
RRM Benchmarking Script for Tribunal
====================================

This script benchmarks the official Reward Reasoning Model (RRM) strategy on 
the Anthropic HH-RLHF harmlessness test split with a fixed random seed of 42.

Pipeline steps:
1. Loads the Anthropic HH-RLHF harmless-base test dataset.
2. Applies deterministic shuffle with seed=42 and extracts prompts.
3. Runs the official RRM strategy (CoT reasoning + Knockout Tournament selection).
4. Formats and writes generated outputs to `tribunal/inputs/harmlessness/rrm.jsonl`.
5. Triggers Tribunal judge evaluation, saving results to `tribunal/eval_results/harmlessness/rrm_eval.csv`.

Usage:
    python benchmarking/run_rrm_benchmark.py --num-prompts 50 --seed 42 --gsi-n 4
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model
from benchmarking.strategies.rrm import RRMGenerator, parse_rrm_judgment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_rrm_benchmark")


def extract_prompt(text: str) -> str:
    """Extract prompt string ending with 'Assistant:'."""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:"
    return text


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark official RRM strategy on Tribunal (HH harmlessness test, seed 42)")
    p.add_argument("--num-prompts", type=int, default=50, help="Number of prompts to evaluate.")
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens per response (default 512).")
    p.add_argument("--gsi-n", type=int, default=4, help="Number of candidate responses per prompt.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for dataset split & generation.")
    p.add_argument("--rrm-model-id", type=str, default=None, help="Optional HuggingFace model ID for dedicated RRM judge (e.g. Reward-Reasoning/RRM-7B). Defaults to base model if omitted.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--input-dir", type=str, default="tribunal/inputs/harmlessness")
    p.add_argument("--eval-dir", type=str, default="tribunal/eval_results/harmlessness")
    p.add_argument("--run-eval", action="store_true", default=True, help="Run Tribunal judge evaluation after generation.")
    p.add_argument("--no-eval", action="store_false", dest="run_eval", help="Skip Tribunal judge evaluation.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("  Official Reward Reasoning Model (RRM) — Tribunal Benchmark")
    print("=" * 80)
    print(f"  Dataset       : Anthropic/hh-rlhf (harmless-base, test split)")
    print(f"  Seed          : {args.seed}")
    print(f"  # Prompts     : {args.num_prompts}")
    print(f"  Candidates N  : {args.gsi_n}")
    print(f"  Max Tokens    : {args.max_tokens}")
    print(f"  RRM Model ID  : {args.rrm_model_id or 'Base Model (Qwen2.5-7B SFT)'}")
    print(f"  Input Target  : {args.input_dir}/rrm.jsonl")
    print(f"  Eval Target   : {args.eval_dir}/rrm_eval.csv")
    print("=" * 80)

    # 1. Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load dataset
    logger.info("Loading HH-RLHF harmless-base test split (seed=%d)...", args.seed)
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    except Exception as e:
        logger.warning("Online load failed: %s. Attempting offline cache load...", e)
        from datasets import DownloadConfig
        dataset = load_dataset(
            "Anthropic/hh-rlhf", data_dir="harmless-base", split="test",
            download_config=DownloadConfig(local_files_only=True),
        )

    # Deterministic shuffle with seed=42
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.num_prompts, len(dataset))))
    test_prompts = [extract_prompt(row["chosen"]) for row in dataset]
    logger.info("Successfully loaded and sampled %d test prompts.", len(test_prompts))

    # 3. Load model & tokenizer
    cfg = SwissKnifeConfig(
        max_new_tokens=args.max_tokens,
        dtype=args.dtype if device == "cuda" else "float32",
        device=device,
        gsi_n=args.gsi_n,
        seed=args.seed,
    )
    cfg.candidate_batch_size = args.gsi_n
    if args.rrm_model_id:
        cfg.rrm_model_id = args.rrm_model_id

    logger.info("Loading base candidate generator model and tokenizer...")
    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg)

    # Dedicated RRM evaluator model (if specified)
    if args.rrm_model_id and args.rrm_model_id != cfg.base_model_id:
        logger.info("Loading dedicated RRM evaluator model: %s...", args.rrm_model_id)
        from Model_mechanics.models import load_rrm_model
        evaluator_model = load_rrm_model(cfg)
    else:
        evaluator_model = base_model

    # 4. Instantiate RRM Generator
    generator = RRMGenerator(
        cfg=cfg,
        tokenizer=tokenizer,
        base_model=base_model,
        evaluator_model=evaluator_model,
    )

    # 5. Run generation across prompts
    os.makedirs(args.input_dir, exist_ok=True)
    jsonl_output_path = os.path.join(args.input_dir, "rrm.jsonl")

    logger.info("Generating responses using RRM Knockout Tournament...")
    records = []
    t_start = time.perf_counter()

    for idx, prompt in enumerate(test_prompts):
        logger.info("-" * 80)
        logger.info("Starting Prompt [%d/%d]...", idx + 1, len(test_prompts))
        winning_response, stats = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            return_stats=True,
            verbose=True,
        )

        record = {
            "id": idx,
            "prompt": prompt,
            "response": winning_response,
            "strategy": "rrm",
            "stats": stats.to_dict(),
        }
        records.append(record)

        prompt_clean = prompt.replace('\n', ' ').strip()
        resp_clean = winning_response.replace('\n', ' ').strip()
        logger.info(
            "COMPLETED Prompt [%d/%d] | Matches: %d | Time: %.2fs",
            idx + 1, len(test_prompts), stats.matches_played, stats.total_time_s
        )
        logger.info("  Prompt Snippet : %s...", prompt_clean[:100])
        logger.info("  Winning Output : %s...", resp_clean[:120])
        logger.info("-" * 80)

    total_time = time.perf_counter() - t_start
    logger.info("Generation complete in %.2f seconds.", total_time)

    # Write outputs to tribunal input JSONL
    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved RRM tribunal input file → %s", jsonl_output_path)

    # 6. Run Tribunal evaluation
    if args.run_eval:
        logger.info("Initiating Tribunal judge evaluation...")
        try:
            from tribunal.tribunal.config import CONFIG
            from tribunal.tribunal import pipeline

            os.makedirs(args.eval_dir, exist_ok=True)
            CONFIG["input_path"] = jsonl_output_path
            CONFIG["output_folder"] = args.eval_dir

            pipeline.run(CONFIG)
            logger.info("Tribunal evaluation finished! Results written to %s", args.eval_dir)
        except Exception as e:
            logger.error("Tribunal evaluation failed: %s", e)
            print("\nNote: You can run tribunal manually using:")
            print(f"  python -m tribunal.tribunal.run_eval --input {jsonl_output_path} --output {args.eval_dir}\n")

    print("\n" + "=" * 80)
    print("  RRM Benchmark Execution Complete")
    print("=" * 80)
    print(f"  Input JSONL : {jsonl_output_path}")
    print(f"  Eval Folder : {args.eval_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
