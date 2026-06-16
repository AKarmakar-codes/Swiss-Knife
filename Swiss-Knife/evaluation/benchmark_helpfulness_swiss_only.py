"""
Swiss Knife — Helpfulness Benchmark (Swiss System Only - 100 Prompts)
=====================================================================

Evaluates Option B (Swiss-system speculative loop) for Helpfulness.
Replicates the MOD (Multi-Objective Decoding) evaluation methodology for a subset of 100 prompts.

Comparison:
  • helpfulness_blade  — α=0.5 → Swiss tournament with helpfulness blade

Methodology:
  1. Loads the official Anthropic/hh-rlhf test split.
  2. Samples 100 prompts uniformly (seed=42).
  3. Generates responses.
  4. Scores responses using Ray2333/gpt2-large-helpful-reward_model.

Run on Vast.ai:
    python evaluation/benchmark_helpfulness_swiss_only.py
"""

import sys
import os
import json
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model
from Model_mechanics.speculative_generator import SwissKnifeSpeculativeGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_NEW_TOKENS = 200
NUM_PROMPTS    = 100
OUTPUT_DIR     = "runs/helpfulness_benchmark_swiss_only"
REWARD_MODEL_ID = "Ray2333/gpt2-large-helpful-reward_model"

CONDITIONS = [
    {
        "name":  "helpfulness_blade",
        "label": "Helpfulness Blade (Swiss α=0.5)",
        "alpha": 0.5,
        "blade": "helpfulness",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Reward Model Scoring
# ─────────────────────────────────────────────────────────────────────────────

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
            max_length=1024
        ).to(device)
        
        with torch.no_grad():
            outputs = rm_model(**inputs)
            # The RM outputs a scalar logit
            score = outputs.logits[0, 0].item()
        return score

    return score_helpfulness

# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def extract_prompt(text: str) -> str:
    """Extracts everything up to and including the last 'Assistant:'"""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:"
    return text

def run_benchmark():
    print("=" * 70)
    print("  Swiss Knife — Helpfulness Benchmark (Swiss System Only)")
    print(f"  Evaluating on {NUM_PROMPTS} Prompts")
    print("=" * 70)

    device = "auto" if torch.cuda.is_available() else "cpu"
    dtype  = "bfloat16" if torch.cuda.is_available() else "float32"
    rm_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading HH-RLHF dataset (test split)...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    
    # Shuffle with seed 42 and select 100 prompts to match MOD methodology
    dataset = dataset.shuffle(seed=42).select(range(min(NUM_PROMPTS, len(dataset))))
    
    test_prompts = []
    for row in dataset:
        full_text = row["chosen"]
        prompt = extract_prompt(full_text)
        test_prompts.append(prompt)
    
    logger.info(f"Sampled {len(test_prompts)} prompts for evaluation.")

    # Load Reward Model
    scorer = get_reward_scorer(rm_device)

    # ── Load base model once ─────────────
    base_cfg = SwissKnifeConfig(
        K=8, gamma=4, beta=0.1,
        tournament_mode="swiss", swiss_rounds=3,
        generation_mode="option_b", normalize_scores=True,
        max_new_tokens=MAX_NEW_TOKENS, device=device, dtype=dtype,
        alpha=0.5,
    )
    logger.info("Loading tokenizer + base model...")
    tokenizer  = load_tokenizer(base_cfg)
    base_model = load_base_model(base_cfg)

    logger.info("Loading helpfulness blade...")
    blade_model = load_blade_model(base_cfg, "helpfulness")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    # ── Run condition ───────────────────────────────────────────────
    for cond in CONDITIONS:
        print("\n" + "━" * 70)
        print(f"  Condition: {cond['label']}")
        print(f"  α = {cond['alpha']}  |  blade = {cond['blade']}")
        print("━" * 70)

        cfg = SwissKnifeConfig(
            K=8, gamma=4, beta=0.1,
            tournament_mode="swiss", swiss_rounds=3,
            generation_mode="option_b", normalize_scores=True,
            max_new_tokens=MAX_NEW_TOKENS, device=device, dtype=dtype,
            alpha=cond["alpha"],
        )

        generator = SwissKnifeSpeculativeGenerator(
            cfg=cfg,
            tokenizer=tokenizer,
            base_model=base_model,
            blade_model=blade_model,
        )

        all_responses = []
        t_start       = time.perf_counter()

        for idx, prompt in enumerate(test_prompts):
            output    = generator.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
            generated = output[len(prompt):].strip()
            
            score = scorer(prompt, " " + generated if not generated.startswith(" ") else generated)

            all_responses.append({
                "prompt":       prompt,
                "generated":    generated,
                "reward_score": score,
            })

            if (idx + 1) % 10 == 0 or idx == 0:
                all_scores = [r["reward_score"] for r in all_responses]
                avg        = sum(all_scores) / len(all_scores)
                logger.info(
                    "[%s] %d/%d | avg_reward=%.4f | last_score=%.4f",
                    cond["name"], idx + 1, len(test_prompts), avg, score,
                )

        elapsed    = time.perf_counter() - t_start
        all_scores = [r["reward_score"] for r in all_responses]
        avg_reward = sum(all_scores) / len(all_scores)

        all_results[cond["name"]] = {
            "label":            cond["label"],
            "alpha":            cond["alpha"],
            "avg_reward_score": round(avg_reward, 4),
            "num_prompts":      len(test_prompts),
            "elapsed_s":        round(elapsed, 1),
        }

        out_file = os.path.join(OUTPUT_DIR, f"helpfulness_{cond['name']}.json")
        with open(out_file, "w") as f:
            json.dump(all_responses, f, indent=2)
        logger.info("Saved responses → %s", out_file)

    # ── Results table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Helpfulness Benchmark Results (Swiss System Only)")
    print("=" * 70)
    print(f"\n  {'Condition':<35} {'Avg Reward':>12} {'Time':>10}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 10}")
    
    for cond in CONDITIONS:
        r = all_results[cond["name"]]
        print(
            f"  {r['label']:<35} {r['avg_reward_score']:>12.4f} "
            f"{r['elapsed_s']:>9.1f}s"
        )

    print(f"\n  MOD Target (PPO Helpfulness): 1.91")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": NUM_PROMPTS,
            "seed": 42,
            "reward_model": REWARD_MODEL_ID,
        },
        "results":      all_results,
    }
    summary_file = os.path.join(OUTPUT_DIR, "helpfulness_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
