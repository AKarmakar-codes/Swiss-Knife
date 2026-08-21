"""
build_hhh_dataset.py — 3D HHH Evaluation Prompt Builder
=========================================================

Samples a balanced prompt set across three HHH objectives and writes them
to a single JSONL file. Every strategy and every weight vector reads this
same file so per-prompt results are paired and bootstrap CIs are valid.

Dataset composition:
  helpfulness  — Anthropic/hh-rlhf  helpful-base  (test)
  harmlessness — Anthropic/hh-rlhf  harmless-base (test)
  honesty      — truthful_qa        generation    (validation)

Usage:
  python benchmarking/build_hhh_dataset.py --per-axis 40
  python benchmarking/build_hhh_dataset.py --per-axis 40 --out data/hhh_eval_prompts.jsonl
"""

import os
import sys
import json
import random
import logging
import argparse
from typing import List, Dict

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("BuildHHH")


def _load(*args, **kwargs):
    """Load dataset, falling back to local cache if offline."""
    try:
        return load_dataset(*args, **kwargs)
    except Exception as e:
        logger.warning("Online load failed (%s). Trying local cache.", e)
        return load_dataset(*args, download_mode="reuse_dataset_if_exists", **kwargs)


def _extract_hh_prompt(text: str) -> str:
    """Strip the reference response, keep everything up to 'Assistant:'."""
    parts = text.rsplit("\n\nAssistant:", 1)
    return (parts[0] + "\n\nAssistant:") if len(parts) == 2 else text.strip()


def _truthfulqa_prompt(question: str) -> str:
    return f"\n\nHuman: {question.strip()}\n\nAssistant:"


def build_prompt_set(per_axis: int, seed: int) -> List[Dict]:
    prompts, idx = [], 0

    # Helpfulness
    ds = _load("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")
    idxs = random.Random(seed).sample(range(len(ds)), min(per_axis, len(ds)))
    for i in idxs:
        prompts.append({"id": idx, "axis": "helpfulness", "prompt": _extract_hh_prompt(ds[i]["chosen"])})
        idx += 1

    # Harmlessness
    ds = _load("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    idxs = random.Random(seed + 1).sample(range(len(ds)), min(per_axis, len(ds)))
    for i in idxs:
        prompts.append({"id": idx, "axis": "harmlessness", "prompt": _extract_hh_prompt(ds[i]["chosen"])})
        idx += 1

    # Honesty
    ds = _load("truthfulqa/truthful_qa", "generation", split="validation")
    idxs = random.Random(seed + 2).sample(range(len(ds)), min(per_axis, len(ds)))
    for i in idxs:
        prompts.append({"id": idx, "axis": "honesty", "prompt": _truthfulqa_prompt(ds[i]["question"])})
        idx += 1

    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-axis", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/hhh_eval_prompts.jsonl")
    args = p.parse_args()

    prompts = build_prompt_set(args.per_axis, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %d prompts → %s", len(prompts), args.out)


if __name__ == "__main__":
    main()
