"""
benchmark_hhh_ablations.py — Swiss-Knife Multi-Blade Logit Mixing Ablation Suite
===================================================================================

This script executes the component-level ablation analysis for the Swiss-Knife 
multi-blade alignment framework as detailed in Section 5 of the AAAI submission.

Scientific Rationale:
---------------------
Swiss-Knife achieves superior multi-objective steerability and Pareto frontier 
uniformity through three core architectural mechanisms operating during 
decode-time Candidate-Batch Normalization (CBN) and Thurstonian Elo selection:

  1. Candidate-Batch Normalization (CBN, normalize_scores=True):
     Normalizes candidate reward distributions (mu = 0, sigma = 1) across the 
     candidate batch for each active blade prior to convex logit mixing. Epistemic 
     uncertainty sigma is scale-normalized (sigma / std(sigma)). This prevents 
     high-magnitude reward blades from dominating low-magnitude blades, restoring 
     true Pareto steerability across heterogeneous DPO reward scales.

  2. Thurstonian Elo Tournament Selection (w_tournament > 0):
     Runs pairwise Thurstonian Elo matches over candidate batches to construct 
     a calibrated global ranking matrix before champion token selection, rather 
     than relying on naive point-estimate reward softmax sampling.

  3. Uncertainty-Weighted Objective (UWO, uwo_lambda > 0):
     Penalizes candidate reward variance (sigma_i) during step selection:
     UWO_i = z_mu_i - lambda_uwo * z_sigma_i
     This filters out high-reward hallucinated or out-of-distribution candidates 
     that exhibit high epistemic model variance.

Ablation Variants Evaluated:
----------------------------
  - `swiss_full`   : Full Swiss-Knife pipeline (CBN=On, Elo=1.1, UWO=0.2).
  - `swiss_no_cbn` : Disables Candidate-Batch Normalization (normalize_scores=False).
                     Raw unstandardized blade rewards are summed directly, causing 
                     reward-scale imbalances between blades.
  - `swiss_no_elo` : Disables Thurstonian Elo Tournament (w_tournament=0.0).
                     Candidates are selected purely via UWO score softmax.
  - `swiss_no_uwo` : Disables Uncertainty-Weighted Objective (uwo_lambda=0.0).
                     Candidate selection ignores epistemic variance (sigma_i = 0).

Automated Diagnostic Metadata Saved:
-------------------------------------
Every generation step automatically logs diagnostic metadata inside output JSON files:
  - `stats`: total steps, total tokens, acceptance rate, tokens/sec, mean reward.
  - `step_details`: step-by-step diagnostic breakdown containing:
      * `raw_blade_mu_stds`   : raw std of reward logits per blade before CBN
      * `raw_blade_mu_means`  : raw mean of reward logits per blade
      * `logit_scale_ratio`   : ratio of tournament rating std to blade UWO std
      * `tournament_term_std` : standard deviation of Elo tournament logits
      * `blade_term_std`      : standard deviation of UWO blade logits
      * `candidate_mus`       : per-candidate composite rewards
      * `candidate_sigmas`    : per-candidate composite uncertainties

Output Directory Structure:
---------------------------
  runs/hhh_ablations/<ablation_variant>/<config_name>.json
  tribunal/inputs/hhh_ablations/<ablation_variant>__<config_name>.jsonl

Usage:
------
  # Step 1: Run 8-GPU parallel ablation generation across all 4 variants:
  mkdir -p runs/logs

  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python benchmarking/benchmark_hhh_ablations.py \
      --prompts data/hhh_eval_prompts.jsonl \
      --variants swiss_full swiss_no_cbn swiss_no_elo swiss_no_uwo \
      --grid symmetric7 \
      --num-shards 8 \
      --shard-id $i \
      > runs/logs/ablation_shard_$i.log 2>&1 &
  done; wait

  # Step 2: Merge GPU shards into final evaluation JSONL files:
  python benchmarking/benchmark_hhh_ablations.py \
    --prompts data/hhh_eval_prompts.jsonl \
    --variants swiss_full swiss_no_cbn swiss_no_elo swiss_no_uwo \
    --grid symmetric7 \
    --num-shards 8 \
    --merge-shards

  # Step 3: Run G-Eval Tribunal judging:
  python tribunal/run_tribunal_eval.py \
    --input-dir tribunal/inputs/hhh_ablations \
    --output-dir tribunal/eval_results/hhh_ablations \
    --parallel

  # Step 4: Analyze ablation results table & plots:
  python benchmarking/analyze_hhh_pareto.py \
    --summary tribunal/eval_results/hhh_ablations/model_summary.csv \
    --out-dir runs/hhh_ablations/plots
"""

import os
import sys
import json
import logging
import argparse
import itertools
from typing import List, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Insert project root into sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_tokenizer,
    load_base_model,
    load_blade_model,
    load_drafter_model,
    load_drafter_tokenizer,
)
from Model_mechanics.elo_swiss_multi_blade_mode_b import EloSwissMultiBladeModeBGenerator
from benchmarking.configs import get_all_configs, ARCHITECTURAL

HHH_BLADES = ["helpfulness", "honesty", "harmlessness"]


def extract_response(text: str, prompt: str) -> str:
    """Safely extract generated response from model output, stripping prompt preambles and turn markers."""
    if not text:
        return ""
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    elif prompt and prompt in text and text.index(prompt) == 0:
        text = text.replace(prompt, "", 1).strip()
    for marker in ["\n\nAssistant:", "Assistant:", "\n\nUser:", "User:", "\n\nHuman:", "Human:"]:
        if marker in text:
            text = text.rsplit(marker, 1)[-1].strip()
    return text.strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HHHAblations")

ABLATION_VARIANTS = ["swiss_full", "swiss_no_cbn", "swiss_no_elo", "swiss_no_uwo"]


# ── Grid Builder ──────────────────────────────────────────────────────────────

def build_grid(spec: str = "symmetric7", steps: int = 2) -> List[Dict[str, float]]:
    grid, seen = [], set()

    def add(raw):
        total = sum(max(0.0, raw.get(b, 0.0)) for b in HHH_BLADES)
        if total <= 0:
            return
        w = {b: round(max(0.0, raw.get(b, 0.0)) / total, 6) for b in HHH_BLADES}
        residual = round(1.0 - sum(w.values()), 6)
        if residual:
            w[max(w, key=w.get)] = round(w[max(w, key=w.get)] + residual, 6)
        key = tuple(w[b] for b in HHH_BLADES)
        if key not in seen:
            seen.add(key)
            grid.append(w)

    if spec in ("symmetric7", "s7", "7"):
        for b in HHH_BLADES:
            add({b: 1.0})
        for a, b in itertools.combinations(HHH_BLADES, 2):
            add({a: 0.5, b: 0.5})
        add({b: 1.0 / 3 for b in HHH_BLADES})
    elif spec == "edges":
        for a, b in itertools.combinations(HHH_BLADES, 2):
            for i in range(steps + 1):
                t = i / steps
                add({a: 1.0 - t, b: t})
        add({b: 1.0 / 3 for b in HHH_BLADES})
    else:
        raise ValueError(f"Unknown grid spec '{spec}'")
    return grid


def config_name(w: Dict[str, float]) -> str:
    h = int(round(w.get("helpfulness", 0.0) * 100))
    o = int(round(w.get("honesty", 0.0) * 100))
    a = int(round(w.get("harmlessness", 0.0) * 100))
    return f"w_help{h:03d}_hone{o:03d}_harm{a:03d}"


# ── I/O Utilities ─────────────────────────────────────────────────────────────

def load_prompts(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_outputs(responses: List[Dict], variant: str, cfg_name: str, out_root: str, write_jsonl: bool):
    variant_dir = os.path.join(out_root, variant)
    os.makedirs(variant_dir, exist_ok=True)
    with open(os.path.join(variant_dir, f"{cfg_name}.json"), "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "config": cfg_name, "responses": responses}, f, indent=2)

    if write_jsonl:
        trib_dir = os.path.join("tribunal", "inputs", "hhh_ablations")
        os.makedirs(trib_dir, exist_ok=True)
        with open(os.path.join(trib_dir, f"{variant}__{cfg_name}.jsonl"), "w", encoding="utf-8") as f:
            for r in responses:
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"]}) + "\n")


def merge_shards(variants: List[str], grid: List[Dict[str, float]], num_shards: int, out_root: str, prompts_file: str):
    total_expected = len(load_prompts(prompts_file)) if os.path.exists(prompts_file) else None

    for var in variants:
        for w in grid:
            name = config_name(w)
            seen_ids = set()
            all_responses = []
            for s in range(num_shards):
                p = os.path.join(out_root, f"{var}_shard{s}", f"{name}.json")
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    for r in data.get("responses", []):
                        if r["id"] not in seen_ids:
                            seen_ids.add(r["id"])
                            all_responses.append(r)
                else:
                    logger.warning("Missing shard file: %s", p)

            all_responses.sort(key=lambda r: r["id"])
            if total_expected is not None and len(all_responses) != total_expected:
                logger.error(
                    "Shard mismatch for %s/%s: Got %d responses, expected %d!",
                    var, name, len(all_responses), total_expected
                )
            else:
                logger.info("Merged %s/%s — %d responses successfully verified", var, name, len(all_responses))

            write_outputs(all_responses, var, name, out_root, write_jsonl=True)


# ── Model Builders ────────────────────────────────────────────────────────────

def build_ablation_config(variant: str, base_cfg_dict: dict, max_tokens: int, device: str) -> SwissKnifeConfig:
    """Constructs SwissKnifeConfig with variant-specific ablation flags."""
    sk_cfg_vals = base_cfg_dict["swiss"]

    cfg_args = dict(
        max_new_tokens=max_tokens,
        device=device,
        dtype="bfloat16",
        temperature=sk_cfg_vals.get("temperature", 1.0),
        top_p=sk_cfg_vals.get("top_p", 0.95),
        gsi_n=sk_cfg_vals["gsi_n"],
        elo_temperature=sk_cfg_vals["elo_temperature"],
        w_tournament=sk_cfg_vals["w_tournament"],
        w_blade=sk_cfg_vals["w_blade"],
        uwo_lambda=sk_cfg_vals["uwo_lambda"],
        elo_rounds=sk_cfg_vals["elo_rounds"],
        probabilistic=ARCHITECTURAL["probabilistic"],
        normalize_scores=ARCHITECTURAL["normalize"],
        beta=ARCHITECTURAL["beta"],
        sigma_mode=ARCHITECTURAL["sigma_mode"],
    )

    if variant == "swiss_no_cbn":
        cfg_args["normalize_scores"] = False
    elif variant == "swiss_no_elo":
        cfg_args["w_tournament"] = 0.0
    elif variant == "swiss_no_uwo":
        cfg_args["uwo_lambda"] = 0.0

    return SwissKnifeConfig(**cfg_args)


# ── Main Runner ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Swiss-Knife Multi-Blade Logit Mixing Ablation Benchmark")
    p.add_argument("--prompts",      default="data/hhh_eval_prompts.jsonl")
    p.add_argument("--variants",     nargs="+", default=ABLATION_VARIANTS, choices=ABLATION_VARIANTS)
    p.add_argument("--grid",         default="symmetric7", choices=["symmetric7", "edges"])
    p.add_argument("--steps",        type=int, default=2)
    p.add_argument("--max-tokens",   type=int, default=512)
    p.add_argument("--output-root",  default="runs/hhh_ablations")
    p.add_argument("--num-shards",   type=int, default=1)
    p.add_argument("--shard-id",     type=int, default=0)
    p.add_argument("--merge-shards", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    grid = build_grid(args.grid, args.steps)

    if args.merge_shards:
        merge_shards(args.variants, grid, args.num_shards, args.output_root, args.prompts)
        return

    all_prompts = load_prompts(args.prompts)
    prompts = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    logger.info("Ablation Variants: %s | Grid: %s (%d configs) | Shard %d/%d (%d prompts)",
                args.variants, args.grid, len(grid), args.shard_id, args.num_shards, len(prompts))

    cfgs = get_all_configs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_cfg_sample = build_ablation_config("swiss_full", cfgs, args.max_tokens, device)

    logger.info("Loading base tokenizer and model...")
    tokenizer = load_tokenizer(base_cfg_sample)
    base_model = load_base_model(base_cfg_sample)

    logger.info("Loading shared PEFT blade host...")
    blade_host = load_blade_model(base_cfg_sample, "helpfulness", base_model=base_model)
    load_blade_model(base_cfg_sample, "honesty", base_model=blade_host)
    load_blade_model(base_cfg_sample, "harmlessness", base_model=blade_host)

    blade_models_map = {
        "helpfulness": blade_host,
        "honesty": blade_host,
        "harmlessness": blade_host,
    }

    logger.info("Loading drafter tokenizer and model...")
    drafter_tok = load_drafter_tokenizer(base_cfg_sample)
    drafter = load_drafter_model(base_cfg_sample)

    # ── Execute Ablation Variants ─────────────────────────────────────────────
    for variant in args.variants:
        variant_cfg = build_ablation_config(variant, cfgs, args.max_tokens, device)
        is_sharded = args.num_shards > 1
        variant_key = f"{variant}_shard{args.shard_id}" if is_sharded else variant

        generator = EloSwissMultiBladeModeBGenerator(
            cfg=variant_cfg,
            drafter_model=drafter,
            drafter_tokenizer=drafter_tok,
            verifier_model=base_model,
            verifier_tokenizer=tokenizer,
            blade_models=blade_models_map,
        )

        for w in grid:
            name = config_name(w)
            logger.info("[%s] config %s ...", variant, name)
            responses = []

            for p in prompts:
                text, stats = generator.generate(
                    p["prompt"],
                    max_new_tokens=args.max_tokens,
                    return_stats=True,
                    blade_coefficients=w,
                )
                resp = extract_response(text, p["prompt"])
                responses.append({
                    "id":       p["id"],
                    "axis":     p.get("axis", "unknown"),
                    "prompt":   p["prompt"],
                    "response": resp,
                    "stats":    stats if isinstance(stats, dict) else stats.to_dict(),
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            write_outputs(responses, variant_key, name, args.output_root, write_jsonl=not is_sharded)
            logger.info("[%s] %s → %d responses written", variant, name, len(responses))

    logger.info("Ablation benchmark complete.")


if __name__ == "__main__":
    main()
