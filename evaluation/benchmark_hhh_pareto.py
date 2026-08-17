"""
Swiss-Knife — HHH Pareto-Frontier Benchmark (MOD-comparable)
=============================================================

Runs the three-objective (Helpfulness / Honesty / Harmlessness) steering
experiment in the exact experimental shape MOD uses (Shi et al., 2024,
"Decoding-Time Language Model Alignment with Multiple Objectives"), so our
Pareto frontiers are plottable against theirs.

WHAT MOD DOES vs WHAT WE DO
---------------------------
MOD mixes the K aligned policies at the TOKEN level, in log-space:

    log p_MOD(y_t | ·)  =  Σ_k w_k · log π_k(y_t | ·)                 (reverse-KL / JSD)

Because Σ_k w_k = 1, that identity rearranges to

    log p_MOD  =  log π_ref  +  Σ_k w_k · [log π_k − log π_ref]
                = log π_ref  +  Σ_k w_k · β⁻¹ r_k(y_t)

i.e. MOD is a convex combination of exactly the same DPO implicit rewards our
blades expose — mixed per token, over the full vocabulary.

Swiss-Knife mixes the same implicit rewards at the STEP level, over N drafted
candidates, after Candidate-Batch Normalization:

    μ̂_i^(k) = znorm_batch(μ_i^(k)),   μ_i = Σ_k w_k μ̂_i^(k)

The two methods therefore take the SAME control input w and produce the SAME
kind of object (a frontier over w), which is what makes the comparison fair.

The substantive difference this benchmark is designed to measure: MOD's per-token
log-ratios are commensurable in *units* (nats) but not in *magnitude* — a blade
trained on a sharper preference distribution has systematically larger |Δ log p|,
so equal w does not mean equal influence, and the frontier bunches up near the
dominant blade. CBN removes exactly that scale asymmetry. So we report not only
frontier hypervolume (MOD's "area") but frontier UNIFORMITY, which MOD's own
paper names as the second thing frontier area is supposed to reflect
(§4.1, "optimality and uniformity of the solution distribution").

METHODS
-------
    swiss   Swiss-Knife multi-blade Mode B — CBN + Thurstonian Elo tournament (ours)
    mod     MOD token-level log-prob mixing, memory-shared reimplementation
    rs      Rewarded Soups — LoRA parameter merging at weights w (MOD's main baseline)
    base    Frozen SFT backbone, no steering (frontier origin)

All four read the SAME frozen prompt file and are driven by the SAME weight grid,
so every point on every frontier is paired by prompt id.

OUTPUTS
-------
    runs/hhh_pareto/<method>/<config>.json            rich: responses + stats + step details
    tribunal/inputs/hhh_pareto/<method>__<config>.jsonl   judge-ready (id/prompt/response)

Both carry `prompt_set_sha256`; scoring scripts refuse to mix hashes.

Run:
    # 0. freeze the prompt set once
    python evaluation/build_hhh_prompt_set.py --per-axis 40

    # 1. ours, on the 3 pairwise edges + centroid (13 configs)
    python evaluation/benchmark_hhh_pareto.py --methods swiss --grid edges

    # 2. baselines on the MOD-comparable edge only (helpfulness↔harmlessness)
    python evaluation/benchmark_hhh_pareto.py --methods mod rs --grid edge:helpfulness:harmlessness

    # 3. same thing across 8 GPUs (prompt-sharded; run one per GPU)
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python evaluation/benchmark_hhh_pareto.py \
        --methods swiss --grid edges --num-shards 8 --shard-id $i \
        > runs/logs/pareto_shard_$i.log 2>&1 &
    done; wait
    python evaluation/benchmark_hhh_pareto.py --merge-shards --methods swiss --grid edges
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
import itertools
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from peft import PeftModel

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_tokenizer,
    load_base_model,
    load_blade_model,
    load_drafter_model,
    load_drafter_tokenizer,
)
from Model_mechanics.blades import DPOBlade
from Model_mechanics.elo_swiss_multi_blade_mode_b import EloSwissMultiBladeModeBGenerator

from evaluation.build_hhh_prompt_set import load_prompt_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("HHHPareto")

HHH_BLADES = ["helpfulness", "honesty", "harmlessness"]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Weight grids
# ═════════════════════════════════════════════════════════════════════════════

def _cfg_name(w: Dict[str, float]) -> str:
    """Stable, filesystem-safe config name from a weight vector."""
    parts = [f"{b[:4]}{int(round(w.get(b, 0.0) * 100)):03d}" for b in HHH_BLADES]
    return "w_" + "_".join(parts)


def build_grid(spec: str, steps: int, blades: List[str]) -> List[Dict[str, float]]:
    """Construct the weight grid.

    'edges'                       — the 3 pairwise edges of the HHH simplex at
                                    `steps`+1 points each, plus the centroid.
                                    Vertices are shared, so 3*(steps-1) + 3 + 1
                                    unique configs (13 at steps=4).
    'edge:<a>:<b>'                — one edge only, `steps`+1 points. This is the
                                    MOD-comparable 2-objective sweep, matching
                                    their w ∈ {i/10} sweep shape at coarser step.
    'simplex'                     — full simplex lattice at resolution `steps`
                                    (15 configs at steps=4; grows fast).
    'vertices'                    — the 3 single-blade endpoints only.
    """
    grid: List[Dict[str, float]] = []
    seen = set()

    def add(w: Dict[str, float]):
        # Renormalise and absorb rounding residue into the largest component, so
        # every emitted vector sums to exactly 1.0. MOD's derivation assumes
        # Σ w_k = 1, and the centroid (1/3, 1/3, 1/3) does not survive rounding
        # to 6 dp otherwise.
        raw = {b: max(0.0, float(w.get(b, 0.0))) for b in blades}
        total = sum(raw.values())
        if total <= 0.0:
            return
        full = {b: round(raw[b] / total, 6) for b in blades}
        residual = round(1.0 - sum(full.values()), 6)
        if residual != 0.0:
            top = max(full, key=lambda b: full[b])
            full[top] = round(full[top] + residual, 6)

        key = tuple(full[b] for b in blades)
        if key not in seen:
            seen.add(key)
            grid.append(full)

    if spec == "vertices":
        for b in blades:
            add({b: 1.0})

    elif spec.startswith("edge:"):
        _, a, b = spec.split(":")
        for i in range(steps + 1):
            t = i / steps
            add({a: 1.0 - t, b: t})

    elif spec == "edges":
        for a, b in itertools.combinations(blades, 2):
            for i in range(steps + 1):
                t = i / steps
                add({a: 1.0 - t, b: t})
        add({b: 1.0 / len(blades) for b in blades})  # centroid

    elif spec == "simplex":
        # All lattice points (i,j,k)/steps with i+j+k = steps.
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                add({blades[0]: i / steps, blades[1]: j / steps, blades[2]: k / steps})

    else:
        raise ValueError(f"Unknown --grid spec '{spec}'")

    return grid


# ═════════════════════════════════════════════════════════════════════════════
# 2. MOD — memory-shared reimplementation
# ═════════════════════════════════════════════════════════════════════════════

class SharedAdapterMODGenerator:
    """MOD token-level mixing over LoRA adapters on ONE shared base model.

    `benchmarking/strategies/mod.py` takes a list of independent full models,
    which for three Qwen2.5-7B policies is ~45GB and does not fit alongside the
    drafter on a 48GB card. Here the K policies are K named LoRA adapters on a
    single base, switched with `set_adapter` between forward passes, with one KV
    cache per adapter (a cache is only valid for the adapter that built it).

    The mixing arithmetic follows MOD's own released inference code:

        reverse_kld / jsd : Σ_k w_k · log p_k
        forward_kld       : −logsumexp_k( −log p_k + log w_k )

    Only the mixing differs from their implementation; the sampling stack
    (temperature / top-p / top-k) is shared with every other method here so the
    generation configuration is identical across algorithms, as MOD requires.
    """

    def __init__(self, cfg, tokenizer, blade_host: PeftModel, f_divergence: str = "reverse_kld"):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.host = blade_host
        self.f_divergence = f_divergence
        self.device = next(blade_host.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 verbose: bool = False, return_stats: bool = False,
                 blade_coefficients: Optional[Dict[str, float]] = None):
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        coeffs = {k: v for k, v in (blade_coefficients or {}).items() if v > 0.0}
        if not coeffs:
            raise ValueError("MOD requires at least one blade with positive weight")
        names = list(coeffs)
        weights = [coeffs[n] for n in names]
        total = sum(weights)
        weights = [w / total for w in weights]  # MOD assumes Σ w = 1

        t_start = time.perf_counter()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        caches = [None] * len(names)
        cur = [input_ids] * len(names)
        generated: List[int] = []
        forward_passes = 0

        for _ in range(max_tokens):
            logprob_stack = []
            for i, name in enumerate(names):
                self.host.set_adapter(name)
                out = self.host(input_ids=cur[i], past_key_values=caches[i], use_cache=True)
                forward_passes += 1
                caches[i] = out.past_key_values
                logprob_stack.append(F.log_softmax(out.logits[0, -1, :].float(), dim=-1))

            if self.f_divergence in ("reverse_kld", "jsd"):
                mixed = sum(w * lp for w, lp in zip(weights, logprob_stack))
            elif self.f_divergence == "forward_kld":
                stacked = torch.stack([
                    -lp + torch.log(torch.tensor(w, device=lp.device))
                    for w, lp in zip(weights, logprob_stack) if w > 0
                ])
                mixed = -torch.logsumexp(stacked, dim=0)
            else:
                raise ValueError(f"Unsupported f-divergence '{self.f_divergence}'")

            next_id = _sample_from_logits(mixed, self.cfg)
            token_id = int(next_id.item())
            if token_id == self.tokenizer.eos_token_id:
                break
            generated.append(token_id)
            step_in = next_id.view(1, 1)
            cur = [step_in] * len(names)

        elapsed = time.perf_counter() - t_start
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        stats = {
            "strategy": "mod",
            "f_divergence": self.f_divergence,
            "total_tokens": len(generated),
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(len(generated) / max(elapsed, 1e-6), 2),
            "forward_passes": forward_passes,
            "forward_passes_per_token": round(forward_passes / max(len(generated), 1), 3),
        }
        return (text, stats) if return_stats else text


def _sample_from_logits(logits: torch.Tensor, cfg) -> torch.Tensor:
    """Shared sampling stack: temperature → top-p → top-k → multinomial.

    Used by MOD, Rewarded Soups and the base arm so that any frontier difference
    is attributable to the mixing mechanism and not to decoding configuration.
    """
    if cfg.temperature <= 0.0:
        return torch.argmax(logits).view(1)

    logits = logits / cfg.temperature
    probs = F.softmax(logits, dim=-1)

    if cfg.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > cfg.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probs = probs.clone()
        probs[sorted_idx[remove]] = 0.0
        probs = probs / probs.sum()

    if cfg.top_k and cfg.top_k > 0:
        k = min(cfg.top_k, probs.numel())
        topk_p, topk_i = torch.topk(probs, k)
        masked = torch.zeros_like(probs)
        masked.scatter_(-1, topk_i, topk_p)
        probs = masked / masked.sum()

    return torch.multinomial(probs, num_samples=1)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Rewarded Soups — LoRA parameter merging
# ═════════════════════════════════════════════════════════════════════════════

class RewardedSoupsGenerator:
    """MOD's primary baseline: merge the K policies' parameters at weights w.

    Rewarded Soups (Ramé et al., 2023) forms θ = Σ_k w_k θ_k. With LoRA blades
    this is cheap and exact via PEFT's `add_weighted_adapter`, so unlike MOD's
    setting (where merging full 7B checkpoints per w is expensive) this baseline
    is essentially free for us — which is worth stating, because it makes the
    comparison harder for our method, not easier.

    A merged adapter is created per weight vector and deleted afterwards, so
    peak memory stays at one base model plus K+1 adapters.
    """

    MERGED = "_rs_merged"

    def __init__(self, cfg, tokenizer, blade_host: PeftModel):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.host = blade_host
        self.device = next(blade_host.parameters()).device

    def _mount(self, coeffs: Dict[str, float]):
        active = {k: v for k, v in coeffs.items() if v > 0.0}
        if self.MERGED in getattr(self.host, "peft_config", {}):
            self.host.delete_adapter(self.MERGED)

        if len(active) == 1:
            self.host.set_adapter(next(iter(active)))
            return

        names = list(active)
        total = sum(active.values())
        weights = [active[n] / total for n in names]
        try:
            self.host.add_weighted_adapter(
                adapters=names, weights=weights,
                adapter_name=self.MERGED, combination_type="linear",
            )
        except Exception as err:
            # 'linear' needs identical rank across adapters. The honesty blade was
            # trained in a separate run, so ranks can differ; 'cat' concatenates
            # instead and is rank-agnostic.
            logger.warning("Linear LoRA merge failed (%s) — falling back to combination_type='cat'.", err)
            self.host.add_weighted_adapter(
                adapters=names, weights=weights,
                adapter_name=self.MERGED, combination_type="cat",
            )
        self.host.set_adapter(self.MERGED)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 verbose: bool = False, return_stats: bool = False,
                 blade_coefficients: Optional[Dict[str, float]] = None):
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        self._mount(blade_coefficients or {})

        t_start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.host.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=self.cfg.temperature > 0.0,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t_start

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        stats = {
            "strategy": "rs",
            "total_tokens": int(gen_ids.shape[0]),
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(int(gen_ids.shape[0]) / max(elapsed, 1e-6), 2),
            "forward_passes": int(gen_ids.shape[0]),
            "forward_passes_per_token": 1.0,
        }
        return (text, stats) if return_stats else text


class BaseGenerator:
    """Frozen SFT backbone with every adapter disabled — the frontier origin."""

    def __init__(self, cfg, tokenizer, model):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 verbose: bool = False, return_stats: bool = False, **kwargs):
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        t_start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        ctx = self.model.disable_adapter() if isinstance(self.model, PeftModel) else _null_ctx()
        with ctx:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.cfg.temperature > 0.0,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t_start
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        stats = {
            "strategy": "base",
            "total_tokens": int(gen_ids.shape[0]),
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(int(gen_ids.shape[0]) / max(elapsed, 1e-6), 2),
            "forward_passes": int(gen_ids.shape[0]),
            "forward_passes_per_token": 1.0,
        }
        return (text, stats) if return_stats else text


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ═════════════════════════════════════════════════════════════════════════════
# 4. Runner
# ═════════════════════════════════════════════════════════════════════════════

def out_paths(root: str, tribunal_root: str, method: str, cfg_name: str,
              shard: Optional[str]) -> tuple:
    suffix = f"_shard{shard}" if shard is not None else ""
    rich = os.path.join(root, method, f"{cfg_name}{suffix}.json")
    jsonl = os.path.join(tribunal_root, f"{method}__{cfg_name}{suffix}.jsonl")
    return rich, jsonl


def run_config(method: str, generator, records: List[dict], coeffs: Dict[str, float],
               max_tokens: int, verbose: bool) -> dict:
    """Generate one response per prompt under one weight vector."""
    responses, step_stats = [], []
    t0 = time.perf_counter()

    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        output, stats = generator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            verbose=verbose,
            return_stats=True,
            blade_coefficients=coeffs,
        )
        # Swiss-Knife returns prompt+continuation; the token-level arms return
        # only the continuation. Normalise to the continuation in both cases.
        generated = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
        sdict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)

        responses.append({
            "id": rec["id"],
            "axis": rec["axis"],
            "category": rec.get("category"),
            "prompt": prompt,
            "generated": generated,
            "blade_coefficients": coeffs,
            "tokens": sdict.get("total_tokens"),
            "time_s": sdict.get("total_time_s"),
            "tokens_per_second": sdict.get("tokens_per_second"),
        })
        step_stats.append(sdict)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[%s | %s] %d/%d  (%s tok, %s tok/s)",
                    method, _cfg_name(coeffs), i + 1, len(records),
                    sdict.get("total_tokens"), sdict.get("tokens_per_second"))

    elapsed = time.perf_counter() - t0
    total_tokens = sum(r["tokens"] or 0 for r in responses)
    return {
        "method": method,
        "config": _cfg_name(coeffs),
        "blade_coefficients": coeffs,
        "num_prompts": len(records),
        "elapsed_s": round(elapsed, 1),
        "total_tokens": total_tokens,
        "aggregate_tokens_per_second": round(total_tokens / max(elapsed, 1e-6), 2),
        "responses": responses,
        "stats": step_stats,
    }


def save_config(result: dict, rich_path: str, jsonl_path: str, prompt_hash: str, meta: dict):
    os.makedirs(os.path.dirname(rich_path), exist_ok=True)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    payload = dict(result)
    payload["prompt_set_sha256"] = prompt_hash
    payload["run_meta"] = meta
    with open(rich_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Tribunal format: response is read from `response`, id must be the frozen
    # prompt id so every arm joins on it.
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in result["responses"]:
            f.write(json.dumps({
                "id": r["id"],
                "prompt": r["prompt"],
                "response": r["generated"],
            }, ensure_ascii=False) + "\n")

    logger.info("Saved %s → %s  (+ judge input %s)", result["config"], rich_path, jsonl_path)


def merge_shards(root: str, tribunal_root: str, methods: List[str],
                 grid: List[Dict[str, float]], num_shards: int, prompt_hash: str, meta: dict):
    """Concatenate per-shard outputs into one file per (method, config)."""
    for method in methods:
        for coeffs in grid:
            name = _cfg_name(coeffs)
            merged_responses, merged_stats = [], []
            found = 0
            for s in range(num_shards):
                rich, _ = out_paths(root, tribunal_root, method, name, str(s))
                if not os.path.exists(rich):
                    continue
                with open(rich, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("prompt_set_sha256") != prompt_hash:
                    raise RuntimeError(
                        f"{rich} was generated against a different prompt set "
                        f"({d.get('prompt_set_sha256')} != {prompt_hash}). Refusing to merge."
                    )
                merged_responses.extend(d["responses"])
                merged_stats.extend(d.get("stats", []))
                found += 1
            if not merged_responses:
                logger.warning("No shards found for %s / %s", method, name)
                continue

            merged_responses.sort(key=lambda r: r["id"])
            ids = [r["id"] for r in merged_responses]
            if len(set(ids)) != len(ids):
                raise RuntimeError(f"Duplicate prompt ids after merging {method}/{name} — shards overlap.")

            total_tokens = sum(r["tokens"] or 0 for r in merged_responses)
            result = {
                "method": method,
                "config": name,
                "blade_coefficients": coeffs,
                "num_prompts": len(merged_responses),
                "elapsed_s": None,       # wall-clock is not additive across shards
                "total_tokens": total_tokens,
                "aggregate_tokens_per_second": None,
                "responses": merged_responses,
                "stats": merged_stats,
            }
            rich, jsonl = out_paths(root, tribunal_root, method, name, None)
            save_config(result, rich, jsonl, prompt_hash, {**meta, "merged_from_shards": found})


def parse_args():
    p = argparse.ArgumentParser(description="HHH Pareto benchmark (Swiss-Knife vs MOD vs Rewarded Soups)")
    p.add_argument("--methods", nargs="+", default=["swiss"], choices=["swiss", "mod", "rs", "base"])
    p.add_argument("--grid", type=str, default="edges",
                   help="'edges' | 'edge:<a>:<b>' | 'simplex' | 'vertices'")
    p.add_argument("--grid-steps", type=int, default=4,
                   help="Points per edge = steps+1 (default 4 → w ∈ {0,.25,.5,.75,1})")
    p.add_argument("--blades", nargs="+", default=HHH_BLADES)
    p.add_argument("--prompt-file", type=str, default="data/hhh_eval_prompts.jsonl")
    p.add_argument("--output-dir", type=str, default="runs/hhh_pareto")
    p.add_argument("--tribunal-dir", type=str, default="tribunal/inputs/hhh_pareto")

    # generation
    p.add_argument("--max-tokens", type=int, default=256,
                   help="MOD used MAX_LENGTH=200 for its DPO experiments; 256 keeps us close.")
    p.add_argument("--gsi-n", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--elo-rounds", type=int, default=6)
    p.add_argument("--elo-temperature", type=float, default=15.0)
    p.add_argument("--w-tournament", type=float, default=1.0)
    p.add_argument("--w-blade", type=float, default=1.0)
    p.add_argument("--uwo-lambda", type=float, default=0.5,
                   help="LCB penalty. NOTE: do not reuse 0.823 from the Bayesian search — "
                        "that was tuned under the log_ratio_proxy sigma, which is a "
                        "deterministic rescaling of |mu| (see analysis.txt).")
    p.add_argument("--sigma-mode", type=str, default="min_entropy",
                   choices=["none", "min_entropy", "log_ratio_proxy", "mc_dropout"])
    p.add_argument("--probabilistic", action="store_true", default=True)
    p.add_argument("--f-divergence", type=str, default="reverse_kld",
                   choices=["reverse_kld", "jsd", "forward_kld"], help="MOD mixing form")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seed", type=int, default=42)

    # sharding / resume
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--merge-shards", action="store_true",
                   help="Merge existing per-shard files and exit (no GPU needed).")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    records, prompt_hash = load_prompt_set(args.prompt_file)
    grid = build_grid(args.grid, args.grid_steps, args.blades)

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "grid": args.grid,
        "grid_steps": args.grid_steps,
        "blades": args.blades,
        "max_tokens": args.max_tokens,
        "gsi_n": args.gsi_n,
        "beta": args.beta,
        "sigma_mode": args.sigma_mode,
        "probabilistic": args.probabilistic,
        "uwo_lambda": args.uwo_lambda,
        "elo_rounds": args.elo_rounds,
        "elo_temperature": args.elo_temperature,
        "f_divergence": args.f_divergence,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "prompt_file": args.prompt_file,
    }

    print("=" * 78)
    print("  Swiss-Knife — HHH Pareto Benchmark (MOD-comparable)")
    print("=" * 78)
    print(f"  prompts   : {len(records)}  (sha256 {prompt_hash[:16]}…)")
    print(f"  methods   : {args.methods}")
    print(f"  grid      : {args.grid} → {len(grid)} weight vectors")
    for w in grid:
        print(f"      {_cfg_name(w):<28} {w}")
    print(f"  total gens: {len(args.methods) * len(grid) * len(records)}")
    print("=" * 78)

    if args.merge_shards:
        merge_shards(args.output_dir, args.tribunal_dir, args.methods, grid,
                     args.num_shards, prompt_hash, run_meta)
        return

    # ── Prompt sharding ──────────────────────────────────────────────────
    shard_tag = None
    if args.shard_id is not None and args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_id]
        shard_tag = str(args.shard_id)
        logger.info("Shard %d/%d → %d prompts", args.shard_id, args.num_shards, len(records))
        if not records:
            logger.warning("Empty shard, nothing to do.")
            return

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = SwissKnifeConfig(
        max_new_tokens=args.max_tokens,
        gsi_n=args.gsi_n,
        alpha=args.alpha,
        beta=args.beta,
        elo_rounds=args.elo_rounds,
        elo_temperature=args.elo_temperature,
        w_tournament=args.w_tournament,
        w_blade=args.w_blade,
        uwo_lambda=args.uwo_lambda,
        sigma_mode=args.sigma_mode,
        probabilistic=args.probabilistic,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dtype=args.dtype,
        seed=args.seed,
        with_fallback=False,
        generation_mode="gsi_softmax",   # any GSI mode passes config validation
    )

    # ── Models: one base + K named adapters on a single host ─────────────
    logger.info("Loading tokenizer + backbone...")
    tokenizer = load_tokenizer(cfg)
    verifier_model = load_base_model(cfg)

    logger.info("Mounting HHH blades onto one shared host: %s", args.blades)
    blade_host = None
    for b in args.blades:
        # Chaining the returned PeftModel back in as `base_model` takes
        # load_blade_model's `isinstance(base, PeftModel)` branch, which calls
        # load_adapter() and keeps ONE host with K named adapters. Passing the
        # raw base each time (as benchmark_multi_blade.py does) re-wraps an
        # already-injected model.
        blade_host = load_blade_model(cfg, b, base_model=blade_host or verifier_model)
    blades = {
        b: DPOBlade(cfg, verifier_model, blade_host, tokenizer, blade_name=b)
        for b in args.blades
    }

    need_drafter = "swiss" in args.methods
    drafter_model = drafter_tokenizer = None
    if need_drafter:
        logger.info("Loading drafter...")
        drafter_tokenizer = load_drafter_tokenizer(cfg)
        drafter_model = load_drafter_model(cfg)

    generators = {}
    if "swiss" in args.methods:
        generators["swiss"] = EloSwissMultiBladeModeBGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=tokenizer,
            blades=blades,
        )
    if "mod" in args.methods:
        generators["mod"] = SharedAdapterMODGenerator(cfg, tokenizer, blade_host, args.f_divergence)
    if "rs" in args.methods:
        generators["rs"] = RewardedSoupsGenerator(cfg, tokenizer, blade_host)
    if "base" in args.methods:
        generators["base"] = BaseGenerator(cfg, tokenizer, blade_host)

    # ── Sweep ────────────────────────────────────────────────────────────
    for method in args.methods:
        # The base arm ignores w entirely — run it once, not once per grid point.
        method_grid = [{b: 0.0 for b in args.blades}] if method == "base" else grid

        for coeffs in method_grid:
            name = _cfg_name(coeffs)
            rich, jsonl = out_paths(args.output_dir, args.tribunal_dir, method, name, shard_tag)

            if args.skip_existing and os.path.exists(rich):
                logger.info("Skipping %s / %s (exists: %s)", method, name, rich)
                continue

            result = run_config(method, generators[method], records, coeffs,
                                args.max_tokens, args.verbose)
            save_config(result, rich, jsonl, prompt_hash, {**run_meta, "shard": shard_tag})

    print("\n" + "=" * 78)
    print(f"  Done. Rich outputs → {args.output_dir}/<method>/")
    print(f"        Judge inputs → {args.tribunal_dir}/")
    print("  Next:")
    print("    python evaluation/score_hhh_rewards.py --input-dir runs/hhh_pareto")
    print(f"    cd tribunal && python -m tribunal.run_eval --input inputs/hhh_pareto "
          f"--output outputs/hhh_pareto --no-honesty --max-workers 8")
    print("    python evaluation/analyze_hhh_pareto.py")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
