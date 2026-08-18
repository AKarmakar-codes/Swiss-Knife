"""
benchmark_hhh_pareto.py — 3D HHH Pareto Benchmark Runner
=========================================================

Runs multi-objective steering on the 3D HHH simplex (Helpfulness, Harmlessness,
Honesty) across all canonical alignment strategies:
  swiss — Swiss-Knife Multi-Blade Mode B (CBN + Thurstonian Elo, ours)
  mod   — Multi-Objective Decoding   (Shi et al., NeurIPS 2024)
  rs    — Rewarded Soups             (Ramé et al., NeurIPS 2023)
  args  — Alignment as Reward-Guided Search (Shi et al., 2023)
  bon   — Best-of-N (compute-matched to Swiss-Knife gsi_n)
  base  — Frozen SFT backbone (frontier origin)

All hyperparameters are read from benchmarking/configs.py.  No tuning is done
here; configs.py is the single source of truth.

Outputs:
  runs/hhh_pareto/<method>/<config>.json           — full responses + stats
  tribunal/inputs/hhh_pareto/<method>__<config>.jsonl — judge-ready JSONL

Usage:
  # Build prompt set first (CPU-only, already done):
  python benchmarking/build_hhh_dataset.py --per-axis 40

  # Single-GPU run:
  python benchmarking/benchmark_hhh_pareto.py --methods swiss mod rs args bon base --grid edges

  # 6-GPU parallel:
  for i in $(seq 0 5); do
    CUDA_VISIBLE_DEVICES=$i python benchmarking/benchmark_hhh_pareto.py \\
      --methods swiss mod rs args bon base \\
      --grid edges --num-shards 6 --shard-id $i \\
      > runs/logs/pareto_shard_$i.log 2>&1 &
  done; wait
  python benchmarking/benchmark_hhh_pareto.py \\
    --methods swiss mod rs args bon base --grid edges --merge-shards
"""

import os
import sys
import json
import time
import logging
import argparse
import itertools
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from peft import PeftModel

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_tokenizer,
    load_base_model,
    load_blade_model,
    load_drafter_model,
    load_drafter_tokenizer,
)
from Model_mechanics.elo_swiss_multi_blade_mode_b import EloSwissMultiBladeModeBGenerator
from benchmarking.strategies.mod import MODGenerator
from benchmarking.strategies.args import ARGSGenerator
from benchmarking.strategies.bon import BestOfNGenerator
from benchmarking.configs import get_all_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("HHHPareto")

HHH_BLADES = ["helpfulness", "honesty", "harmlessness"]


# ── Weight Grid ───────────────────────────────────────────────────────────────

def config_name(w: Dict[str, float]) -> str:
    return "w_" + "_".join(f"{b[:4]}{int(round(w.get(b, 0.0) * 100)):03d}" for b in HHH_BLADES)


def build_grid(spec: str, steps: int) -> List[Dict[str, float]]:
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

    if spec == "vertices":
        for b in HHH_BLADES:
            add({b: 1.0})
    elif spec.startswith("edge:"):
        _, a, b = spec.split(":")
        for i in range(steps + 1):
            t = i / steps
            add({a: 1.0 - t, b: t})
    elif spec == "edges":
        for a, b in itertools.combinations(HHH_BLADES, 2):
            for i in range(steps + 1):
                t = i / steps
                add({a: 1.0 - t, b: t})
        add({b: 1.0 / 3 for b in HHH_BLADES})  # centroid
    elif spec == "simplex":
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                add({HHH_BLADES[0]: i / steps, HHH_BLADES[1]: j / steps, HHH_BLADES[2]: (steps - i - j) / steps})
    else:
        raise ValueError(f"Unknown grid spec '{spec}'")
    return grid


# ── Rewarded Soups generator (needs blade_host, lives here not in strategies/) ─

class RewardedSoupsGenerator:
    """Rewarded Soups (Ramé et al., 2023): LoRA linear weight merge."""

    _MERGED = "_rs_merged"

    def __init__(self, cfg, tokenizer, blade_host: PeftModel):
        self.cfg = cfg
        self.tok = tokenizer
        self.host = blade_host
        self.device = next(blade_host.parameters()).device

    def _mount(self, coeffs: Dict[str, float]):
        active = {k: v for k, v in coeffs.items() if v > 0.0}
        if self._MERGED in getattr(self.host, "peft_config", {}):
            self.host.delete_adapter(self._MERGED)
        if len(active) == 1:
            self.host.set_adapter(next(iter(active)))
            return
        names = list(active)
        total = sum(active.values())
        weights = [active[n] / total for n in names]
        try:
            self.host.add_weighted_adapter(adapters=names, weights=weights,
                                           adapter_name=self._MERGED, combination_type="linear")
        except Exception:
            self.host.add_weighted_adapter(adapters=names, weights=weights,
                                           adapter_name=self._MERGED, combination_type="cat")
        self.host.set_adapter(self._MERGED)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 return_stats: bool = False, blade_coefficients: Optional[Dict] = None, **_):
        self._mount(blade_coefficients or {})
        t0 = time.perf_counter()
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.host.generate(**inp,
                                 max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                                 do_sample=self.cfg.temperature > 0.0,
                                 temperature=self.cfg.temperature,
                                 top_p=self.cfg.top_p,
                                 pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        elapsed = time.perf_counter() - t0
        ids = out[0][inp["input_ids"].shape[1]:]
        text = self.tok.decode(ids, skip_special_tokens=True)
        stats = {"strategy": "rs", "total_tokens": int(ids.shape[0]),
                 "total_time_s": round(elapsed, 3),
                 "tokens_per_second": round(ids.shape[0] / max(elapsed, 1e-6), 2)}
        return (text, stats) if return_stats else text


class BaseGenerator:
    """Frozen SFT backbone — unsteered control."""

    def __init__(self, cfg, tokenizer, model):
        self.cfg = cfg
        self.tok = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 return_stats: bool = False, **_):
        t0 = time.perf_counter()
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        ctx = self.model.disable_adapter() if isinstance(self.model, PeftModel) else _NullCtx()
        with ctx:
            out = self.model.generate(**inp,
                                      max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                                      do_sample=self.cfg.temperature > 0.0,
                                      temperature=self.cfg.temperature,
                                      top_p=self.cfg.top_p,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        elapsed = time.perf_counter() - t0
        ids = out[0][inp["input_ids"].shape[1]:]
        text = self.tok.decode(ids, skip_special_tokens=True)
        stats = {"strategy": "base", "total_tokens": int(ids.shape[0]),
                 "total_time_s": round(elapsed, 3),
                 "tokens_per_second": round(ids.shape[0] / max(elapsed, 1e-6), 2)}
        return (text, stats) if return_stats else text


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *_): return False


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_prompts(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_outputs(responses: List[Dict], method: str, cfg_name: str, out_root: str, write_jsonl: bool):
    method_dir = os.path.join(out_root, method)
    os.makedirs(method_dir, exist_ok=True)
    with open(os.path.join(method_dir, f"{cfg_name}.json"), "w") as f:
        json.dump({"method": method, "config": cfg_name, "responses": responses}, f, indent=2)

    if write_jsonl:
        trib_dir = os.path.join("tribunal", "inputs", "hhh_pareto")
        os.makedirs(trib_dir, exist_ok=True)
        with open(os.path.join(trib_dir, f"{method}__{cfg_name}.jsonl"), "w") as f:
            for r in responses:
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"]}) + "\n")


def merge_shards(methods, grid, num_shards, out_root, prompts_file: str = "data/hhh_eval_prompts.jsonl"):
    total_expected = len(load_prompts(prompts_file)) if os.path.exists(prompts_file) else None

    for method in methods:
        for w in grid:
            name = config_name(w)
            seen_ids = set()
            all_responses = []
            for s in range(num_shards):
                p = os.path.join(out_root, f"{method}_shard{s}", f"{name}.json")
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
                    method, name, len(all_responses), total_expected
                )
            else:
                logger.info("Merged %s/%s — %d responses successfully verified", method, name, len(all_responses))

            write_outputs(all_responses, method, name, out_root, write_jsonl=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts",      default="data/hhh_eval_prompts.jsonl")
    p.add_argument("--methods",      nargs="+", default=["swiss", "mod", "rs", "args", "bon", "base"])
    p.add_argument("--grid",         default="edges", choices=["edges", "simplex", "vertices"])
    p.add_argument("--steps",        type=int, default=4)
    p.add_argument("--max-tokens",   type=int, default=200)
    p.add_argument("--output-root",  default="runs/hhh_pareto")
    p.add_argument("--num-shards",   type=int, default=1)
    p.add_argument("--shard-id",     type=int, default=0)
    p.add_argument("--merge-shards", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    grid = build_grid(args.grid, args.steps)

    if args.merge_shards:
        merge_shards(args.methods, grid, args.num_shards, args.output_root, args.prompts)
        return

    logger.info("Methods: %s | Grid: %s (%d configs) | Shard %d/%d",
                args.methods, args.grid, len(grid), args.shard_id + 1, args.num_shards)

    all_prompts = load_prompts(args.prompts)
    prompts = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    logger.info("Evaluating %d prompts (shard %d/%d)", len(prompts), args.shard_id + 1, args.num_shards)

    # ── Load configs ──────────────────────────────────────────────────────────
    cfgs = get_all_configs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sk_cfg_vals = cfgs["swiss"]

    base_cfg = SwissKnifeConfig(
        max_new_tokens=args.max_tokens,
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

    # ── Load models ───────────────────────────────────────────────────────────
    tokenizer  = load_tokenizer(base_cfg)
    base_model = load_base_model(base_cfg)

    # Shared blade host (for RS, Swiss)
    blade_host = PeftModel.from_pretrained(base_model, base_cfg.helpfulness_adapter_path,
                                           adapter_name="helpfulness")
    blade_host.load_adapter(base_cfg.honesty_adapter_path,     adapter_name="honesty")
    blade_host.load_adapter(base_cfg.harmlessness_adapter_path, adapter_name="harmlessness")

    # ── Build generators ──────────────────────────────────────────────────────
    generators = {}

    if "swiss" in args.methods:
        drafter_tok = load_drafter_tokenizer(base_cfg)
        drafter     = load_drafter_model(base_cfg)
        blade_models_map = {
            "helpfulness": blade_host,
            "honesty": blade_host,
            "harmlessness": blade_host,
        }
        generators["swiss"] = EloSwissMultiBladeModeBGenerator(
            cfg=base_cfg,
            drafter_model=drafter,
            drafter_tokenizer=drafter_tok,
            verifier_model=base_model,
            verifier_tokenizer=tokenizer,
            blade_models=blade_models_map,
        )

    if "mod" in args.methods:
        mod_cfg = SwissKnifeConfig(**{**vars(base_cfg),
                                     "temperature": cfgs["mod"]["temperature"],
                                     "top_p":       cfgs["mod"]["top_p"]})
        blade_models = [load_blade_model(base_cfg, b) for b in HHH_BLADES]
        generators["mod"] = MODGenerator(mod_cfg, tokenizer,
                                         models=blade_models,
                                         weights=[1/3, 1/3, 1/3])

    if "rs" in args.methods:
        rs_cfg = SwissKnifeConfig(**{**vars(base_cfg),
                                    "temperature": cfgs["rs"]["temperature"],
                                    "top_p":       cfgs["rs"]["top_p"]})
        generators["rs"] = RewardedSoupsGenerator(rs_cfg, tokenizer, blade_host)

    if "args" in args.methods:
        args_cfg = SwissKnifeConfig(**{**vars(base_cfg),
                                      "temperature": cfgs["args"]["temperature"],
                                      "top_p":       cfgs["args"]["top_p"],
                                      "alpha":       cfgs["args"]["lambda_reward"],
                                      "beta":        0.1})
        generators["args"] = ARGSGenerator(args_cfg, tokenizer, base_model, blade_host)

    if "bon" in args.methods:
        bon_cfg = SwissKnifeConfig(**{**vars(base_cfg),
                                     "temperature": cfgs["bon"]["temperature"],
                                     "top_p":       cfgs["bon"]["top_p"],
                                     "gsi_n":       cfgs["bon"]["N"]})
        generators["bon"] = BestOfNGenerator(bon_cfg, tokenizer, base_model, blade_host)

    if "base" in args.methods:
        generators["base"] = BaseGenerator(base_cfg, tokenizer, base_model)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    for method in args.methods:
        gen = generators[method]
        is_sharded = args.num_shards > 1
        method_key = f"{method}_shard{args.shard_id}" if is_sharded else method

        for w in grid:
            name = config_name(w)
            logger.info("[%s] config %s ...", method, name)
            responses = []

            for p in prompts:
                text, stats = gen.generate(p["prompt"],
                                           max_new_tokens=args.max_tokens,
                                           return_stats=True,
                                           blade_coefficients=w)
                resp = text[len(p["prompt"]):].strip() if text.startswith(p["prompt"]) else text.strip()
                responses.append({
                    "id":       p["id"],
                    "axis":     p.get("axis", "unknown"),
                    "prompt":   p["prompt"],
                    "response": resp,
                    "stats":    stats if isinstance(stats, dict) else stats.to_dict(),
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            write_outputs(responses, method_key, name, args.output_root,
                          write_jsonl=not is_sharded)
            logger.info("[%s] %s → %d responses written", method, name, len(responses))

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
