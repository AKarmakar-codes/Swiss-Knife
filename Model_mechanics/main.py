"""
Swiss Knife — CLI Entry Point

Usage:
    python -m Model_mechanics.main \\
        --prompt "Explain quantum computing simply." \\
        --blade helpfulness \\
        --alpha 0.5 \\
        --K 8 \\
        --L 5 \\
        --max-tokens 200 \\
        --verbose
"""

import argparse
import logging
import sys
import time

import torch

from .config import SwissKnifeConfig
from .models import load_all
from .generation import SwissKnifeGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="Model_mechanics",
        description="Swiss Knife — Decode-Time Alignment via Tournament Sampling (Option A POC)",
    )
    p.add_argument(
        "--prompt", type=str, required=True,
        help="Input prompt for generation.",
    )
    p.add_argument(
        "--blade", type=str, default="helpfulness",
        choices=["helpfulness", "harmlessness", "truthfulness"],
        help="Active alignment blade (default: helpfulness).",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Draft-vs-blade mixing coefficient α ∈ [0, 1]  (default: 0.5).",
    )
    p.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO implicit reward scaling β  (default: 0.1).",
    )
    p.add_argument(
        "--K", type=int, default=8,
        help="Number of candidate spans per tournament (default: 8).",
    )
    p.add_argument(
        "--L", type=int, default=5,
        help="Span length in tokens (default: 5).",
    )
    p.add_argument(
        "--max-tokens", type=int, default=200,
        help="Maximum new tokens to generate (default: 200).",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype (default: float16).",
    )
    p.add_argument(
        "--blade-bias", type=float, default=0.0,
        help="Additive offset on every blade score before the tournament. "
             "Use to test calibration invariance — the chosen text should "
             "not change as this value varies.",
    )
    p.add_argument(
        "--no-normalize", dest="normalize_scores", action="store_false",
        default=True,
        help="Disable per-round z-score normalisation of the draft and "
             "blade score tensors. By default normalisation is ON; it "
             "fixes the scale mismatch that otherwise drowns out the "
             "blade signal. Use --no-normalize for the pristine kernel-"
             "level calibration-invariance test.",
    )
    p.add_argument(
        "--scores-log", type=str, default="",
        help="Optional JSONL path. When set, each tournament round appends "
             "one line with raw + post-normalisation score vectors and the "
             "winner index. Used for plotting.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-round tournament details.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Logging setup ──────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Build config ───────────────────────────────────────────────────
    cfg = SwissKnifeConfig(
        K=args.K,
        L=args.L,
        alpha=args.alpha,
        beta=args.beta,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        dtype=args.dtype,
        blade_bias=args.blade_bias,
        normalize_scores=args.normalize_scores,
        scores_log=args.scores_log,
    )

    # ── Reproducibility ────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Print banner ───────────────────────────────────────────────────
    print("=" * 72)
    print("  Swiss Knife — Decode-Time Alignment via Tournament Sampling")
    print("  Option A: Non-Speculative Best-of-K Tournament")
    print("=" * 72)
    print(f"  Base model : {cfg.base_model_id}/{cfg.base_model_subfolder}")
    print(f"  Blade      : {args.blade}")
    print(f"  α (mix)    : {cfg.alpha}")
    print(f"  β (DPO)    : {cfg.beta}")
    print(f"  K (cands)  : {cfg.K}")
    print(f"  L (span)   : {cfg.L}")
    print(f"  Max tokens : {cfg.max_new_tokens}")
    print(f"  Dtype      : {cfg.dtype}")
    print(f"  Device     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 72)
    print(f"  Prompt     : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print("=" * 72)
    print()

    # ── Load models ────────────────────────────────────────────────────
    print("⏳ Loading models...")
    t0 = time.time()
    tokenizer, base_model, blade_model = load_all(cfg, args.blade)
    t_load = time.time() - t0
    print(f"✓ Models loaded in {t_load:.1f}s\n")

    # ── Generate ───────────────────────────────────────────────────────
    generator = SwissKnifeGenerator(cfg, tokenizer, base_model, blade_model)

    print("⏳ Generating with tournament sampling...\n")
    t0 = time.time()
    output = generator.generate(
        prompt=args.prompt,
        verbose=args.verbose,
    )
    t_gen = time.time() - t0

    # ── Output ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  GENERATED OUTPUT")
    print("=" * 72)
    print(output)
    print("=" * 72)
    print(f"  Generation time: {t_gen:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
