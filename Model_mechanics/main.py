"""
Swiss Knife — CLI Entry Point

Usage (existing):
    python -m Model_mechanics.main \\
        --prompt "Explain quantum computing simply." \\
        --blade helpfulness --generation-mode option_a

Usage (GSI strategies):
    python -m Model_mechanics.main \\
        --prompt "Solve: What is 2+2?" \\
        --blade helpfulness --generation-mode gsi_softmax \\
        --gsi-n 8 --beta 0.1 --gsi-threshold 0.0

    python -m Model_mechanics.main \\
        --prompt "Solve: What is 2+2?" \\
        --blade helpfulness --generation-mode gsi_pairwise \\
        --gsi-n 8 --alpha 0.5 --gsi-tau 1.0

    python -m Model_mechanics.main \\
        --prompt "Solve: What is 2+2?" \\
        --blade helpfulness --generation-mode gsi_swiss \\
        --gsi-n 8 --alpha 0.5 --swiss-rounds 3
"""

import argparse
import json
import logging
import sys
import time

import torch

from .config import SwissKnifeConfig
from .models import load_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="Model_mechanics",
        description="Swiss Knife — Decode-Time Alignment via Tournament / GSI Sampling",
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
        "--generation-mode", type=str, default="option_a",
        choices=["option_a", "option_b", "gsi_softmax", "gsi_pairwise", "swiss", "elo_swiss", "swiss_mode_b", "elo_swiss_mode_b", "gsi_gumbel", "rrm_pool"],
        help="Generation strategy (default: option_a).",
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
        "--elo-temperature", type=float, default=15.0,
        help="Temperature parameter for Elo relative strength selection (default: 15.0).",
    )
    p.add_argument(
        "--elo-rounds", type=int, default=6,
        help="Number of rounds in the Elo rating system tournament (default: 6).",
    )

    # ── GSI-specific arguments ──────────────────────────────────────────
    gsi = p.add_argument_group("GSI Options (for gsi_softmax, gsi_pairwise, swiss)")
    gsi.add_argument(
        "--gsi-n", type=int, default=8,
        help="Number of candidate reasoning steps per iteration (default: 8).",
    )
    gsi.add_argument(
        "--gsi-threshold", type=float, default=0.0,
        help="Rejection threshold u. Steps with reward < u trigger resampling (default: 0.0).",
    )
    gsi.add_argument(
        "--gsi-max-step-tokens", type=int, default=512,
        help="Max tokens per reasoning step (default: 512).",
    )
    gsi.add_argument(
        "--gsi-tau", type=float, default=1.0,
        help="Temperature τ for Bradley-Terry pairwise selection (default: 1.0).",
    )
    gsi.add_argument(
        "--swiss-rounds", type=int, default=6,
        help="Swiss-system rounds. 0 = auto ceil(log2(n)) (default: 6).",
    )
    gsi.add_argument(
        "--stats-out", type=str, default="",
        help="Optional path to write JSON stats after generation.",
    )

    # ── Phase 1 & 2 Thurstonian / Uncertainty arguments ─────────────────
    unc = p.add_argument_group("Phase 1 & 2 Thurstonian/Uncertainty Options")
    unc.add_argument(
        "--sigma-mode", type=str, default="none",
        choices=["none", "mc_dropout", "log_ratio_proxy"],
        help="Uncertainty mode (default: none)."
    )
    unc.add_argument(
        "--sigma-mc-samples", type=int, default=5,
        help="Number of MC dropout samples (default: 5)."
    )
    unc.add_argument(
        "--sigma-dropout-p", type=float, default=None,
        help="Test-time dropout override probability (default: None)."
    )
    unc.add_argument(
        "--w-tournament", type=float, default=1.0,
        help="Weight of tournament score in final choice logits (default: 1.0)."
    )
    unc.add_argument(
        "--w-blade", type=float, default=1.0,
        help="Weight of blade UWO score in final choice logits (default: 1.0)."
    )
    unc.add_argument(
        "--uwo-lambda", type=float, default=0.5,
        help="Uncertainty penalty lambda (default: 0.5)."
    )
    unc.add_argument(
        "--adaptive-threshold", action="store_true",
        help="Enable running percentile threshold calibration."
    )
    unc.add_argument(
        "--threshold-percentile", type=float, default=10.0,
        help="Percentile for threshold calibration (default: 10.0)."
    )
    unc.add_argument(
        "--threshold-buffer-size", type=int, default=20,
        help="Rolling buffer size for threshold calibration (default: 20)."
    )
    unc.add_argument(
        "--no-fallback", dest="with_fallback", action="store_false",
        default=True,
        help="Disable verifier fallback, running in Mode B (accepting unconditionally)."
    )
    unc.add_argument(
        "--hard-draw", action="store_true",
        help="Use hard Bernoulli draws for matchups instead of soft probabilities."
    )
    unc.add_argument(
        "--probabilistic", action="store_true",
        help="Force Thurstonian CDF P(A beats B)=Φ(z) for every Elo match so the "
             "weaker-scored candidate can still win.  Without this flag the default "
             "Bradley-Terry sigmoid is used, which behaves like a soft sort."
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
        generation_mode=args.generation_mode,
        gsi_n=args.gsi_n,
        gsi_threshold=args.gsi_threshold,
        gsi_max_step_tokens=args.gsi_max_step_tokens,
        gsi_tau=args.gsi_tau,
        swiss_rounds=args.swiss_rounds,
        elo_temperature=args.elo_temperature,
        elo_rounds=args.elo_rounds,
        sigma_mode=args.sigma_mode,
        sigma_mc_samples=args.sigma_mc_samples,
        sigma_dropout_p=args.sigma_dropout_p,
        w_tournament=args.w_tournament,
        w_blade=args.w_blade,
        uwo_lambda=args.uwo_lambda,
        adaptive_threshold=args.adaptive_threshold,
        threshold_percentile=args.threshold_percentile,
        threshold_buffer_size=args.threshold_buffer_size,
        with_fallback=args.with_fallback,
        hard_draw=args.hard_draw,
        probabilistic=args.probabilistic,
    )

    # ── Reproducibility ────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Print banner ───────────────────────────────────────────────────
    mode_labels = {
        "option_a": "Option A: Non-Speculative Best-of-K Tournament",
        "option_b": "Option B: Speculative-Decoding-Integrated Tournament",
        "gsi_softmax": "GSI Strategy 1: Softmax Selection",
        "gsi_pairwise": "GSI Strategy 2: Pairwise Bradley-Terry",
        "swiss": "GSI Strategy 3: Swiss-System → Points → Softmax",
        "elo_swiss": "GSI Strategy 4: Elo Tournament Selection",
        "gsi_gumbel": "GSI Strategy 5: Speculative Gumbel-Top-k with Fallback",
    }
    print("=" * 72)
    print("  Swiss Knife — Decode-Time Alignment")
    print(f"  {mode_labels.get(cfg.generation_mode, cfg.generation_mode)}")
    print("=" * 72)
    print(f"  Base model : {cfg.base_model_id}/{cfg.base_model_subfolder}")
    print(f"  Blade      : {args.blade}")
    print(f"  α (mix)    : {cfg.alpha}")
    print(f"  β (DPO)    : {cfg.beta}")
    if cfg.generation_mode.startswith("gsi_") or cfg.generation_mode in ("swiss", "elo_swiss"):
        print(f"  n (cands)  : {cfg.gsi_n}")
        print(f"  threshold  : {cfg.gsi_threshold}")
        print(f"  max step   : {cfg.gsi_max_step_tokens} tokens")
        if cfg.generation_mode == "gsi_pairwise":
            print(f"  τ (B-T)    : {cfg.gsi_tau}")
        if cfg.generation_mode == "swiss":
            print(f"  swiss rnds : {cfg.swiss_rounds or 'auto'}")
        if cfg.generation_mode == "elo_swiss" or cfg.generation_mode == "elo_swiss_mode_b":
            print(f"  elo temp   : {cfg.elo_temperature}")
            print(f"  elo rounds : {cfg.elo_rounds}")
            print(f"  probabilistic: {getattr(cfg, 'probabilistic', False)}")
            print(f"  sigma mode : {cfg.sigma_mode}")
            print(f"  uwo λ      : {cfg.uwo_lambda} (selection penalty, not rejection gate)")
    else:
        print(f"  K (cands)  : {cfg.K}")
        print(f"  L (span)   : {cfg.L}")
        print(f"  Tourney    : {cfg.tournament_mode}")
        if cfg.tournament_mode == "elo":
            print(f"  Elo temp   : {cfg.elo_temperature}")
            print(f"  Elo rounds : {cfg.elo_rounds}")
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
    if cfg.generation_mode in ["gsi_softmax", "gsi_pairwise", "swiss", "elo_swiss", "swiss_mode_b", "elo_swiss_mode_b", "gsi_gumbel", "rrm_pool"]:
        from .models import (
            load_drafter_model,
            load_drafter_tokenizer,
            load_verifier_model,
            load_verifier_tokenizer,
            load_blade_model,
            load_rrm_model,
            load_rrm_tokenizer,
        )
        drafter_model = load_drafter_model(cfg)
        drafter_tokenizer = load_drafter_tokenizer(cfg)
        verifier_model = load_verifier_model(cfg)
        verifier_tokenizer = load_verifier_tokenizer(cfg)
        blade_model = load_blade_model(cfg, args.blade)
        if cfg.generation_mode == "rrm_pool":
            rrm_model = load_rrm_model(cfg)
            rrm_tokenizer = load_rrm_tokenizer(cfg)
        else:
            rrm_model = None
            rrm_tokenizer = None
        tokenizer = verifier_tokenizer
        base_model = verifier_model
    else:
        tokenizer, base_model, blade_model = load_all(cfg, args.blade)
        drafter_model = None
        drafter_tokenizer = None
        rrm_model = None
        rrm_tokenizer = None
    t_load = time.time() - t0
    print(f"✓ Models loaded in {t_load:.1f}s\n")

    # ── Instantiate the right generator ────────────────────────────────
    if cfg.generation_mode == "option_a":
        from .generation import SwissKnifeGenerator
        generator = SwissKnifeGenerator(cfg, tokenizer, base_model, blade_model)
        print("⏳ Generating with Option A tournament sampling...\n")
        t0 = time.time()
        output = generator.generate(prompt=args.prompt, verbose=args.verbose)
        t_gen = time.time() - t0
        stats_dict = None

    elif cfg.generation_mode == "option_b":
        from .speculative_generator import SwissKnifeSpeculativeGenerator
        generator = SwissKnifeSpeculativeGenerator(cfg, tokenizer, base_model, blade_model)
        print("⏳ Generating with Option B speculative tournament...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "gsi_softmax":
        from .gsi_softmax import GSISoftmaxGenerator
        generator = GSISoftmaxGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 1 (Softmax)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "gsi_pairwise":
        from .gsi_pairwise import GSIPairwiseGenerator
        generator = GSIPairwiseGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 2 (Pairwise Bradley-Terry)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "swiss":
        from .swiss import SwissGenerator
        generator = SwissGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 3 (Swiss → Points → Softmax)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "swiss_mode_b":
        from .swiss_mode_b import SwissModeBGenerator
        generator = SwissModeBGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 3 Mode B (Swiss no fallback)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "elo_swiss":
        from .elo_swiss import EloSwissGenerator
        generator = EloSwissGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 4 (Elo Tournament Selection)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "elo_swiss_mode_b":
        from .elo_swiss_mode_b import EloSwissModeBGenerator
        generator = EloSwissModeBGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 4 Mode B (Elo no fallback)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "gsi_gumbel":
        from .gsi_gumbel import GSIGumbelGenerator
        generator = GSIGumbelGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        print("⏳ Generating with GSI Strategy 5 (Gumbel Speculative)...\n")
        t0 = time.time()
        output, stats = generator.generate(
            prompt=args.prompt, verbose=args.verbose, return_stats=True,
        )
        t_gen = time.time() - t0
        stats_dict = stats.to_dict()

    elif cfg.generation_mode == "rrm_pool":
        from .rrm_candidate_pool import RRMCandidatePoolGenerator
        generator = RRMCandidatePoolGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            rrm_model=rrm_model,
            rrm_tokenizer=rrm_tokenizer,
            blade_model=blade_model,
        )
        print(f"⏳ Generating {cfg.rrm_n_candidates} candidates and ranking via RRM tournament...\n")
        t0 = time.time()
        output, candidates = generator.generate(
            prompt=args.prompt, verbose=args.verbose, blade=args.blade
        )
        t_gen = time.time() - t0
        stats_dict = {
            "strategy": "rrm_pool",
            "num_candidates": len(candidates),
            "generation_time_s": round(t_gen, 3),
        }

    else:
        print(f"ERROR: Unknown generation_mode '{cfg.generation_mode}'")
        sys.exit(1)

    # ── Output ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  GENERATED OUTPUT")
    print("=" * 72)
    print(output)
    print("=" * 72)
    print(f"  Generation time: {t_gen:.1f}s")
    if stats_dict:
        print(f"  Stats: {json.dumps(stats_dict, indent=2)}")
    print("=" * 72)

    # ── Write stats if requested ────────────────────────────────────────
    if args.stats_out and stats_dict:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\n📊 Stats written to {args.stats_out}")


if __name__ == "__main__":
    main()
