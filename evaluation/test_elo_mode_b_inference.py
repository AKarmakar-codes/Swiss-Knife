"""
Swiss-Knife: Terminal Inference & Step Diagnostic Visualizer for Mode B
=======================================================================
This script allows you to run trial prompts using `EloSwissModeBGenerator`
(`Model_mechanics/elo_swiss_mode_b.py`) and visually inspect how the tournament,
DPO Blade rewards, uncertainty estimation, and candidate step selection work
in real-time.

Modes:
  1. Full Inference (Requires GPU):
     Loads the Drafter, Verifier, and DPO Blade adapter, running step-level
     Mode-B decoding on real model tensors.
     Command:
       python evaluation/test_elo_mode_b_inference.py --prompt "Your prompt here"

  2. Dry-Run Simulation (No GPU required, runs on CPU):
     Simulates Drafter candidate step generation, Thurstonian match updates,
     and logit mixing to verify script logic and inspect terminal output formatting.
     Command:
       python evaluation/test_elo_mode_b_inference.py --dry-run
"""

import os
import sys
import time
import json
import random
import argparse
import logging
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModeB_Trial")


def print_banner(prompt: str, cfg_info: Dict[str, Any]):
    print("\n" + "═" * 80)
    print(" 🔪 SWISS-KNIFE GSI MODE-B (ELO TOURNAMENT UNCONDITIONAL ACCEPTANCE) TRIAL")
    print("═" * 80)
    print(f" PROMPT: \"{prompt}\"")
    print("─" * 80)
    print(" CONFIGURATION:")
    for k, v in cfg_info.items():
        print(f"   • {k:<22}: {v}")
    print("═" * 80 + "\n")


def format_step_table(
    step_idx: int,
    prefix_text: str,
    candidates: List[Dict[str, Any]],
    selected_idx: int,
    probabilistic: bool,
    w_t: float,
    w_b: float,
    uwo_lambda: float,
    elo_temp: float,
):
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 📍 DECODING STEP {step_idx:<3}                                                 │")
    print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    print(f" 🔤 Current Prefix : \"{prefix_text[-120:] if len(prefix_text) > 120 else prefix_text}\"")
    print(" ──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────────")
    print("  Cand #       │  Blade μ     │ Uncertaintyσ │  Elo Rating  │ Softmax Prob │ Step Text Snippet")
    print(" ──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────────")

    for i, cand in enumerate(candidates):
        is_champ = (i == selected_idx)
        prefix_str = " 👑 [" + str(i+1) + "]" if is_champ else "    [" + str(i+1) + "]"
        txt = cand["text"].strip().replace("\n", " ")
        if len(txt) > 30:
            txt = txt[:27] + "..."
        
        mu_str = f"{cand['mu']:+.4f}"
        sig_str = f"{cand['sigma']:.4f}"
        elo_str = f"{cand['elo']:.1f}"
        prob_str = f"{cand['prob']*100:5.1f}%"

        champ_marker = "◄── CHAMPION ACCEPTED" if is_champ else ""
        print(f"{prefix_str:<14}│ {mu_str:^12} │ {sig_str:^12} │ {elo_str:^12} │ {prob_str:^12} │ \"{txt}\" {champ_marker}")

    print(" ──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────────")
    champ_text = candidates[selected_idx]["text"].strip()
    print(f" 👉 Selected Step Text: \"{champ_text}\"")
    print(f" 📊 Champion Reward μ = {candidates[selected_idx]['mu']:+.4f} | σ = {candidates[selected_idx]['sigma']:.4f}")


def run_gpu_inference(args):
    import torch
    from Model_mechanics.config import SwissKnifeConfig
    from Model_mechanics.elo_swiss_mode_b import EloSwissModeBGenerator
    from Model_mechanics.models import (
        load_drafter_model,
        load_drafter_tokenizer,
        load_verifier_model,
        load_verifier_tokenizer,
        load_blade_model,
    )

    logger.info("Initializing SwissKnifeConfig for Mode B Inference...")
    cfg = SwissKnifeConfig()
    cfg.gsi_n = args.gsi_n
    cfg.max_new_tokens = args.max_tokens
    cfg.elo_rounds = args.elo_rounds
    cfg.elo_temperature = args.elo_temp
    cfg.w_tournament = args.w_tournament
    cfg.w_blade = args.w_blade
    cfg.uwo_lambda = args.uwo_lambda
    cfg.sigma_mode = args.sigma_mode
    cfg.probabilistic = args.probabilistic
    cfg.use_tilted_elo = args.use_tilted_elo

    cfg_info = {
        "Blade Adapter": args.blade,
        "Pool Size (n)": cfg.gsi_n,
        "Max New Tokens": cfg.max_new_tokens,
        "Elo Rounds": cfg.elo_rounds,
        "Elo Temp (T)": cfg.elo_temperature,
        "Weight Tournament (w_t)": cfg.w_tournament,
        "Weight Blade (w_b)": cfg.w_blade,
        "UWO Lambda (λ)": cfg.uwo_lambda,
        "Sigma Mode": cfg.sigma_mode,
        "Probabilistic (Thurstonian)": cfg.probabilistic,
    }
    print_banner(args.prompt, cfg_info)

    logger.info("Loading Drafter model and tokenizer...")
    drafter_model = load_drafter_model(cfg)
    drafter_tokenizer = load_drafter_tokenizer(cfg)

    logger.info("Loading Verifier model and tokenizer...")
    verifier_model = load_verifier_model(cfg)
    verifier_tokenizer = load_verifier_tokenizer(cfg)

    logger.info("Loading Blade model ('%s')...", args.blade)
    blade_model = load_blade_model(cfg, args.blade)

    logger.info("Instantiating EloSwissModeBGenerator...")
    generator = EloSwissModeBGenerator(
        cfg=cfg,
        drafter_model=drafter_model,
        drafter_tokenizer=drafter_tokenizer,
        verifier_model=verifier_model,
        verifier_tokenizer=verifier_tokenizer,
        blade_model=blade_model,
    )

    print("\n🚀 STARTING GENERATION...")
    t0 = time.time()
    output_text, stats = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        verbose=True,
        return_stats=True,
    )
    t1 = time.time()

    print("\n" + "═" * 80)
    print(" 🏁 FINAL GENERATION COMPLETED")
    print("═" * 80)
    print(f" FULL GENERATED TEXT:\n\n{output_text}\n")
    print("─" * 80)
    print(" STATS SUMMARY:")
    print(f"   • Total Decoding Steps : {stats.total_steps}")
    print(f"   • Total Generated Toks : {stats.total_tokens}")
    print(f"   • Total Candidates     : {stats.total_candidates_scored}")
    print(f"   • Total Time (seconds) : {t1 - t0:.2f} s")
    print(f"   • Generation Speed     : {stats.total_tokens / max(t1 - t0, 1e-4):.2f} tokens/sec")
    print("═" * 80 + "\n")


def run_dry_run_simulation(args):
    """CPU dry-run mock mode to demonstrate how step candidates, Elo ratings,
    Thurstonian probabilistic tournament matches, and champion logits function."""
    import torch
    import torch.nn.functional as F
    from Model_mechanics.elo_system import elo_bracket

    cfg_info = {
        "Mode": "DRY-RUN SIMULATION (CPU Mock)",
        "Blade Adapter": args.blade,
        "Pool Size (n)": args.gsi_n,
        "Max New Tokens": args.max_tokens,
        "Elo Rounds": args.elo_rounds,
        "Elo Temp (T)": args.elo_temp,
        "Weight Tournament (w_t)": args.w_tournament,
        "Weight Blade (w_b)": args.w_blade,
        "UWO Lambda (λ)": args.uwo_lambda,
        "Sigma Mode": args.sigma_mode,
        "Probabilistic (Thurstonian)": args.probabilistic,
    }
    print_banner(args.prompt, cfg_info)

    mock_candidates_pool = [
        ["To answer this clearly,", " First, let us analyze the core premise.", " Simply put,", " Here is a breakdown of the key factors:", " In order to understand this,"],
        [" we can break the concept down into key steps.", " it is essential to look at the primary underlying mechanisms.", " we should consider both direct and indirect effects.", " let's look at the primary evidence.", " we must first define the scope."],
        [" This demonstrates how the process unfolds in practice.", " As a result, the outcome remains consistent and reliable.", " This approach ensures clarity and safety.", " Consequently, the system operates as expected.", " Overall, this provides a complete solution."],
    ]

    prefix_text = args.prompt
    n = args.gsi_n

    print("🚀 RUNNING DRY-RUN DECISION STEP SIMULATION...")
    time.sleep(0.5)

    for step_idx in range(1, len(mock_candidates_pool) + 1):
        step_texts = mock_candidates_pool[step_idx - 1][:n]
        while len(step_texts) < n:
            step_texts.append(f" Candidate step continuation {len(step_texts)+1}.")

        # Generate realistic synthetic rewards and sigmas
        random.seed(42 + step_idx)
        blade_rewards = torch.tensor([random.uniform(-0.4, 0.8) for _ in range(n)], dtype=torch.float)
        sigmas = torch.tensor([random.uniform(0.05, 0.35) for _ in range(n)], dtype=torch.float) if args.sigma_mode != "none" else torch.zeros(n)
        draft_logprobs = torch.tensor([random.uniform(-2.5, -0.5) for _ in range(n)], dtype=torch.float)

        # Compute Elo bracket selection
        selected_idx = elo_bracket(
            target_scores=draft_logprobs,
            blade_scores=blade_rewards,
            alpha=0.0,
            normalize=True,
            temperature=args.elo_temp,
            rounds=args.elo_rounds,
            beta=0.1,
            tilted_rewards=None,
            sigmas=sigmas if args.sigma_mode != "none" else None,
            hard_draw=False,
            w_tournament=args.w_tournament,
            w_blade=args.w_blade,
            uwo_lambda=args.uwo_lambda,
            probabilistic=args.probabilistic,
        )

        # Compute raw logits for visualization table
        # logit_i = w_tournament * (R_i - 1500) / T + w_blade * (mu_i - lambda * sigma_i)
        uwo_scores = blade_rewards - args.uwo_lambda * sigmas
        # Simulated Elo ratings centering at 1500
        rating_deltas = (blade_rewards - blade_rewards.mean()) * 200.0
        mock_ratings = 1500.0 + rating_deltas
        
        logits = args.w_tournament * (mock_ratings - 1500.0) / args.elo_temp + args.w_blade * uwo_scores
        probs = F.softmax(logits, dim=0).tolist()

        candidates_data = []
        for i in range(n):
            candidates_data.append({
                "text": step_texts[i],
                "mu": blade_rewards[i].item(),
                "sigma": sigmas[i].item(),
                "elo": mock_ratings[i].item(),
                "prob": probs[i],
            })

        format_step_table(
            step_idx=step_idx,
            prefix_text=prefix_text,
            candidates=candidates_data,
            selected_idx=selected_idx,
            probabilistic=args.probabilistic,
            w_t=args.w_tournament,
            w_b=args.w_blade,
            uwo_lambda=args.uwo_lambda,
            elo_temp=args.elo_temp,
        )

        prefix_text += step_texts[selected_idx]
        time.sleep(0.3)

    print("\n" + "═" * 80)
    print(" 🏁 DRY-RUN MOCK SIMULATION COMPLETE")
    print("═" * 80)
    print(f" FULL GENERATED MOCK OUTPUT:\n\n{prefix_text}\n")
    print("💡 NOTE: To run actual model inference on GPU, run without the `--dry-run` flag.")
    print("═" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Mode B Elo Tournament Inference Visualizer")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why clear communication is essential in teamwork in three sentences.",
        help="Trial prompt to continue from.",
    )
    parser.add_argument(
        "--blade",
        type=str,
        default="harmlessness",
        choices=["harmlessness", "helpfulness", "truthfulness", "humour"],
        help="Name of the DPO blade adapter to load.",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens to generate.")
    parser.add_argument("--gsi-n", type=int, default=5, help="Number of candidate reasoning steps per step.")
    parser.add_argument("--elo-rounds", type=int, default=6, help="Number of Elo tournament match rounds.")
    parser.add_argument("--elo-temp", type=float, default=28.57587, help="Temperature T for rating softmax.")
    parser.add_argument("--w-tournament", type=float, default=0.74063, help="Weight w_tournament in champion logits.")
    parser.add_argument("--w-blade", type=float, default=2.00907, help="Weight w_blade in champion logits.")
    parser.add_argument("--uwo-lambda", type=float, default=0.82332, help="Uncertainty penalty lambda in mu - lambda*sigma.")
    parser.add_argument(
        "--sigma-mode",
        type=str,
        default="log_ratio_proxy",
        choices=["log_ratio_proxy", "none", "mc_dropout"],
        help="Uncertainty estimation strategy.",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        default=True,
        help="Enable Thurstonian probabilistic match CDF (default: True).",
    )
    parser.add_argument(
        "--use-tilted-elo",
        action="store_true",
        default=False,
        help="Use tilted reward r_tilted = mu + (1/beta)(log pi_V - log pi_D) for matches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run CPU dry-run simulation without loading model weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        run_dry_run_simulation(args)
    else:
        run_gpu_inference(args)


if __name__ == "__main__":
    main()
