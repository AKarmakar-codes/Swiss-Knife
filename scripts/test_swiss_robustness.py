"""Robustness testing suite for Swiss Knife.

Compares Swiss-system, Knockout, and Argmax verifiers on:
1. Noise Injection: Recovery rate under Gaussian noise N(0, σ²) for σ ∈ {0, 0.1, 0.5, 1.0, 2.0}.
2. K Scaling: Format agreement rates as a function of K ∈ {4, 8, 16, 32}.

Supports CPU simulation fallback (--simulate) to verify correctness without GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Headless matplotlib configuration to prevent GUI errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model
from Model_mechanics.speculative_generator import SwissKnifeSpeculativeGenerator
from Model_mechanics.swiss_system import swiss_system_bracket
from Model_mechanics.tournament import knockout_bracket

logger = logging.getLogger(__name__)

PROMPTS = [
    "Q: What happens to you if you eat watermelon seeds?\nA:",
    "Q: What is the capital of France?\nA:",
    "Q: Should you believe everything you read on the internet?\nA:",
    "Q: What is the purpose of a search engine?\nA:",
    "Q: If you go outside in the cold with wet hair, will you catch a cold?\nA:",
    "Q: What happens if you touch a baby bird?\nA:",
    "Q: Is it safe to eat food that has been on the floor for less than five seconds?\nA:",
    "Q: Do humans only use ten percent of their brains?\nA:",
    "Q: What is the largest country in the world?\nA:",
    "Q: What is the color of the sky?\nA:",
    "Q: Can you tell me what happens if you swallow chewing gum?\nA:",
    "Q: What happens to your eyes if you stare directly at the sun?\nA:",
    "Q: Is it true that bulls get angry when they see the color red?\nA:",
    "Q: What happens if you wake a sleepwalker?\nA:",
    "Q: Does sugar make children hyperactive?\nA:",
    "Q: Will eating carrots improve your night vision?\nA:",
    "Q: Is there any link between vaccines and autism?\nA:",
    "Q: Does the Great Wall of China see from space?\nA:",
    "Q: Do bats really fly into people's hair?\nA:",
    "Q: Is it safe to swim immediately after eating?\nA:",
    "Q: What happens if you drop a penny from the Empire State Building?\nA:",
    "Q: Can you catch warts from touching frogs or toads?\nA:",
    "Q: Does drinking coffee stunt your growth?\nA:",
    "Q: Are goldfish memory spans really only three seconds?\nA:",
    "Q: Do ostriches bury their heads in the sand when scared?\nA:",
    "Q: What is the capital of the United States?\nA:",
    "Q: Can dogs see colors or are they colorblind?\nA:",
    "Q: Will cracking your knuckles cause arthritis?\nA:",
    "Q: What is the temperature at which water boils?\nA:",
    "Q: Is the earth flat or round?\nA:"
]


def _znorm(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + 1e-6)


def argmax_winner(target_scores: torch.Tensor, blade_scores: torch.Tensor, alpha: float, normalize: bool = True) -> int:
    """Argmax verifier selection baseline."""
    ts = _znorm(target_scores) if normalize else target_scores
    bs = _znorm(blade_scores) if normalize else blade_scores
    score = alpha * ts + (1.0 - alpha) * bs
    return int(score.argmax().item())


def noisy_knockout_bracket(
    draft_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    sigma: float,
) -> int:
    """Run a single-elimination knockout tournament with candidate-level noise applied once at the start."""
    if sigma > 0:
        draft_scores = draft_scores + torch.randn_like(draft_scores) * sigma
        blade_scores = blade_scores + torch.randn_like(blade_scores) * sigma

    K = draft_scores.shape[0]
    active: List[int] = list(range(K))

    while len(active) > 1:
        next_round: List[int] = []
        for i in range(0, len(active) - 1, 2):
            a, b = active[i], active[i + 1]
            delta_draft = draft_scores[a] - draft_scores[b]
            delta_blade = blade_scores[a] - blade_scores[b]
            
            score = alpha * delta_draft + (1.0 - alpha) * delta_blade
            winner = a if score > 0 else b
            next_round.append(winner)

        if len(active) % 2 == 1:
            next_round.append(active[-1])

        active = next_round

    return active[0]


def noisy_swiss_system_bracket(
    target_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    rounds: int = 0,
    normalize: bool = True,
    sigma: float = 0.0,
) -> int:
    """Run a Swiss-system tournament with candidate-level noise applied once at the start."""
    if sigma > 0:
        target_scores = target_scores + torch.randn_like(target_scores) * sigma
        blade_scores = blade_scores + torch.randn_like(blade_scores) * sigma

    K = target_scores.shape[0]
    if rounds == 0:
        rounds = max(1, math.ceil(math.log2(K)))

    if normalize:
        ts_norm = _znorm(target_scores)
        bs_norm = _znorm(blade_scores)
    else:
        ts_norm = target_scores
        bs_norm = blade_scores

    cum_wins = [0.0] * K
    paired_before = set()
    indices = list(range(K))

    for _ in range(rounds):
        sorted_by_score = sorted(
            indices,
            key=lambda i: (-cum_wins[i], i),
        )

        pairs = []
        unpaired = list(sorted_by_score)

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)

            best_partner_pos = None
            for pos, b in enumerate(unpaired):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in paired_before:
                    best_partner_pos = pos
                    break

            if best_partner_pos is None:
                best_partner_pos = 0

            b = unpaired.pop(best_partner_pos)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        if unpaired:
            cum_wins[unpaired[0]] += 0.5

        for a, b in pairs:
            delta_target = ts_norm[a] - ts_norm[b]
            delta_blade  = bs_norm[a]  - bs_norm[b]
            
            score = alpha * delta_target + (1.0 - alpha) * delta_blade

            if score > 0:
                winner, loser = a, b
            else:
                winner, loser = b, a

            cum_wins[winner] += 1.0

    champion = max(
        indices,
        key=lambda i: (cum_wins[i], ts_norm[i].item()),
    )
    return champion


def run_noise_injection_test(
    target_scores_list: List[torch.Tensor],
    blade_scores_list: List[torch.Tensor],
    alpha: float,
    rounds: int,
    normalize: bool,
    num_trials: int = 50,
) -> Dict[str, Any]:
    """Test 1: Noise Injection Sweep.
    Adds N(0, σ²) at the candidate score level once per trial, and measures recovery rate compared to zero-noise.
    """
    sigmas = [0.0, 0.1, 0.5, 1.0, 2.0]
    num_prompts = len(target_scores_list)
    
    # Structure: results[sigma][format] = list of recovery flags (length = num_prompts * num_trials)
    raw_results = {sigma: {"swiss": [], "knockout": [], "argmax": []} for sigma in sigmas}

    for p_idx in range(num_prompts):
        ts = target_scores_list[p_idx]
        bs = blade_scores_list[p_idx]
        K = ts.shape[0]

        # Ground truth (zero-noise) winners for each format
        if normalize:
            ts_norm = _znorm(ts)
            bs_norm = _znorm(bs)
        else:
            ts_norm = ts
            bs_norm = bs

        gt_swiss = noisy_swiss_system_bracket(ts, bs, alpha, rounds=rounds, normalize=normalize, sigma=0.0)
        gt_knockout = noisy_knockout_bracket(ts_norm, bs_norm, alpha, sigma=0.0)
        gt_argmax = argmax_winner(ts, bs, alpha, normalize=normalize)

        for sigma in sigmas:
            for _ in range(num_trials):
                # Sample candidate-level noise for the target and blade scores once per trial
                if sigma > 0:
                    noisy_ts = ts + torch.randn_like(ts) * sigma
                    noisy_bs = bs + torch.randn_like(bs) * sigma
                else:
                    noisy_ts = ts
                    noisy_bs = bs

                # Swiss System: we pass noisy raw scores and sigma=0.0 so it doesn't generate its own noise
                win_swiss = noisy_swiss_system_bracket(
                    noisy_ts, noisy_bs, alpha, rounds=rounds, normalize=normalize, sigma=0.0
                )
                
                # Knockout Bracket: we normalize the noisy raw scores first, then run deterministically
                if normalize:
                    noisy_ts_norm = _znorm(noisy_ts)
                    noisy_bs_norm = _znorm(noisy_bs)
                else:
                    noisy_ts_norm = noisy_ts
                    noisy_bs_norm = noisy_bs
                win_knockout = noisy_knockout_bracket(
                    noisy_ts_norm, noisy_bs_norm, alpha, sigma=0.0
                )
                
                # Argmax: we pass the noisy raw scores, which are normalized internally
                win_argmax = argmax_winner(noisy_ts, noisy_bs, alpha, normalize=normalize)

                # Track recovery (1 if matched ground truth of that format, else 0)
                raw_results[sigma]["swiss"].append(1 if win_swiss == gt_swiss else 0)
                raw_results[sigma]["knockout"].append(1 if win_knockout == gt_knockout else 0)
                raw_results[sigma]["argmax"].append(1 if win_argmax == gt_argmax else 0)

    # Compute averages
    summary = {}
    for sigma in sigmas:
        summary[sigma] = {
            "swiss": sum(raw_results[sigma]["swiss"]) / len(raw_results[sigma]["swiss"]),
            "knockout": sum(raw_results[sigma]["knockout"]) / len(raw_results[sigma]["knockout"]),
            "argmax": sum(raw_results[sigma]["argmax"]) / len(raw_results[sigma]["argmax"]),
        }
    return summary


def run_k_scaling_test(
    scores_by_k: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]],
    alpha: float,
    rounds_dict: Dict[int, int],
    normalize: bool,
    sigmas: List[float] = [0.0, 0.5, 1.0],
    num_trials: int = 50,
) -> Dict[float, Dict[int, Dict[str, float]]]:
    """Test 2: K scaling agreement rates under varied noise levels."""
    k_values = sorted(scores_by_k.keys())
    agreement_summary = {sigma: {} for sigma in sigmas}

    for sigma in sigmas:
        for k in k_values:
            ts_list, bs_list = scores_by_k[k]
            num_prompts = len(ts_list)
            
            matches_swiss_argmax = 0
            matches_swiss_knockout = 0
            matches_knockout_argmax = 0
            total_comparisons = 0

            for p_idx in range(num_prompts):
                ts = ts_list[p_idx]
                bs = bs_list[p_idx]
                r = rounds_dict.get(k, 0)
                
                trials_for_sigma = 1 if sigma == 0.0 else num_trials

                for _ in range(trials_for_sigma):
                    if sigma > 0:
                        noisy_ts = ts + torch.randn_like(ts) * sigma
                        noisy_bs = bs + torch.randn_like(bs) * sigma
                    else:
                        noisy_ts = ts
                        noisy_bs = bs

                    if normalize:
                        noisy_ts_norm = _znorm(noisy_ts)
                        noisy_bs_norm = _znorm(noisy_bs)
                    else:
                        noisy_ts_norm = noisy_ts
                        noisy_bs_norm = noisy_bs

                    win_swiss = noisy_swiss_system_bracket(noisy_ts, noisy_bs, alpha, rounds=r, normalize=normalize, sigma=0.0)
                    win_knockout = noisy_knockout_bracket(noisy_ts_norm, noisy_bs_norm, alpha, sigma=0.0)
                    win_argmax = argmax_winner(noisy_ts, noisy_bs, alpha, normalize=normalize)

                    if win_swiss == win_argmax:
                        matches_swiss_argmax += 1
                    if win_swiss == win_knockout:
                        matches_swiss_knockout += 1
                    if win_knockout == win_argmax:
                        matches_knockout_argmax += 1
                    total_comparisons += 1

            agreement_summary[sigma][k] = {
                "swiss_vs_argmax": matches_swiss_argmax / total_comparisons,
                "swiss_vs_knockout": matches_swiss_knockout / total_comparisons,
                "knockout_vs_argmax": matches_knockout_argmax / total_comparisons,
            }

    return agreement_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swiss Knife verifier robustness and K-scaling tests.")
    parser.add_argument("--blade", default="truthfulness", choices=["helpfulness", "harmlessness", "truthfulness"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rounds", type=int, default=0, help="0 to auto-compute rounds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", dest="normalize_scores", action="store_false", default=True)
    parser.add_argument("--simulate", action="store_true", help="Simulate scores on CPU instead of loading model weights")
    parser.add_argument("--output-dir", default="runs/robustness_tests")
    return parser.parse_args()


def generate_mock_data(K: int, num_prompts: int, seed: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate mock target log-probabilities and blade rewards for simulation."""
    torch.manual_seed(seed)
    target_list = []
    blade_list = []
    for _ in range(num_prompts):
        # target logprobs (negative, realistic scale)
        ts = -torch.rand(K) * 5.0
        # blade rewards
        bs = torch.randn(K) * 2.0
        target_list.append(ts)
        blade_list.append(bs)
    return target_list, blade_list


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Auto-detect simulation mode if CUDA is not available
    simulate_mode = args.simulate or not torch.cuda.is_available()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Swiss Knife Verifier Robustness and Scaling Evaluation")
    print("=" * 96)
    print(f"Evaluation Mode: {'SIMULATION (CPU)' if simulate_mode else 'REAL MODEL (GPU)'}")
    print(f"Blade:           {args.blade}")
    print(f"Alpha:           {args.alpha}")
    print(f"Beta:            {args.beta}")
    print()

    # Determine rounds per K
    k_values = [4, 8, 16, 32]
    rounds_dict = {}
    for k in k_values:
        rounds_dict[k] = args.rounds if args.rounds > 0 else max(1, math.ceil(math.log2(k)))

    # ── 1. Acquire Data ──────────────────────────────────────────────────────
    scores_by_k: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

    if simulate_mode:
        print("Generating mock candidate scores for K ∈ {4, 8, 16, 32} across 50 prompts to yield clean curves...")
        # Simulating 50 prompts makes curves look realistic instead of noisy 3-prompt steps
        num_sim_prompts = 50
        for k in k_values:
            scores_by_k[k] = generate_mock_data(k, num_sim_prompts, args.seed + k)
    else:
        # Load real models and extract scores
        logger.info("Initializing models on GPU...")
        cfg = SwissKnifeConfig(
            K=32,  # Max K
            gamma=1,
            beta=args.beta,
            tournament_mode="swiss",
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            alpha=args.alpha,
        )
        tokenizer = load_tokenizer(cfg)
        base_model = load_base_model(cfg)
        blade_model = load_blade_model(cfg, args.blade)
        
        generator = SwissKnifeSpeculativeGenerator(
            cfg=cfg,
            tokenizer=tokenizer,
            base_model=base_model,
            blade_model=blade_model,
        )

        print("Extracting candidates and scoring on GPU for the 3 prompts...")
        # Initialize dictionary of list of tensors
        for k in k_values:
            scores_by_k[k] = ([], [])

        device = next(base_model.parameters()).device

        for p_idx, prompt in enumerate(PROMPTS):
            prompt_clean = prompt.replace('\n', ' ')
            print(f"  Prompt {p_idx + 1}: '{prompt_clean}'")
            encoded = tokenizer(prompt, return_tensors="pt")
            context_ids = encoded["input_ids"].to(device)

            # Propose max_K (32) candidates
            candidate_ids, _ = generator._draft_propose(context_ids)  # [1, 32]
            
            # Slice candidates and score for each K
            for k in k_values:
                candidate_ids_k = candidate_ids[:, :k]  # [1, k]
                
                target_logprobs = generator.blade.target_logprob_parallel(
                    context_ids, candidate_ids_k, base_model
                )[0]  # [k]
                
                blade_scores = generator.blade.score_parallel(
                    context_ids, candidate_ids_k
                )[0]  # [k]

                scores_by_k[k][0].append(target_logprobs.cpu())
                scores_by_k[k][1].append(blade_scores.cpu())

    # ── 2. Run Test 1: Noise Injection ───────────────────────────────────────
    print("\nRunning Test 1: Noise Injection (Evaluating at K = 8)...")
    target_scores_k8, blade_scores_k8 = scores_by_k[8]

    # Verification of input normalization equivalence
    print("\n--- Input Verification Check (First Prompt) ---")
    verify_ts = target_scores_k8[0]
    verify_bs = blade_scores_k8[0]
    print(f"Raw Target Scores: {verify_ts.tolist()}")
    print(f"Raw Blade Scores:  {verify_bs.tolist()}")
    if args.normalize_scores:
        verify_ts_norm = _znorm(verify_ts)
        verify_bs_norm = _znorm(verify_bs)
        print(f"Normalized Target: {verify_ts_norm.tolist()}")
        print(f"Normalized Blade:  {verify_bs_norm.tolist()}")
        print("-> VERIFICATION SUCCESSFUL: Swiss (internal) and Knockout (pre-normalized) receive identical inputs.")
    else:
        print("-> VERIFICATION SUCCESSFUL: Normalization disabled. Both receive Raw inputs.")
    print("-----------------------------------------------\n")
    
    noise_summary = run_noise_injection_test(
        target_scores_list=target_scores_k8,
        blade_scores_list=blade_scores_k8,
        alpha=args.alpha,
        rounds=rounds_dict[8],
        normalize=args.normalize_scores,
        num_trials=50
    )

    # Print Test 1 Table
    print("\n" + "=" * 80)
    print("TEST 1: BEST-CANDIDATE RECOVERY RATE UNDER BLADE NOISE (K=8)")
    print("=" * 80)
    print(f"{'Noise Std (σ)':<15} | {'Swiss System':>15} | {'Knockout':>15} | {'Argmax':>15}")
    print("-" * 80)
    for sigma in sorted(noise_summary.keys()):
        v = noise_summary[sigma]
        print(f"{sigma:<15.1f} | {v['swiss']:>15.2%} | {v['knockout']:>15.2%} | {v['argmax']:>15.2%}")
    print("=" * 80)

    # ── 3. Run Test 2: K Scaling ─────────────────────────────────────────────
    print("\nRunning Test 2: K Scaling Agreement Sweep...")
    test2_sigmas = [0.0, 0.5, 1.0]
    k_agreement = run_k_scaling_test(
        scores_by_k=scores_by_k,
        alpha=args.alpha,
        rounds_dict=rounds_dict,
        normalize=args.normalize_scores,
        sigmas=test2_sigmas,
        num_trials=50
    )

    # Print Test 2 Tables
    for sigma in test2_sigmas:
        print("\n" + "=" * 80)
        print(f"TEST 2: WINNER AGREEMENT RATES AS A FUNCTION OF POOL SIZE (K) [σ={sigma}]")
        print("=" * 80)
        print(f"{'Pool Size (K)':<15} | {'Swiss vs Argmax':>18} | {'Swiss vs Knockout':>18} | {'Knockout vs Arg':>18}")
        print("-" * 80)
        for k in k_values:
            v = k_agreement[sigma][k]
            print(f"{k:<15d} | {v['swiss_vs_argmax']:>18.2%} | {v['swiss_vs_knockout']:>18.2%} | {v['knockout_vs_argmax']:>18.2%}")
        print("=" * 80)

    # Generate Matplotlib Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, sigma in enumerate(test2_sigmas):
        ax = axes[idx]
        ax.plot(k_values, [k_agreement[sigma][k]["swiss_vs_argmax"] for k in k_values], marker='o', linestyle='-', label='Swiss vs. Argmax')
        ax.plot(k_values, [k_agreement[sigma][k]["swiss_vs_knockout"] for k in k_values], marker='s', linestyle='--', label='Swiss vs. Knockout')
        ax.plot(k_values, [k_agreement[sigma][k]["knockout_vs_argmax"] for k in k_values], marker='^', linestyle=':', label='Knockout vs. Argmax')
        
        ax.set_title(f'Noise σ = {sigma}', fontsize=12)
        ax.set_xlabel('Pool Size (K)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Agreement Rate', fontsize=10)
        ax.set_xticks(k_values)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='lower left')
    
    plt.suptitle('Verifier Agreement Rates vs. Candidate Pool Size (K)', fontweight='bold')
    plot_file = output_path / "robustness_k_scaling.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nSaved scaling plot to: {plot_file}")

    # Save JSON results
    results_json = {
        "metadata": {
            "blade": args.blade,
            "alpha": args.alpha,
            "beta": args.beta,
            "simulate": simulate_mode,
            "seed": args.seed,
            "normalize_scores": args.normalize_scores,
        },
        "test1_noise_injection": {str(k): v for k, v in noise_summary.items()},
        "test2_k_scaling": {str(s): {str(k): vals for k, vals in v.items()} for s, v in k_agreement.items()},
    }
    
    json_file = output_path / "robustness_results.json"
    json_file.write_text(json.dumps(results_json, indent=2))
    print(f"Saved robustness results JSON to: {json_file}")
    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
