"""
Verification script for Thurstonian/Uncertainty-Aware GSI Elo Tournament.
Mocks candidate scores and uncertainties to verify:
1. deterministic vs probabilistic win rates (upsets).
2. combined selection logits behavior under uwo_lambda.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from Model_mechanics.elo_system import elo_bracket

def run_verification():
    print("=" * 60)
    print("  GSI Probabilistic Tournament Verification")
    print("=" * 60)

    # 4 candidates:
    # c0: High reward, high uncertainty (potential reward-hacked outlier)
    # c1: Moderate reward, zero uncertainty (safe & consistent choice)
    # c2: Low reward, zero uncertainty
    # c3: Extremely low reward, zero uncertainty
    target_scores = torch.zeros(4, dtype=torch.float32)
    blade_scores  = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    sigmas        = torch.tensor([1.5, 0.0, 0.0, 0.0], dtype=torch.float32)

    # ── Test 1: Weaker Candidate Upset Verification ──
    # Under standard Bradley-Terry / non-probabilistic mode:
    # c1 (2.0) vs c2 (1.0) is entirely deterministic/soft-sorted.
    # Under probabilistic mode, we want to see that c2 can sometimes win matches
    # or even the tournament due to Thurstonian CDF sampling.
    print("\n[TEST 1] Verifying Thurstonian Upsets (probabilistic=True)")
    
    winners_bt = []
    winners_prob = []
    
    # Run tournaments with T=1.0 to sample champions based on final ratings
    for _ in range(200):
        # Bradley-Terry (no sigmas)
        w_bt = elo_bracket(
            target_scores, blade_scores, alpha=0.0, normalize=False,
            temperature=1.0, rounds=6, sigmas=None, hard_draw=True,
            w_tournament=1.0, w_blade=0.0, uwo_lambda=0.0, probabilistic=False
        )
        winners_bt.append(w_bt)

        # Thurstonian CDF (with sigmas & probabilistic=True)
        w_prob = elo_bracket(
            target_scores, blade_scores, alpha=0.0, normalize=False,
            temperature=1.0, rounds=6, sigmas=sigmas, hard_draw=True,
            w_tournament=1.0, w_blade=0.0, uwo_lambda=0.0, probabilistic=True
        )
        winners_prob.append(w_prob)

    bt_counts = {i: winners_bt.count(i) for i in range(4)}
    prob_counts = {i: winners_prob.count(i) for i in range(4)}

    print(f"  Bradley-Terry winner distribution (Deterministic Matchups):")
    for i, count in bt_counts.items():
        print(f"    c{i} (r={blade_scores[i].item():.1f}, σ={sigmas[i].item():.1f}): {count} times")
        
    print(f"  Thurstonian winner distribution (Stochastic Upsets Enabled):")
    for i, count in prob_counts.items():
        print(f"    c{i} (r={blade_scores[i].item():.1f}, σ={sigmas[i].item():.1f}): {count} times")

    # c2 or c3 should occasionally win under Thurstonian CDF
    assert prob_counts[2] > 0 or prob_counts[3] > 0, "No upsets recorded in Thurstonian mode!"
    print("  ✓ Thurstonian upsets verified successfully (weaker candidates occasionally win).")

    # ── Test 2: Uncertainty Gating (UWO-penalised Softmax Selection) ──
    # If w_blade=1.0 and uwo_lambda=2.0, c0 (r=3.0, σ=1.5) should be penalized heavily.
    # logit penalty: 3.0 - 2.0*1.5 = 0.0, making it perform worse than c1 (2.0 - 2.0*0 = 2.0).
    print("\n[TEST 2] Verifying UWO Selection Penalty (uwo_lambda)")
    
    # Run with w_blade=1.0, w_tournament=0.0 (only choose based on UWO score)
    # T=0.1 to make it highly greedy.
    winner_no_penalty = elo_bracket(
        target_scores, blade_scores, alpha=0.0, normalize=False,
        temperature=0.1, rounds=6, sigmas=sigmas, hard_draw=True,
        w_tournament=0.0, w_blade=1.0, uwo_lambda=0.0, probabilistic=True
    )
    
    winner_with_penalty = elo_bracket(
        target_scores, blade_scores, alpha=0.0, normalize=False,
        temperature=0.1, rounds=6, sigmas=sigmas, hard_draw=True,
        w_tournament=0.0, w_blade=1.0, uwo_lambda=2.0, probabilistic=True
    )
    
    print(f"  Winner with λ=0.0 (No Penalty): c{winner_no_penalty} (Expected: c0)")
    print(f"  Winner with λ=2.0 (High Penalty): c{winner_with_penalty} (Expected: c1)")
    
    assert winner_no_penalty == 0, f"Expected c0 to win without penalty, got c{winner_no_penalty}"
    assert winner_with_penalty == 1, f"Expected c1 to win with penalty, got c{winner_with_penalty}"
    print("  ✓ UWO selection penalty verified successfully.")

    print("\n" + "=" * 60)
    print("  ALL MATHEMATICAL SANITY CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
