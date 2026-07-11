"""
Unit tests for the Elo rating system tournament paradigm with decaying K-factors.
Checks:
1. Tournament winner index is valid.
2. Calibration invariance: adding a constant bias to blade scores does not change the greedy winner.
3. Temperature effects: T=0 is deterministic and selects the highest rating, T>0 samples probabilistically.
4. Stochastic Elo bracket execution.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from unittest.mock import MagicMock
from Model_mechanics.elo_system import elo_bracket, stochastic_elo_bracket, elo_score_summary


def test_elo_bracket_greedy():
    print("Testing elo_bracket in greedy mode (temperature = 0)...")
    # 8 candidates
    torch.manual_seed(42)
    target_scores = torch.randn(8, dtype=torch.float32)
    blade_scores  = torch.randn(8, dtype=torch.float32)

    # Run with alpha = 0.5
    winner = elo_bracket(
        target_scores,
        blade_scores,
        alpha=0.5,
        normalize=True,
        temperature=0.0,
    )
    assert 0 <= winner < 8, f"Winner index {winner} is invalid"
    print(f"  ✓ Winner selected: c{winner}")

    # Test shift invariance
    winner_shifted = elo_bracket(
        target_scores,
        blade_scores + 1000.0,  # Offset all blade scores by +1000.0
        alpha=0.5,
        normalize=True,
        temperature=0.0,
    )
    assert winner == winner_shifted, f"Shift invariance broken: {winner} vs {winner_shifted}"
    print("  ✓ Calibration shift-invariance verified (greedy).")


def test_elo_bracket_temperature():
    print("Testing elo_bracket temperature behavior...")
    torch.manual_seed(42)
    target_scores = torch.randn(8, dtype=torch.float32)
    blade_scores  = torch.randn(8, dtype=torch.float32)

    # With T=0 (greedy), winner must be deterministic
    winners_t0 = []
    for _ in range(50):
        w = elo_bracket(
            target_scores,
            blade_scores,
            alpha=0.5,
            normalize=True,
            temperature=0.0,
        )
        winners_t0.append(w)
    assert len(set(winners_t0)) == 1, f"T=0 is not deterministic: {set(winners_t0)}"
    greedy_winner = winners_t0[0]
    print(f"  ✓ T=0 is deterministic: winner is always c{greedy_winner}")

    # With high temperature (T=10.0), selection should cover other candidates
    winners_t10 = []
    for _ in range(100):
        w = elo_bracket(
            target_scores,
            blade_scores,
            alpha=0.5,
            normalize=True,
            temperature=10.0,
        )
        winners_t10.append(w)
    unique_winners = set(winners_t10)
    assert len(unique_winners) > 1, f"T=10.0 selection did not sample diverse winners: {unique_winners}"
    print(f"  ✓ T=10.0 samples diverse winners: {unique_winners}")


def test_stochastic_elo_bracket():
    print("Testing stochastic_elo_bracket...")
    torch.manual_seed(42)
    target_scores = torch.randn(8, dtype=torch.float32)
    
    # Mock a stochastic auditor
    mock_auditor = MagicMock()
    # Mock score_candidates_for_match to return a tensor of shape [gamma, K] -> e.g. [1, 8]
    mock_auditor.score_candidates_for_match.return_value = torch.randn(1, 8, dtype=torch.float32)

    context_ids = torch.zeros((1, 5), dtype=torch.long)
    candidate_matrix = torch.zeros((1, 8, 3), dtype=torch.long)
    ref_logprobs = torch.zeros((1, 8), dtype=torch.float32)

    winner = stochastic_elo_bracket(
        target_scores=target_scores,
        auditor=mock_auditor,
        context_ids=context_ids,
        candidate_matrix=candidate_matrix,
        ref_logprobs=ref_logprobs,
        position_idx=0,
        alpha=0.5,
        normalize=True,
        temperature=1.0,
    )
    assert 0 <= winner < 8, f"Stochastic Elo winner {winner} is invalid"
    assert mock_auditor.draw_fresh_functional.call_count > 0, "draw_fresh_functional was not called"
    print(f"  ✓ Stochastic Elo winner: c{winner}")


def test_elo_bracket_6_rounds():
    print("Testing elo_bracket with 6 rounds...")
    torch.manual_seed(42)
    target_scores = torch.randn(8, dtype=torch.float32)
    blade_scores  = torch.randn(8, dtype=torch.float32)

    winner = elo_bracket(
        target_scores,
        blade_scores,
        alpha=0.5,
        normalize=True,
        temperature=0.0,
        rounds=6,
    )
    assert 0 <= winner < 8, f"Winner index {winner} is invalid for 6 rounds"
    print(f"  ✓ 6-round Winner selected: c{winner}")


if __name__ == "__main__":
    test_elo_bracket_greedy()
    print("-" * 40)
    test_elo_bracket_temperature()
    print("-" * 40)
    test_stochastic_elo_bracket()
    print("-" * 40)
    test_elo_bracket_6_rounds()
    print("\nALL DECAYING ELO TOURNAMENT TESTS PASSED ✓")
