"""
Unit tests for tournament/swiss_system.py (Phase 2).

Tests:
  1. Condorcet winner always wins (same as knockout guarantee)
  2. Calibration invariance: constant offset to blade scores doesn't change winner
  3. alpha=1.0 → target-only winner
  4. alpha=0.0 → blade-only winner
  5. Rounds = ceil(log2(K)) by default
  6. Handles odd K (bye candidate)
  7. swiss_score_summary returns correct structure
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
from tournament.swiss_system import swiss_system_bracket, swiss_score_summary


def test_condorcet_winner():
    """If candidate 0 dominates all others, it must win the Swiss bracket."""
    K = 8
    target = torch.tensor([10.0, 1.0, 2.0, 0.5, 1.5, 0.3, 0.8, 1.2])
    blade  = torch.tensor([10.0, 1.0, 2.0, 0.5, 1.5, 0.3, 0.8, 1.2])
    alpha = 0.5

    winner = swiss_system_bracket(target, blade, alpha)
    print(f"  [condorcet] Winner: c{winner}")
    assert winner == 0, f"Condorcet winner c0 should win, got c{winner}"
    print("  ✓ PASSED\n")


def test_calibration_invariance():
    """Adding a constant offset to ALL blade scores must not change the winner."""
    target = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 0.5, 1.5, 2.5])
    blade  = torch.tensor([2.0, 1.0, 4.0, 3.0, 0.5, 5.0, 2.5, 1.5])
    alpha = 0.5

    winner_original = swiss_system_bracket(target, blade, alpha)
    winner_shifted  = swiss_system_bracket(target, blade + 1000.0, alpha)

    print(f"  [calibration] Original: c{winner_original} | Shifted: c{winner_shifted}")
    assert winner_original == winner_shifted, (
        f"Winner changed under constant shift: c{winner_original} → c{winner_shifted}"
    )
    print("  ✓ PASSED — Winner stable under +1000 blade offset\n")


def test_alpha_one_target_only():
    """α=1.0 → only target scores matter."""
    target = torch.tensor([1.0, 2.0, 8.0, 3.0, 5.0, 4.0, 6.0, 7.0])
    blade  = torch.tensor([9.0, 9.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0])  # irrelevant

    winner = swiss_system_bracket(target, blade, alpha=1.0)
    print(f"  [alpha=1.0] Winner: c{winner}")
    assert winner == 2, f"Expected c2 (highest target=8.0), got c{winner}"
    print("  ✓ PASSED — α=1.0 selects highest target score\n")


def test_alpha_zero_blade_only():
    """α=0.0 → only blade scores matter."""
    target = torch.tensor([9.0, 9.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0])  # irrelevant
    blade  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    winner = swiss_system_bracket(target, blade, alpha=0.0)
    print(f"  [alpha=0.0] Winner: c{winner}")
    assert winner == 7, f"Expected c7 (highest blade=8.0), got c{winner}"
    print("  ✓ PASSED — α=0.0 selects highest blade score\n")


def test_default_rounds():
    """Default rounds should be ceil(log2(K))."""
    for K in [4, 8, 16]:
        expected_rounds = math.ceil(math.log2(K))
        target = torch.randn(K)
        blade  = torch.randn(K)
        summary = swiss_score_summary(target, blade, alpha=0.5)
        assert summary["rounds"] == expected_rounds, (
            f"K={K}: expected {expected_rounds} rounds, got {summary['rounds']}"
        )
    print("  ✓ PASSED — Default rounds = ceil(log2(K)) for K ∈ {4, 8, 16}\n")


def test_handles_odd_K():
    """Swiss system should handle odd K without error (bye mechanism)."""
    K = 5  # Odd
    target = torch.randn(K)
    blade  = torch.randn(K)
    try:
        winner = swiss_system_bracket(target, blade, alpha=0.5)
        assert 0 <= winner < K, f"Winner {winner} out of range [0, {K})"
        print(f"  [odd K=5] Winner: c{winner}")
        print("  ✓ PASSED — Odd K handled without error\n")
    except Exception as e:
        assert False, f"Swiss system crashed on odd K={K}: {e}"


def test_summary_structure():
    """swiss_score_summary should return dict with correct keys."""
    K = 8
    target = torch.randn(K)
    blade  = torch.randn(K)
    summary = swiss_score_summary(target, blade, alpha=0.5)
    assert "champion" in summary
    assert "cum_wins" in summary
    assert "rounds" in summary
    assert isinstance(summary["cum_wins"], list)
    assert len(summary["cum_wins"]) == K
    assert 0 <= summary["champion"] < K
    print(f"  [summary] champion=c{summary['champion']}  wins={summary['cum_wins']}")
    print("  ✓ PASSED — swiss_score_summary structure correct\n")


def test_consistent_with_knockout_on_clear_winner():
    """On a clear dominant candidate, Swiss and Knockout should agree."""
    from tournament.tournament import knockout_bracket

    target = torch.tensor([10.0, 1.0, 2.0, 0.5, 1.5, 0.3, 0.8, 1.2])
    blade  = torch.tensor([10.0, 1.0, 2.0, 0.5, 1.5, 0.3, 0.8, 1.2])
    alpha = 0.5

    ko_winner    = knockout_bracket(target, blade, alpha)
    swiss_winner = swiss_system_bracket(target, blade, alpha)
    print(f"  Knockout: c{ko_winner} | Swiss: c{swiss_winner}")
    assert ko_winner == swiss_winner == 0, (
        f"Both should pick c0 on clear dominant: knockout=c{ko_winner}, swiss=c{swiss_winner}"
    )
    print("  ✓ PASSED — Swiss and Knockout agree on dominant candidate\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — Swiss-System Tournament Tests")
    print("=" * 60)
    print()

    test_condorcet_winner()
    test_calibration_invariance()
    test_alpha_one_target_only()
    test_alpha_zero_blade_only()
    test_default_rounds()
    test_handles_odd_K()
    test_summary_structure()
    test_consistent_with_knockout_on_clear_winner()

    print("=" * 60)
    print("  ALL SWISS-SYSTEM TESTS PASSED ✓")
    print("=" * 60)
