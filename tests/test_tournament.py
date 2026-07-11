"""
Synthetic test for the knockout bracket tournament engine.

Verifies:
1. Correct winner selection with known scores
2. Alpha=1.0 → draft-only winner
3. Alpha=0.0 → blade-only winner
4. Invariance to additive calibration shift (core theoretical claim)
5. Anti-symmetry of the match score matrix
"""

import sys
sys.path.insert(0, ".")

import torch
from Model_mechanics.tournament import knockout_bracket, tournament_score_matrix


def test_basic_winner():
    """Candidate 2 has the highest combined score — should win."""
    draft = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 0.5, 1.5, 2.5])
    blade = torch.tensor([2.0, 1.0, 4.0, 3.0, 0.5, 5.0, 2.5, 1.5])
    alpha = 0.5
    winner = knockout_bracket(draft, blade, alpha)
    print(f"[basic_winner] Winner: c{winner}")
    # With alpha=0.5, combined = 0.5*draft + 0.5*blade
    # c0=1.5, c1=2.0, c2=4.5, c3=2.5, c4=2.25, c5=2.75, c6=2.0, c7=2.0
    # Bracket: (c0 vs c1)->c1, (c2 vs c3)->c2, (c4 vs c5)->c5, (c6 vs c7)->c6
    # Semi:    (c1 vs c2)->c2, (c5 vs c6)->c5
    # Final:   (c2 vs c5)->c2
    assert winner == 2, f"Expected c2, got c{winner}"
    print("  ✓ PASSED\n")


def test_alpha_one_draft_only():
    """α=1.0 → only draft scores matter. Highest draft should win."""
    draft = torch.tensor([1.0, 2.0, 8.0, 3.0, 5.0, 4.0, 6.0, 7.0])
    blade = torch.tensor([9.0, 9.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0])  # blade is irrelevant
    winner = knockout_bracket(draft, blade, alpha=1.0)
    print(f"[alpha=1.0] Winner: c{winner}")
    assert winner == 2, f"Expected c2 (highest draft=8.0), got c{winner}"
    print("  ✓ PASSED\n")


def test_alpha_zero_blade_only():
    """α=0.0 → only blade scores matter. Highest blade should win."""
    draft = torch.tensor([9.0, 9.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0])  # draft irrelevant
    blade = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    winner = knockout_bracket(draft, blade, alpha=0.0)
    print(f"[alpha=0.0] Winner: c{winner}")
    assert winner == 7, f"Expected c7 (highest blade=8.0), got c{winner}"
    print("  ✓ PASSED\n")


def test_calibration_invariance():
    """Adding a constant offset to ALL blade scores should not change the winner.

    This is the CORE THEORETICAL CLAIM of tournament sampling:
    pairwise differences Δr are invariant to additive miscalibration.
    """
    draft = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 0.5, 1.5, 2.5])
    blade = torch.tensor([2.0, 1.0, 4.0, 3.0, 0.5, 5.0, 2.5, 1.5])
    alpha = 0.5

    winner_original = knockout_bracket(draft, blade, alpha)

    # Add a large constant offset to ALL blade scores
    blade_shifted = blade + 1000.0
    winner_shifted = knockout_bracket(draft, blade_shifted, alpha)

    print(f"[calibration_invariance] Original winner: c{winner_original}")
    print(f"[calibration_invariance] Shifted winner:  c{winner_shifted}")
    assert winner_original == winner_shifted, \
        f"Winner changed under constant shift: c{winner_original} → c{winner_shifted}"
    print("  ✓ PASSED — Winner unchanged under +1000 blade offset\n")


def test_antisymmetry():
    """Match score matrix should be anti-symmetric: M[i,j] = -M[j,i]."""
    draft = torch.tensor([1.0, 3.0, 5.0, 2.0])
    blade = torch.tensor([4.0, 2.0, 1.0, 3.0])
    alpha = 0.5

    M = tournament_score_matrix(draft, blade, alpha)
    print(f"[antisymmetry] Score matrix:\n{M}")

    # Check M + M^T ≈ 0
    residual = (M + M.T).abs().max().item()
    assert residual < 1e-6, f"Anti-symmetry violated: max|M + M^T| = {residual}"
    print(f"  ✓ PASSED — max|M + M^T| = {residual:.2e}\n")


def test_condorcet_winner():
    """If one candidate beats ALL others pairwise, it must win the bracket."""
    # c0 has highest draft AND blade — Condorcet winner
    draft = torch.tensor([10.0, 3.0, 5.0, 2.0, 4.0, 0.5, 1.5, 2.5])
    blade = torch.tensor([10.0, 1.0, 4.0, 3.0, 0.5, 5.0, 2.5, 1.5])
    alpha = 0.5

    M = tournament_score_matrix(draft, blade, alpha)
    # c0 should have positive score against everyone
    c0_vs_all = M[0, 1:]
    assert (c0_vs_all > 0).all(), "c0 is not Condorcet winner!"

    winner = knockout_bracket(draft, blade, alpha)
    print(f"[condorcet] Winner: c{winner}")
    assert winner == 0, f"Condorcet winner c0 lost! Got c{winner}"
    print("  ✓ PASSED — Condorcet winner wins the bracket\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — Tournament Engine Verification")
    print("=" * 60)
    print()

    test_basic_winner()
    test_alpha_one_draft_only()
    test_alpha_zero_blade_only()
    test_calibration_invariance()
    test_antisymmetry()
    test_condorcet_winner()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
