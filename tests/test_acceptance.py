"""
Unit tests for tournament/acceptance.py (Phase 1).

Tests:
  1. Z_local is positive and numerically stable
  2. acceptance_prob is in [0, 1]
  3. High auditor score → high acceptance (relative)
  4. coin_flip returns bool
  5. resample_from_residual returns valid index
  6. Derivation check: draft probability cancels (Z_local is independent of uniform draft)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from Model_mechanics.acceptance import (
    compute_z_local,
    acceptance_prob,
    speculative_coin_flip,
    resample_from_residual,
)


def test_z_local_positive():
    """Z_local must always be positive (guaranteed by exp)."""
    draft_logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.2, -0.1, -0.5, -1.0])
    auditor_scores = torch.tensor([0.5, 0.3, -0.1, 0.2, 0.0, -0.3, 0.1, -0.2])
    z = compute_z_local(draft_logits, auditor_scores, beta=0.1)
    assert z.item() > 0, f"Z_local must be positive, got {z.item()}"
    print(f"  ✓ Z_local = {z.item():.6f}  (positive)")


def test_z_local_numerical_stability():
    """Z_local should be stable even with large or extreme values."""
    # Very negative auditor scores
    draft_logits = torch.zeros(8)
    auditor_scores = torch.tensor([-100.0] * 8)
    z = compute_z_local(draft_logits, auditor_scores, beta=1.0)
    assert torch.isfinite(z), f"Z_local not finite for extreme negative scores: {z}"

    # Very positive auditor scores
    auditor_scores = torch.tensor([100.0] * 8)
    z = compute_z_local(draft_logits, auditor_scores, beta=1.0)
    assert torch.isfinite(z), f"Z_local not finite for extreme positive scores: {z}"
    print("  ✓ Z_local numerically stable under extreme inputs")


def test_acceptance_prob_range():
    """acceptance_prob must always be in [0, 1]."""
    for _ in range(50):
        logits = torch.randn(8)
        scores = torch.randn(8)
        z = compute_z_local(logits, scores, beta=0.1)
        winner_score = scores[scores.argmax()]
        p = acceptance_prob(winner_score, z, beta=0.1)
        assert 0.0 <= p.item() <= 1.0 + 1e-6, f"p_accept out of range: {p.item()}"
    print("  ✓ acceptance_prob always in [0, 1]  (50 random trials)")


def test_higher_score_higher_acceptance():
    """Token with higher auditor score should have higher acceptance probability."""
    draft_logits = torch.zeros(8)  # uniform draft
    low_score  = torch.tensor(-0.5)
    high_score = torch.tensor(+0.5)
    auditor_scores = torch.zeros(8)

    z = compute_z_local(draft_logits, auditor_scores, beta=0.1)
    p_low  = acceptance_prob(low_score,  z, beta=0.1)
    p_high = acceptance_prob(high_score, z, beta=0.1)
    assert p_high.item() >= p_low.item(), (
        f"Higher score should yield higher p_accept: {p_high.item()} vs {p_low.item()}"
    )
    print(f"  ✓ Higher auditor score → higher p_accept ({p_high:.4f} > {p_low:.4f})")


def test_coin_flip_returns_bool():
    """speculative_coin_flip must return a Python bool."""
    p = torch.tensor(0.5)
    result = speculative_coin_flip(p)
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    print(f"  ✓ coin_flip returns bool (sampled: {result})")


def test_coin_flip_always_accept():
    """p_accept = 1.0 → always accept."""
    p = torch.tensor(1.0)
    for _ in range(20):
        assert speculative_coin_flip(p), "Should always accept when p=1.0"
    print("  ✓ coin_flip: p=1.0 always accepts (20 trials)")


def test_coin_flip_never_accept():
    """p_accept = 0.0 → never accept."""
    p = torch.tensor(0.0)
    for _ in range(20):
        assert not speculative_coin_flip(p), "Should never accept when p=0.0"
    print("  ✓ coin_flip: p=0.0 never accepts (20 trials)")


def test_resample_valid_index():
    """resample_from_residual must return a valid top-K index."""
    K = 8
    draft_logits = torch.randn(K)
    auditor_scores = torch.randn(K)
    winner_idx = 0
    fallback = resample_from_residual(draft_logits, auditor_scores, beta=0.1, winner_idx=winner_idx)
    assert 0 <= fallback < K, f"Fallback index {fallback} out of [0, K)"
    print(f"  ✓ resample_from_residual returns valid index: {fallback}")


def test_z_local_draft_cancellation():
    """
    Theoretical check: Z_local is weighted by p_draft, but for a UNIFORM draft
    (all logits equal), Z_local = (1/K) * Σ exp(β * S_k).

    The acceptance probability should then be:
        p_accept(winner) = min(1, exp(β*S_winner) / Σ_k exp(β*S_k) * K)
    which simplifies to: min(1, softmax(β*S)[winner] * K)

    We verify this numerically.
    """
    K = 4
    beta = 0.5
    # Uniform draft logits
    draft_logits = torch.zeros(K)  # all equal → uniform
    auditor_scores = torch.tensor([1.0, 0.5, 0.2, -0.3])

    z = compute_z_local(draft_logits, auditor_scores, beta)

    # Manual computation: Z = (1/K) * Σ exp(β*S) = mean(exp(β*S))
    expected_z = torch.exp(beta * auditor_scores).mean()
    assert abs(z.item() - expected_z.item()) < 1e-5, (
        f"Z_local mismatch: got {z.item():.6f}, expected {expected_z.item():.6f}"
    )
    print(f"  ✓ Z_local derivation check: {z.item():.6f} ≈ {expected_z.item():.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — Acceptance Math Unit Tests")
    print("=" * 60)
    print()

    test_z_local_positive()
    test_z_local_numerical_stability()
    test_acceptance_prob_range()
    test_higher_score_higher_acceptance()
    test_coin_flip_returns_bool()
    test_coin_flip_always_accept()
    test_coin_flip_never_accept()
    test_resample_valid_index()
    test_z_local_draft_cancellation()

    print()
    print("=" * 60)
    print("  ALL ACCEPTANCE TESTS PASSED ✓")
    print("=" * 60)
