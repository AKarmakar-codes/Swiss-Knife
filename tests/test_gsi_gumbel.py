"""
Unit tests for Model_mechanics/gsi_gumbel.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.gsi_gumbel import GSIGumbelStats, GSIGumbelGenerator


VOCAB_SIZE = 1000
PROMPT_LEN = 10
GAMMA = 4
K = 4
BETA = 0.1


def _make_mock_model(vocab_size: int = VOCAB_SIZE):
    """Create a mock model that returns random logits of correct shape."""
    mock = MagicMock()
    def _forward(input_ids, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        out = MagicMock()
        out.logits = torch.randn(B, T, vocab_size)
        return out
    mock.side_effect = _forward
    mock.__call__ = mock
    return mock


def _make_mock_tokenizer(vocab_size: int = VOCAB_SIZE, eos_id: int = 2):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = eos_id
    tok.pad_token_id = 0
    def _encode(text, return_tensors=None, **kwargs):
        ids = torch.randint(3, vocab_size, (1, PROMPT_LEN))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    tok.side_effect = _encode
    tok.__call__ = _encode
    tok.decode = lambda ids, **kw: "mocked output text"
    return tok


def _make_generator():
    cfg = SwissKnifeConfig(
        K=K,
        gamma=GAMMA,
        alpha=0.5,
        beta=BETA,
        generation_mode="gsi_gumbel",
        gsi_threshold=0.0,
        gsi_tau=1.0,
        normalize_scores=True,
        max_new_tokens=12,
    )
    tok = _make_mock_tokenizer()
    drafter = _make_mock_model()
    verifier = _make_mock_model()
    blade_m = _make_mock_model()

    # Make parameters() work on the mock model for device detection
    drafter.parameters = lambda: iter([torch.zeros(1)])
    verifier.parameters = lambda: iter([torch.zeros(1)])

    gen = GSIGumbelGenerator.__new__(GSIGumbelGenerator)
    gen.cfg = cfg
    gen.drafter_tokenizer = tok
    gen.verifier_tokenizer = tok
    gen.drafter_model = drafter
    gen.verifier_model = verifier
    gen.blade_model = blade_m
    gen.drafter_device = torch.device("cpu")
    gen.verifier_device = torch.device("cpu")

    return gen, cfg, tok, drafter, verifier, blade_m


def test_gsi_gumbel_stats():
    """Verify GSIGumbelStats fields and properties."""
    stats = GSIGumbelStats()
    assert stats.total_rounds == 0
    assert stats.acceptance_rate == 0.0

    stats.total_rounds = 10
    stats.full_accept_rounds = 6
    stats.total_tokens_accepted = 35
    stats.total_time_s = 2.0
    stats.fallback_rounds = 3

    assert abs(stats.acceptance_rate - 0.6) < 1e-6
    assert abs(stats.tokens_per_second - 17.5) < 1e-6

    d = stats.to_dict()
    assert d["strategy"] == "gsi_gumbel"
    assert d["total_rounds"] == 10
    assert d["fallback_rounds"] == 3
    assert d["acceptance_rate"] == 0.6
    assert d["tokens_per_second"] == 17.5
    print("  ✓ GSIGumbelStats initialized and computed metrics correctly")


def test_gumbel_topk_verifier_shape_and_range():
    """_gumbel_topk_verifier must select correct winner indexes of shape [gamma]."""
    gen, _, _, _, _, _ = _make_generator()

    target_logprobs = torch.randn(GAMMA, K)
    blade_scores = torch.randn(GAMMA, K)

    winners = gen._gumbel_topk_verifier(
        target_logprobs=target_logprobs,
        blade_scores=blade_scores,
        alpha=0.5,
        tau=1.0
    )

    assert winners.shape == (GAMMA,), f"Expected shape ({GAMMA},), got {winners.shape}"
    for val in winners:
        assert 0 <= val.item() < K, f"Winner index out of range: {val.item()}"
    print("  ✓ _gumbel_topk_verifier output shape and bounds are correct")


def test_gsi_gumbel_generator_draft_propose():
    """_draft_propose must return candidate and logit matrices of shape [gamma, K]."""
    gen, _, _, _, _, _ = _make_generator()
    context_ids = torch.randint(3, VOCAB_SIZE, (1, PROMPT_LEN))

    candidate_matrix, draft_topk_logits = gen._draft_propose(context_ids)

    assert candidate_matrix.shape == (GAMMA, K)
    assert draft_topk_logits.shape == (GAMMA, K)
    print("  ✓ _draft_propose shapes are correct")


def test_drafter_logprob_parallel():
    """_drafter_logprob_parallel must return log-probabilities of shape [gamma, K]."""
    gen, _, _, _, _, _ = _make_generator()
    context_ids = torch.randint(3, VOCAB_SIZE, (1, PROMPT_LEN))
    candidate_matrix = torch.randint(3, VOCAB_SIZE, (GAMMA, K))

    log_probs = gen._drafter_logprob_parallel(context_ids, candidate_matrix)
    assert log_probs.shape == (GAMMA, K)
    print("  ✓ _drafter_logprob_parallel shapes are correct")


def test_full_generate_loop_cheap_accepts():
    """Verify that generate() works with mock models under cheap accept path (all pass threshold)."""
    gen, _, _, _, _, _ = _make_generator()

    # Mock DPOBlade functions
    mock_blade = MagicMock()
    # Ensure they return high reward/prob to pass threshold
    mock_blade.score_parallel = lambda ctx, cand: torch.ones(GAMMA, K) * 5.0
    mock_blade.target_logprob_parallel = lambda ctx, cand, model: torch.ones(GAMMA, K) * -0.1
    gen.blade = mock_blade

    # Mock _drafter_logprob_parallel to return high logprob
    gen._drafter_logprob_parallel = lambda ctx, cand: torch.ones(GAMMA, K) * -0.2

    # Set threshold very low so everything accepts easily
    gen.cfg.gsi_threshold = -100.0

    output, stats = gen.generate("Test prompt.", max_new_tokens=8, return_stats=True)
    assert isinstance(output, str)
    assert isinstance(stats, GSIGumbelStats)
    assert stats.total_rounds >= 1
    assert stats.fallback_rounds == 0
    print("  ✓ generate() runs successfully with cheap accepts path")


def test_full_generate_loop_fallback():
    """Verify that generate() falls back to target model when threshold is high."""
    gen, _, _, _, _, _ = _make_generator()

    mock_blade = MagicMock()
    # Low blade scores to fail threshold check
    mock_blade.score_parallel = lambda ctx, cand: torch.ones(cand.shape[0], cand.shape[1]) * -10.0
    mock_blade.target_logprob_parallel = lambda ctx, cand, model: torch.ones(GAMMA, K) * -5.0
    gen.blade = mock_blade

    gen._drafter_logprob_parallel = lambda ctx, cand: torch.ones(GAMMA, K) * -0.1

    # High threshold to guarantee fallback triggers
    gen.cfg.gsi_threshold = 100.0

    output, stats = gen.generate("Test prompt.", max_new_tokens=8, return_stats=True)
    assert isinstance(output, str)
    assert isinstance(stats, GSIGumbelStats)
    assert stats.total_rounds >= 1
    assert stats.fallback_rounds > 0
    print("  ✓ generate() triggers fallback path correctly when threshold is unmet")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — GSI Gumbel Generator Tests")
    print("=" * 60)
    print()
    test_gsi_gumbel_stats()
    test_gumbel_topk_verifier_shape_and_range()
    test_gsi_gumbel_generator_draft_propose()
    test_drafter_logprob_parallel()
    test_full_generate_loop_cheap_accepts()
    test_full_generate_loop_fallback()
    print("=" * 60)
    print("  ALL GSI GUMBEL TESTS PASSED ✓")
    print("=" * 60)
