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
GSI_N = 4
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
    mock.parameters = lambda: iter([torch.zeros(1)])
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
        gsi_n=GSI_N,
        alpha=0.5,
        beta=BETA,
        generation_mode="gsi_gumbel",
        gsi_threshold=0.0,
        gsi_tau=1.0,
        max_new_tokens=12,
    )
    tok = _make_mock_tokenizer()
    drafter = _make_mock_model()
    verifier = _make_mock_model()
    blade_m = _make_mock_model()

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
    assert stats.total_steps == 0
    assert stats.acceptance_rate == 0.0

    stats.total_steps = 10
    stats.accepted_steps = 6
    stats.total_tokens = 50
    stats.total_time_s = 2.0
    stats.rejected_steps = 4

    assert abs(stats.acceptance_rate - 0.6) < 1e-6
    assert abs(stats.tokens_per_second - 25.0) < 1e-6
    assert abs(stats.avg_step_tokens - 5.0) < 1e-6

    d = stats.to_dict()
    assert d["strategy"] == "gsi_gumbel"
    assert d["total_steps"] == 10
    assert d["accepted_steps"] == 6
    assert d["acceptance_rate"] == 0.6
    assert d["tokens_per_second"] == 25.0
    print("  ✓ GSIGumbelStats initialized and computed metrics correctly")


def test_gumbel_select():
    """Verify _gumbel_select selects a valid index."""
    gen, _, _, _, _, _ = _make_generator()
    rewards = torch.tensor([1.0, 2.0, 5.0, 0.5])
    idx = gen._gumbel_select(rewards, beta=0.1, tau=1.0)
    assert 0 <= idx < 4
    print("  ✓ _gumbel_select selects a valid index")


def test_sample_reasoning_steps():
    """Verify _sample_reasoning_steps runs correctly with mocks."""
    gen, _, tok, drafter, _, _ = _make_generator()
    # Mock model.generate to return some token sequence
    drafter.generate = MagicMock(return_value=torch.ones((GSI_N, 15), dtype=torch.long) * 5)
    tok.decode = MagicMock(return_value="step 1 text\n\n")
    
    # Custom encode mock that returns a squeezeable tensor
    mock_squeeze_tensor = MagicMock()
    mock_squeeze_tensor.squeeze = MagicMock(return_value=torch.tensor([5, 6, 7]))
    tok.encode = MagicMock(return_value=mock_squeeze_tensor)
    
    prefix_ids = torch.tensor([[1, 2, 3]])
    
    step_ids_list, step_texts = gen._sample_reasoning_steps(
        drafter, tok, prefix_ids, n=GSI_N, device=torch.device("cpu")
    )
    assert len(step_ids_list) == GSI_N
    assert len(step_texts) == GSI_N
    print("  ✓ _sample_reasoning_steps returned steps correctly")


@patch("Model_mechanics.gsi_gumbel.compute_logprob")
def test_full_generate_loop_cheap_accepts(mock_compute_lp):
    """Verify generate() runs successfully with cheap accepts path."""
    mock_compute_lp.return_value = -1.0
    gen, cfg, tok, drafter, verifier, blade_m = _make_generator()
    
    # Mock DPOBlade functions
    mock_blade = MagicMock()
    mock_blade.score_reasoning_steps = lambda ctx, steps: torch.tensor([5.0])
    gen.blade = mock_blade

    # Mock _sample_reasoning_steps
    gen._sample_reasoning_steps = lambda model, tokenizer, prefix_ids, n, device: (
        [torch.tensor([5, 6, 7])] * n,
        ["step text\n\n"] * n
    )

    # Set threshold very low so everything accepts easily
    gen.cfg.gsi_threshold = -100.0

    output, stats = gen.generate("Test prompt.", max_new_tokens=10, return_stats=True)
    assert isinstance(output, str)
    assert isinstance(stats, GSIGumbelStats)
    assert stats.total_steps >= 1
    assert stats.rejected_steps == 0
    print("  ✓ generate() runs successfully with cheap accepts path")


@patch("Model_mechanics.gsi_gumbel.compute_logprob")
def test_full_generate_loop_fallback(mock_compute_lp):
    """Verify generate() triggers fallback path when threshold is unmet."""
    mock_compute_lp.return_value = -1.0
    gen, cfg, tok, drafter, verifier, blade_m = _make_generator()
    
    # Mock DPOBlade functions
    mock_blade = MagicMock()
    mock_blade.score_reasoning_steps = lambda ctx, steps: torch.tensor([-5.0])
    gen.blade = mock_blade

    # Mock _sample_reasoning_steps
    gen._sample_reasoning_steps = lambda model, tokenizer, prefix_ids, n, device: (
        [torch.tensor([5, 6, 7])] * n,
        ["step text\n\n"] * n
    )

    # High threshold to guarantee fallback triggers
    gen.cfg.gsi_threshold = 100.0

    output, stats = gen.generate("Test prompt.", max_new_tokens=10, return_stats=True)
    assert isinstance(output, str)
    assert isinstance(stats, GSIGumbelStats)
    assert stats.total_steps >= 1
    assert stats.rejected_steps > 0
    print("  ✓ generate() triggers fallback path correctly when threshold is unmet")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — GSI Gumbel Step-Level Generator Tests")
    print("=" * 60)
    print()
    test_gsi_gumbel_stats()
    test_gumbel_select()
    test_sample_reasoning_steps()
    test_full_generate_loop_cheap_accepts()
    test_full_generate_loop_fallback()
    print("=" * 60)
    print("  ALL GSI GUMBEL TESTS PASSED ✓")
    print("=" * 60)
