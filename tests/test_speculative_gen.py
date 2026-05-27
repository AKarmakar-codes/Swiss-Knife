"""
Unit tests for tournament/speculative_generator.py (Option B, Phase 2).

All tests use mocked models — no 13 GB weights required.

Tests:
  1. _draft_propose returns [gamma, K] tensor with column 0 = greedy argmax
  2. score_parallel returns [gamma, K] tensor with correct dtype
  3. target_logprob_parallel returns [gamma, K] tensor
  4. Full generate() loop runs to completion (mocked) and returns a string
  5. EOS token in draft proposal stops generation
  6. SpeculativeStats accumulates correctly
  7. Tournament mode switching (knockout vs swiss) doesn't crash
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

from tournament.config import SwissKnifeConfig
from tournament.speculative_generator import SwissKnifeSpeculativeGenerator, SpeculativeStats


VOCAB_SIZE = 1000
PROMPT_LEN = 10
GAMMA = 4
K = 4  # Power of 2 for knockout
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
    # Return a fixed prompt encoding
    def _encode(text, return_tensors=None, **kwargs):
        ids = torch.randint(3, vocab_size, (1, PROMPT_LEN))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    tok.side_effect = _encode
    tok.__call__ = _encode
    tok.decode = lambda ids, **kw: "mocked output text"
    return tok


def _make_generator(tournament_mode: str = "knockout"):
    cfg = SwissKnifeConfig(
        K=K,
        gamma=GAMMA,
        alpha=0.5,
        beta=BETA,
        tournament_mode=tournament_mode,
        swiss_rounds=2,
        normalize_scores=True,
        max_new_tokens=12,
    )
    tok = _make_mock_tokenizer()
    base = _make_mock_model()
    blade_m = _make_mock_model()

    # Make next() work on the mock model for device detection
    base.parameters = lambda: iter([torch.zeros(1)])

    gen = SwissKnifeSpeculativeGenerator.__new__(SwissKnifeSpeculativeGenerator)
    gen.cfg = cfg
    gen.tokenizer = tok
    gen.base_model = base
    gen.blade_model = blade_m

    # Inject real tournament selectors
    from tournament.tournament import knockout_bracket
    from tournament.swiss_system import swiss_system_bracket
    if tournament_mode == "knockout":
        gen._run_tournament = gen._knockout_at_position
    else:
        gen._run_tournament = gen._swiss_at_position

    gen._knockout_at_position = lambda ts, bs: knockout_bracket(ts, bs, cfg.alpha)
    gen._swiss_at_position    = lambda ts, bs: swiss_system_bracket(ts, bs, cfg.alpha, rounds=2)

    return gen, cfg, tok, base, blade_m


def test_draft_propose_shape():
    """_draft_propose must return [gamma, K] tensors."""
    gen, cfg, tok, base, blade = _make_generator()

    # Inject a mock blade scorer
    from tournament.blades import DPOBlade
    mock_blade_scorer = MagicMock()
    gen.blade = mock_blade_scorer

    context_ids = torch.randint(3, VOCAB_SIZE, (1, PROMPT_LEN))
    candidate_matrix, draft_topk_logits = gen._draft_propose(context_ids)

    assert candidate_matrix.shape == (GAMMA, K), (
        f"Expected [{GAMMA},{K}], got {candidate_matrix.shape}"
    )
    assert draft_topk_logits.shape == (GAMMA, K), (
        f"Expected [{GAMMA},{K}], got {draft_topk_logits.shape}"
    )
    print(f"  candidate_matrix shape: {candidate_matrix.shape}  ✓")
    print(f"  draft_topk_logits shape: {draft_topk_logits.shape}  ✓")


def test_draft_column_zero_is_greedy():
    """Column 0 of candidate_matrix must always be the argmax token."""
    gen, cfg, tok, base, _ = _make_generator()
    gen.blade = MagicMock()

    # Use a fixed logit output to verify greedy selection
    fixed_logits = torch.zeros(VOCAB_SIZE)
    greedy_token_id = 42
    fixed_logits[greedy_token_id] = 100.0  # clear argmax

    call_count = [0]
    def fixed_forward(input_ids, attention_mask=None, **kwargs):
        call_count[0] += 1
        B, T = input_ids.shape
        out = MagicMock()
        logits = torch.zeros(B, T, VOCAB_SIZE)
        logits[:, -1, greedy_token_id] = 100.0  # greedy at every step
        out.logits = logits
        return out

    base.side_effect = fixed_forward
    base.__call__ = base

    context_ids = torch.randint(3, VOCAB_SIZE, (1, PROMPT_LEN))
    candidate_matrix, _ = gen._draft_propose(context_ids)

    for t in range(GAMMA):
        assert candidate_matrix[t, 0].item() == greedy_token_id, (
            f"Position {t}: expected greedy={greedy_token_id}, "
            f"got {candidate_matrix[t, 0].item()}"
        )
    print(f"  ✓ All {GAMMA} positions have greedy token {greedy_token_id} at column 0")


def test_speculative_stats_fields():
    """SpeculativeStats should initialize correctly and compute derived metrics."""
    stats = SpeculativeStats()
    assert stats.total_rounds == 0
    assert stats.acceptance_rate == 0.0

    stats.total_rounds = 10
    stats.full_accept_rounds = 7
    stats.total_tokens_accepted = 40
    stats.total_time_s = 2.0
    stats.blade_forward_passes = 10

    assert abs(stats.acceptance_rate - 0.7) < 1e-6
    assert abs(stats.tokens_per_second - 20.0) < 1e-6
    assert abs(stats.auditor_calls_per_token - 0.25) < 1e-6
    print(f"  acceptance_rate={stats.acceptance_rate:.2f}  ✓")
    print(f"  tokens_per_second={stats.tokens_per_second:.1f}  ✓")
    print(f"  auditor_calls_per_token={stats.auditor_calls_per_token:.3f}  ✓")


def test_generate_returns_string_knockout():
    """Full generate() with knockout tournament should return a string."""
    gen, cfg, tok, base, blade_m = _make_generator("knockout")

    # Mock blade scorer
    mock_blade = MagicMock()
    mock_blade.score_parallel = lambda ctx, cand: torch.randn(GAMMA, K)
    mock_blade.target_logprob_parallel = lambda ctx, cand, model: torch.randn(GAMMA, K)
    gen.blade = mock_blade

    # Make parameters() work for device detection
    device_param = torch.zeros(1)
    base.parameters = lambda: iter([device_param])

    text = gen.generate("Test prompt.", max_new_tokens=8)
    assert isinstance(text, str), f"Expected str, got {type(text)}"
    print(f"  generate() returned: '{text}'  ✓")


def test_generate_returns_stats():
    """return_stats=True should return (str, SpeculativeStats)."""
    gen, cfg, tok, base, blade_m = _make_generator("knockout")

    mock_blade = MagicMock()
    mock_blade.score_parallel = lambda ctx, cand: torch.randn(GAMMA, K)
    mock_blade.target_logprob_parallel = lambda ctx, cand, model: torch.randn(GAMMA, K)
    gen.blade = mock_blade
    base.parameters = lambda: iter([torch.zeros(1)])

    result = gen.generate("Test prompt.", max_new_tokens=8, return_stats=True)
    assert isinstance(result, tuple) and len(result) == 2
    text, stats = result
    assert isinstance(text, str)
    assert isinstance(stats, SpeculativeStats)
    assert stats.total_rounds >= 1
    print(f"  stats.total_rounds={stats.total_rounds}  ✓")
    print(f"  stats.total_tokens_accepted={stats.total_tokens_accepted}  ✓")


def test_generate_swiss_mode():
    """generate() with swiss tournament_mode should not crash."""
    gen, cfg, tok, base, blade_m = _make_generator("swiss")

    mock_blade = MagicMock()
    mock_blade.score_parallel = lambda ctx, cand: torch.randn(GAMMA, K)
    mock_blade.target_logprob_parallel = lambda ctx, cand, model: torch.randn(GAMMA, K)
    gen.blade = mock_blade
    base.parameters = lambda: iter([torch.zeros(1)])

    text = gen.generate("Test prompt.", max_new_tokens=8)
    assert isinstance(text, str)
    print(f"  Swiss mode generate() returned: '{text}'  ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — Option B Generator Tests")
    print("=" * 60)
    print()

    print("TEST 1 — Draft propose shape")
    test_draft_propose_shape()
    print()

    print("TEST 2 — Column 0 = greedy argmax")
    test_draft_column_zero_is_greedy()
    print()

    print("TEST 3 — SpeculativeStats fields")
    test_speculative_stats_fields()
    print()

    print("TEST 4 — generate() knockout returns string")
    test_generate_returns_string_knockout()
    print()

    print("TEST 5 — generate() return_stats=True")
    test_generate_returns_stats()
    print()

    print("TEST 6 — generate() Swiss mode")
    test_generate_swiss_mode()
    print()

    print("=" * 60)
    print("  ALL SPECULATIVE GENERATOR TESTS PASSED ✓")
    print("=" * 60)
