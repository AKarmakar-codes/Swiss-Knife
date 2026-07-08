"""
Unit tests for Model_mechanics/gsi_elo.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from unittest.mock import MagicMock, patch
from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.gsi_elo import GSIEloStats, GSIEloGenerator

VOCAB_SIZE = 1000
PROMPT_LEN = 10

def _make_mock_model(vocab_size: int = VOCAB_SIZE):
    mock = MagicMock()
    def _forward(input_ids, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        out = MagicMock()
        out.logits = torch.randn(B, T, vocab_size)
        return out
    mock.side_effect = _forward
    mock.__call__ = _forward
    mock.parameters = lambda: iter([torch.zeros(1)])
    return mock

def _make_mock_tokenizer(vocab_size: int = VOCAB_SIZE, eos_id: int = 2):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = eos_id
    tok.pad_token_id = 0
    def _encode(text, return_tensors=None, **kwargs):
        ids = torch.randint(3, vocab_size, (1, PROMPT_LEN))
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return ids.squeeze(0).tolist()
    tok.side_effect = _encode
    tok.__call__ = _encode
    tok.decode = lambda ids, **kw: "Mocked step completion.\n\n"
    return tok

def test_gsi_elo_stats():
    stats = GSIEloStats()
    assert stats.total_steps == 0
    assert stats.acceptance_rate == 0.0
    
    stats.total_steps = 5
    stats.accepted_steps = 4
    stats.rejected_steps = 1
    stats.total_tokens = 20
    stats.total_time_s = 2.0
    
    assert stats.acceptance_rate == 0.8
    assert stats.tokens_per_second == 10.0
    d = stats.to_dict()
    assert d["strategy"] == "gsi_elo"
    assert d["acceptance_rate"] == 0.8
    print("  ✓ GSIEloStats works correctly")

def test_gsi_elo_generator():
    cfg = SwissKnifeConfig(
        generation_mode="gsi_elo",
        gsi_n=4,
        alpha=0.5,
        beta=0.1,
        elo_rounds=3,
        gsi_threshold=0.0,
        max_new_tokens=20,
    )
    drafter_model = _make_mock_model()
    drafter_tokenizer = _make_mock_tokenizer()
    verifier_model = _make_mock_model()
    verifier_tokenizer = _make_mock_tokenizer()
    blade_model = _make_mock_model()
    
    # Mock compute_logprob from evaluation.retokenisation_llama_to_qwen
    with patch("Model_mechanics.gsi_elo.compute_logprob", return_value=0.5):
        generator = GSIEloGenerator(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=blade_model,
        )
        
        # Mock generator._sample_reasoning_steps
        generator._sample_reasoning_steps = MagicMock(return_value=(
            [torch.tensor([1, 2, 3]) for _ in range(4)],
            ["Step 1\n\n", "Step 2\n\n", "Step 3\n\n", "Step 4\n\n"]
        ))
        
        # Mock DPOBlade score_reasoning_steps and compute_step_draft_logprobs
        generator.blade.score_reasoning_steps = MagicMock(return_value=torch.tensor([0.2, 0.4, 0.1, 0.3]))
        generator.blade.compute_step_draft_logprobs = MagicMock(return_value=torch.tensor([-0.1, -0.2, -0.05, -0.15]))
        
        output, stats = generator.generate("Mock prompt.", max_new_tokens=15, return_stats=True)
        
        assert isinstance(output, str)
        assert isinstance(stats, GSIEloStats)
        assert stats.total_steps >= 1
        print("  ✓ GSIEloGenerator runs and generates text correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — GSI Elo Generator Tests")
    print("=" * 60)
    print()
    test_gsi_elo_stats()
    test_gsi_elo_generator()
    print("=" * 60)
    print("  ALL GSI ELO TESTS PASSED ✓")
    print("=" * 60)
