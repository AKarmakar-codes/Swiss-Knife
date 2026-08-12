"""
Unit tests for Model_mechanics/elo_swiss_dual_blade_mode_b.py (Multi-Blade Mode B Generator).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Model_mechanics.blades import DPOBlade
from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.elo_swiss_multi_blade_mode_b import (
    EloSwissMultiBladeModeBGenerator,
)

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
    tok.decode = lambda ids, **kw: "Step complete."
    return tok


def test_multi_blade_init_and_filtering():
    cfg = SwissKnifeConfig(gsi_n=4, max_new_tokens=10)
    drafter_m = _make_mock_model()
    drafter_t = _make_mock_tokenizer()
    verifier_m = _make_mock_model()
    verifier_t = _make_mock_tokenizer()

    b1_m = _make_mock_model()
    b2_m = _make_mock_model()
    b3_m = _make_mock_model()

    blade_models = {
        "helpfulness": b1_m,
        "harmlessness": b2_m,
        "honesty": b3_m,
    }

    gen = EloSwissMultiBladeModeBGenerator(
        cfg=cfg,
        drafter_model=drafter_m,
        drafter_tokenizer=drafter_t,
        verifier_model=verifier_m,
        verifier_tokenizer=verifier_t,
        blade_models=blade_models,
    )

    assert len(gen.blades) == 3
    assert "helpfulness" in gen.blades
    assert "harmlessness" in gen.blades
    assert "honesty" in gen.blades


def test_individual_cbn_normalization_and_generation():
    cfg = SwissKnifeConfig(gsi_n=3, max_new_tokens=5, sigma_mode="min_entropy", probabilistic=True)
    drafter_m = _make_mock_model()
    drafter_t = _make_mock_tokenizer()
    verifier_m = _make_mock_model()
    verifier_t = _make_mock_tokenizer()

    mock_blade_help = MagicMock(spec=DPOBlade)
    mock_blade_harm = MagicMock(spec=DPOBlade)
    mock_blade_hons = MagicMock(spec=DPOBlade)

    blades = {
        "helpfulness": mock_blade_help,
        "harmlessness": mock_blade_harm,
        "honesty": mock_blade_hons,
    }

    gen = EloSwissMultiBladeModeBGenerator(
        cfg=cfg,
        drafter_model=drafter_m,
        drafter_tokenizer=drafter_t,
        verifier_model=verifier_m,
        verifier_tokenizer=verifier_t,
        blades=blades,
    )

    # Mock candidate step sampling
    step_ids = [torch.tensor([10, 11]), torch.tensor([12, 13]), torch.tensor([14, 15])]
    step_texts = ["step 1", "step 2", "step 3"]
    gen._sample_reasoning_steps = MagicMock(return_value=(step_ids, step_texts))

    # Mock estimate_mu_sigma return values per blade
    def _mock_estimate(prefix_ids, step_token_ids_list, blade, sigma_mode, **kwargs):
        n_cands = len(step_token_ids_list)
        if blade == mock_blade_help:
            # rewards [1.0, 2.0, 3.0], sigmas [0.1, 0.2, 0.3]
            mu = torch.tensor([1.0, 2.0, 3.0])
            sigma = torch.tensor([0.1, 0.2, 0.3])
        elif blade == mock_blade_harm:
            # rewards [10.0, 20.0, 30.0] (different scale!), sigmas [1.0, 2.0, 3.0]
            mu = torch.tensor([10.0, 20.0, 30.0])
            sigma = torch.tensor([1.0, 2.0, 3.0])
        else:
            # rewards [100.0, 200.0, 300.0]
            mu = torch.tensor([100.0, 200.0, 300.0])
            sigma = torch.tensor([10.0, 20.0, 30.0])
        return mu, sigma

    with patch("Model_mechanics.elo_swiss_multi_blade_mode_b.estimate_mu_sigma", side_effect=_mock_estimate):
        with patch("Model_mechanics.elo_swiss_multi_blade_mode_b.compute_logprobs_batched", return_value=[-1.0, -2.0, -1.5]):
            with patch("Model_mechanics.elo_swiss_multi_blade_mode_b.compute_logprob", return_value=-0.5):
                # Pass weight for helpfulness and harmlessness, leave honesty unspecified (weight 0.0)
                text, stats = gen.generate(
                    prompt="Hello test",
                    max_new_tokens=5,
                    return_stats=True,
                    blade_coefficients={"helpfulness": 0.6, "harmlessness": 0.4, "honesty": 0.0},
                )

                assert isinstance(text, str)
                assert stats.total_steps >= 1
                assert len(stats.step_details) >= 1
                first_step = stats.step_details[0]
                assert "candidate_mus" in first_step
                assert "candidate_sigmas" in first_step
                assert "champion_sigma_rank" in first_step


def test_dual_blade_special_case():
    cfg = SwissKnifeConfig(gsi_n=2, max_new_tokens=5)
    drafter_m = _make_mock_model()
    drafter_t = _make_mock_tokenizer()
    verifier_m = _make_mock_model()
    verifier_t = _make_mock_tokenizer()

    b1_m = _make_mock_model()
    b2_m = _make_mock_model()

    gen_dual = EloSwissMultiBladeModeBGenerator(
        cfg=cfg,
        drafter_model=drafter_m,
        drafter_tokenizer=drafter_t,
        verifier_model=verifier_m,
        verifier_tokenizer=verifier_t,
        blade_models={"helpfulness": b1_m, "harmlessness": b2_m},
    )

    assert len(gen_dual.blades) == 2
    assert "helpfulness" in gen_dual.blades
    assert "harmlessness" in gen_dual.blades

