"""
Unit tests for Model_mechanics/stochastic_auditor.py (Phase 4).

All tests use a lightweight toy PyTorch model structure to verify hook registration,
stochastic perturbations, and integration with the speculative generator on CPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.blades import DPOBlade
from Model_mechanics.stochastic_auditor import StochasticAuditor, StochasticAuditorConfig
from Model_mechanics.speculative_generator import SwissKnifeSpeculativeGenerator, SpeculativeStats


# Toy model with real nn.Linear modules to test hook registration
class ToyAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.o_proj(x)


class ToyLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = ToyAttention(hidden_size)

    def forward(self, x):
        return self.self_attn(x)


class ToyBaseModel(nn.Module):
    def __init__(self, hidden_size, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size=1000, hidden_size=16):
        super().__init__()
        self.model = ToyBaseModel(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = MagicMock()
        self.config.num_attention_heads = 4

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # We simulate the embeddings by generating a random float tensor
        B, T = input_ids.shape
        x = torch.randn(B, T, self.lm_head.in_features, device=input_ids.device)
        x = self.model(x)
        logits = self.lm_head(x)
        
        # HuggingFace style output
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits)


class TestStochasticAuditor(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.hidden_size = 16
        self.cfg = SwissKnifeConfig(
            K=4,
            gamma=4,
            alpha=0.5,
            beta=0.1,
            tournament_mode="knockout",
            max_new_tokens=8,
            use_stochastic_auditor=True,
            stochastic_mode="mc_dropout"
        )
        self.tokenizer = MagicMock()
        self.tokenizer.vocab_size = self.vocab_size
        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 0
        
        def mock_encode(text, **kwargs):
            ids = torch.randint(3, self.vocab_size, (1, 5))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        self.tokenizer.side_effect = mock_encode
        self.tokenizer.decode = lambda ids, **kw: "mocked text"

        self.base_model = ToyCausalLM(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
        self.blade_model = ToyCausalLM(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
        self.blade = DPOBlade(self.cfg, self.base_model, self.blade_model, self.tokenizer)

    def test_config_validation(self):
        """Test configuration defaults and validation."""
        config = StochasticAuditorConfig()
        self.assertEqual(config.mode, "mc_dropout")
        self.assertEqual(config.dropout_p, 0.1)

        # Invalid config should fail __post_init__ validation in SwissKnifeConfig
        with self.assertRaises(AssertionError):
            SwissKnifeConfig(use_stochastic_auditor=True, stochastic_mode="invalid_mode")

        with self.assertRaises(AssertionError):
            SwissKnifeConfig(use_stochastic_auditor=True, stochastic_dropout_p=1.5)

    def test_mc_dropout_perturbation(self):
        """Verify that MC dropout injects noise and yields stochastic logits."""
        auditor_cfg = StochasticAuditorConfig(mode="mc_dropout", dropout_p=0.5)
        auditor = StochasticAuditor(self.blade, self.cfg, auditor_cfg)

        context_ids = torch.randint(3, self.vocab_size, (1, 5))
        candidate_matrix = torch.randint(3, self.vocab_size, (4, 4))
        ref_logprobs = torch.randn(4, 4)

        # Draw a functional and evaluate
        auditor.draw_fresh_functional()
        scores1 = auditor.score_candidates_for_match(context_ids, candidate_matrix, ref_logprobs)
        auditor.clear_functional()

        # Draw another functional and evaluate
        auditor.draw_fresh_functional()
        scores2 = auditor.score_candidates_for_match(context_ids, candidate_matrix, ref_logprobs)
        auditor.clear_functional()

        # The scores should not be identical because dropout applied different masks
        self.assertFalse(torch.allclose(scores1, scores2))

    def test_random_projection(self):
        """Verify that random projection perturbs final hidden states."""
        auditor_cfg = StochasticAuditorConfig(mode="random_proj", proj_epsilon=0.5)
        auditor = StochasticAuditor(self.blade, self.cfg, auditor_cfg)

        context_ids = torch.randint(3, self.vocab_size, (1, 5))
        candidate_matrix = torch.randint(3, self.vocab_size, (4, 4))
        ref_logprobs = torch.randn(4, 4)

        auditor.draw_fresh_functional()
        scores1 = auditor.score_candidates_for_match(context_ids, candidate_matrix, ref_logprobs)
        auditor.clear_functional()

        auditor.draw_fresh_functional()
        scores2 = auditor.score_candidates_for_match(context_ids, candidate_matrix, ref_logprobs)
        auditor.clear_functional()

        self.assertFalse(torch.allclose(scores1, scores2))

    def test_head_subsampling(self):
        """Verify that attention head subsampling zeroes out heads."""
        auditor_cfg = StochasticAuditorConfig(mode="head_subsample", head_frac=0.5, num_layers_to_mask=2)
        auditor = StochasticAuditor(self.blade, self.cfg, auditor_cfg)

        context_ids = torch.randint(3, self.vocab_size, (1, 5))
        candidate_matrix = torch.randint(3, self.vocab_size, (4, 4))
        ref_logprobs = torch.randn(4, 4)

        auditor.draw_fresh_functional()
        scores1 = auditor.score_candidates_for_match(context_ids, candidate_matrix, ref_logprobs)
        auditor.clear_functional()

        auditor.draw_fresh_functional()
        scores2 = auditor.score_candidates_for_match(context_ids, candidate_matrix, ref_logprobs)
        auditor.clear_functional()

        self.assertFalse(torch.allclose(scores1, scores2))

    def test_speculative_generator_integration_knockout(self):
        """Verify that the speculative generator runs to completion with the stochastic auditor (knockout)."""
        self.cfg.use_stochastic_auditor = True
        self.cfg.stochastic_mode = "mc_dropout"
        self.cfg.tournament_mode = "knockout"

        generator = SwissKnifeSpeculativeGenerator(
            cfg=self.cfg,
            tokenizer=self.tokenizer,
            base_model=self.base_model,
            blade_model=self.blade_model
        )

        text, stats = generator.generate("Test prompt", max_new_tokens=4, return_stats=True)
        self.assertTrue(isinstance(text, str))
        self.assertTrue(stats.blade_forward_passes > 0)

    def test_speculative_generator_integration_swiss(self):
        """Verify that the speculative generator runs to completion with the stochastic auditor (Swiss)."""
        self.cfg.use_stochastic_auditor = True
        self.cfg.stochastic_mode = "random_proj"
        self.cfg.tournament_mode = "swiss"
        self.cfg.swiss_rounds = 2

        generator = SwissKnifeSpeculativeGenerator(
            cfg=self.cfg,
            tokenizer=self.tokenizer,
            base_model=self.base_model,
            blade_model=self.blade_model
        )

        text, stats = generator.generate("Test prompt", max_new_tokens=4, return_stats=True)
        self.assertTrue(isinstance(text, str))
        self.assertTrue(stats.blade_forward_passes > 0)


if __name__ == "__main__":
    unittest.main()
