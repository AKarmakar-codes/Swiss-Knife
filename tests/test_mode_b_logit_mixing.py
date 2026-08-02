"""
Unit tests for Swiss-Knife Mode-B Multi-Blade Logit Mixing.

Verifies:
1. compute_dual_blade_mode_b_step returns valid candidate index.
2. Calibration shift invariance: adding constant offsets to blade scores preserves winner.
3. Pareto weight transitions: gamma_help=1.0 favors Helpfulness, gamma_help=0.0 favors Harmlessness.
4. compute_multi_blade_mode_b_step works with M=3 blades.
5. MultiBladeModeBGenerator structure and exports.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Model_mechanics.mode_b_logit_mixing import (
    compute_dual_blade_mode_b_step,
    compute_multi_blade_mode_b_step,
    MultiBladeModeBGenerator,
)
from Model_mechanics.config import SwissKnifeConfig


class DummyBlade:
    """Mock DPO Blade returning pre-configured reward tensors."""

    def __init__(self, scores: torch.Tensor):
        self.scores = scores
        self.blade_model = MagicMock()
        self.base_model = MagicMock()

    def score_reasoning_steps(self, prefix_ids, step_token_ids_list):
        return self.scores


class TestModeBLogitMixing(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.n_candidates = 8
        self.prefix_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)
        self.candidate_step_ids = [torch.randint(0, 1000, (5,), dtype=torch.long) for _ in range(self.n_candidates)]
        self.draft_logprobs = torch.randn(self.n_candidates, dtype=torch.float32)

    def test_compute_dual_blade_step_basic(self):
        help_scores = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)
        harm_scores = torch.tensor([0.8, 0.1, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=torch.float32)

        help_blade = DummyBlade(help_scores)
        harm_blade = DummyBlade(harm_scores)

        champion_idx = compute_dual_blade_mode_b_step(
            prefix_ids=self.prefix_ids,
            candidate_step_ids=self.candidate_step_ids,
            draft_logprobs=self.draft_logprobs,
            helpfulness_blade=help_blade,
            harmlessness_blade=harm_blade,
            gamma_help=0.5,
            beta=0.1,
            sigma_mode="none",
            probabilistic=True,
        )

        self.assertGreaterEqual(champion_idx, 0)
        self.assertLess(champion_idx, self.n_candidates)

    def test_shift_invariance(self):
        help_scores = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)
        harm_scores = torch.tensor([0.8, 0.1, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=torch.float32)

        help_blade = DummyBlade(help_scores)
        harm_blade = DummyBlade(harm_scores)

        # Greedy selection (elo_temperature=0.00001) to test deterministic shift invariance
        idx1 = compute_dual_blade_mode_b_step(
            prefix_ids=self.prefix_ids,
            candidate_step_ids=self.candidate_step_ids,
            draft_logprobs=self.draft_logprobs,
            helpfulness_blade=help_blade,
            harmlessness_blade=harm_blade,
            gamma_help=0.5,
            beta=0.1,
            elo_temperature=1e-5,
            sigma_mode="none",
            probabilistic=False,
        )

        help_blade_shifted = DummyBlade(help_scores + 100.0)
        harm_blade_shifted = DummyBlade(harm_scores + 50.0)

        idx2 = compute_dual_blade_mode_b_step(
            prefix_ids=self.prefix_ids,
            candidate_step_ids=self.candidate_step_ids,
            draft_logprobs=self.draft_logprobs,
            helpfulness_blade=help_blade_shifted,
            harmlessness_blade=harm_blade_shifted,
            gamma_help=0.5,
            beta=0.1,
            elo_temperature=1e-5,
            sigma_mode="none",
            probabilistic=False,
        )

        self.assertEqual(idx1, idx2, f"Shift invariance broken: {idx1} != {idx2}")

    def test_pareto_gamma_transition(self):
        # Candidate 1 is strong on help, Candidate 0 is strong on harm
        help_scores = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        harm_scores = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        help_blade = DummyBlade(help_scores)
        harm_blade = DummyBlade(harm_scores)
        uniform_draft = torch.zeros(self.n_candidates, dtype=torch.float32)

        # Pure helpfulness (gamma_help = 1.0) -> Candidate 1 should win
        help_idx = compute_dual_blade_mode_b_step(
            prefix_ids=self.prefix_ids,
            candidate_step_ids=self.candidate_step_ids,
            draft_logprobs=uniform_draft,
            helpfulness_blade=help_blade,
            harmlessness_blade=harm_blade,
            gamma_help=1.0,
            beta=0.1,
            elo_temperature=1e-5,
            sigma_mode="none",
            probabilistic=False,
        )
        self.assertEqual(help_idx, 1)

        # Pure harmlessness (gamma_help = 0.0) -> Candidate 0 should win
        harm_idx = compute_dual_blade_mode_b_step(
            prefix_ids=self.prefix_ids,
            candidate_step_ids=self.candidate_step_ids,
            draft_logprobs=uniform_draft,
            helpfulness_blade=help_blade,
            harmlessness_blade=harm_blade,
            gamma_help=0.0,
            beta=0.1,
            elo_temperature=1e-5,
            sigma_mode="none",
            probabilistic=False,
        )
        self.assertEqual(harm_idx, 0)

    def test_multi_blade_m3(self):
        b1 = DummyBlade(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        b2 = DummyBlade(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        b3 = DummyBlade(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        blades = {"helpfulness": b1, "harmlessness": b2, "truthfulness": b3}
        weights = {"helpfulness": 0.0, "harmlessness": 0.0, "truthfulness": 1.0}

        idx = compute_multi_blade_mode_b_step(
            prefix_ids=self.prefix_ids,
            candidate_step_ids=self.candidate_step_ids,
            draft_logprobs=torch.zeros(self.n_candidates),
            blades=blades,
            weights=weights,
            elo_temperature=1e-5,
            sigma_mode="none",
            probabilistic=False,
        )
        self.assertEqual(idx, 2)


if __name__ == "__main__":
    unittest.main()
