import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch
import torch

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.gsi_softmax import GSISoftmaxGenerator
from Model_mechanics.blades import DPOBlade
from Model_mechanics.rrm_judge import RRMJudge


class TestRRMPoolAndParallelGSI(unittest.TestCase):

    def setUp(self):
        self.cfg = SwissKnifeConfig(
            generation_mode="rrm_pool",
            gsi_n=3,
            max_new_tokens=20,
            drafter_model_id="Qwen/Qwen2.5-3B-Instruct"
        )
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 151643
        self.tokenizer.eos_token_id = 151645
        
        # Mock tokenizer encode/decode
        def mock_encode(text, *args, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return torch.tensor([[1, 2, 3]], dtype=torch.long)
            return [1, 2, 3]
        self.tokenizer.encode.side_effect = mock_encode
        self.tokenizer.decode.side_effect = lambda tokens, *args, **kwargs: "hello world \n\n"
        
        # When calling verifier_tokenizer(p, return_tensors="pt")
        def mock_call(p, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long)
            }
        self.tokenizer.side_effect = mock_call

        # Mock models
        self.mock_drafter = MagicMock()
        self.mock_verifier = MagicMock()
        self.mock_blade_model = MagicMock()

        # Set up parameters mock to return a CPU parameter iterator
        param = torch.nn.Parameter(torch.zeros(1))
        self.mock_drafter.parameters = lambda: iter([param])
        self.mock_verifier.parameters = lambda: iter([param])
        self.mock_blade_model.parameters = lambda: iter([param])

        # Mock generate for drafter: return padded prefixes + generated step
        def mock_generate(*args, **kwargs):
            # Shape should be [Batch, prefix_len + generated_len]
            input_ids = kwargs.get("input_ids")
            B = input_ids.shape[0]
            # append some token ids
            gen = torch.full((B, 5), 100, dtype=torch.long, device=input_ids.device)
            return torch.cat([input_ids, gen], dim=1)

        self.mock_drafter.generate = mock_generate
        self.mock_verifier.generate = mock_generate

        # Mock forward pass logits for logprob computation
        def mock_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            B, T = input_ids.shape
            # Return dummy logits of shape [B, T, 200000]
            logits = torch.zeros(B, T, 152064, dtype=torch.float, device=input_ids.device)
            out = MagicMock()
            out.logits = logits
            return out

        self.mock_drafter.side_effect = mock_forward
        self.mock_verifier.side_effect = mock_forward
        self.mock_blade_model.side_effect = mock_forward

    def test_parallel_gsi_softmax_generation(self):
        """Verify that generate_batched completes and produces the correct shapes/types under mocks."""
        generator = GSISoftmaxGenerator.__new__(GSISoftmaxGenerator)
        generator.cfg = self.cfg
        generator.drafter_model = self.mock_drafter
        generator.verifier_model = self.mock_verifier
        generator.verifier_tokenizer = self.tokenizer
        generator.drafter_tokenizer = self.tokenizer
        generator.verifier_device = torch.device("cpu")
        generator.drafter_device = torch.device("cpu")

        # Set up a mock blade
        mock_blade = MagicMock(spec=DPOBlade)
        mock_blade.base_model = self.mock_verifier
        mock_blade.blade_model = self.mock_blade_model
        mock_blade.tokenizer = self.tokenizer
        mock_blade.beta = 0.1
        generator.blade = mock_blade

        prompts = ["Test prompt 1", "Test prompt 2"]
        
        calls = [0]
        def mock_decode(tokens, *args, **kwargs):
            calls[0] += 1
            if calls[0] > 6:
                return "finished <|endoftext|>"
            return f"step {calls[0]} \n\n"

        with patch.object(self.tokenizer, 'decode', side_effect=mock_decode):
            outputs = generator.generate_batched(
                prompts=prompts,
                max_new_tokens=10,
                verbose=False
            )

        self.assertEqual(len(outputs), 2)
        self.assertTrue(isinstance(outputs[0], str))
        self.assertTrue(isinstance(outputs[1], str))

    def test_rrm_judge_and_tournament(self):
        """Verify the ELO tournament selection of RRMJudge works correctly."""
        mock_rrm_model = MagicMock()
        
        # Mock generate of RRM model to return winner text
        def rrm_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            # We want to return text that contains \boxed{Assistant 1} or \boxed{Assistant 2}
            # Token ID 9999 represents "\boxed{Assistant 1}" in decoded text
            return torch.cat([input_ids, torch.tensor([[9999]], dtype=torch.long, device=input_ids.device)], dim=1)
            
        mock_rrm_model.generate = rrm_generate
        
        # Mock parameters to get device
        param = torch.nn.Parameter(torch.zeros(1))
        mock_rrm_model.parameters.return_value = [param]

        # Mock decode
        self.tokenizer.decode.side_effect = lambda *args, **kwargs: "Based on analysis, \\boxed{Assistant 1} is better."

        judge = RRMJudge(mock_rrm_model, self.tokenizer)
        winner = judge.run_tournament(
            prompt="Prompt",
            candidates=["Candidate 1", "Candidate 2", "Candidate 3"],
            temperature=1.0
        )
        self.assertTrue(0 <= winner < 3)

    def test_rrm_candidate_pool_generator(self):
        """Verify that RRMCandidatePoolGenerator successfully coordinates parallel GSI and RRM tournament."""
        from Model_mechanics.rrm_candidate_pool import RRMCandidatePoolGenerator

        mock_rrm_model = MagicMock()
        param = torch.nn.Parameter(torch.zeros(1))
        mock_rrm_model.parameters.return_value = [param]

        # Mock generate of RRM model to return winner text
        def rrm_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            return torch.cat([input_ids, torch.tensor([[9999]], dtype=torch.long, device=input_ids.device)], dim=1)
        mock_rrm_model.generate = rrm_generate

        pool_generator = RRMCandidatePoolGenerator(
            cfg=self.cfg,
            drafter_model=self.mock_drafter,
            drafter_tokenizer=self.tokenizer,
            verifier_model=self.mock_verifier,
            verifier_tokenizer=self.tokenizer,
            rrm_model=mock_rrm_model,
            rrm_tokenizer=self.tokenizer,
            blade_model=self.mock_blade_model,
        )

        calls = [0]
        def mock_decode(tokens, *args, **kwargs):
            calls[0] += 1
            if calls[0] > 9:
                return "finished \\boxed{Assistant 1} <|endoftext|>"
            return f"step {calls[0]} \n\n"

        with patch.object(self.tokenizer, 'decode', side_effect=mock_decode):
            champion, candidates = pool_generator.generate(
                prompt="Explain something.",
                max_new_tokens=10,
                verbose=False
            )

        self.assertEqual(len(candidates), self.cfg.rrm_n_candidates)
        self.assertTrue(isinstance(champion, str))


if __name__ == "__main__":
    unittest.main()
