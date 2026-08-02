"""
Best-of-N (BoN) Generation Strategy
===================================

This module implements the Best-of-N (BoN) sampling baseline, a fundamental
strategy for decode-time alignment (Nakano et al., 2021). 

How it works:
1. For a given prompt, the generator samples `N` distinct, complete candidate
   responses from the base model (e.g., Qwen 2.5) using sampling with temperature.
2. Once all `N` candidate responses are generated, the alignment blade (e.g., a
   DPO adapter acting as an implicit reward model) scores each candidate.
3. The candidate that achieves the highest reward according to the blade is 
   selected as the final response.

This strategy requires generating full trajectories before any selection occurs, 
making it computationally expensive but straightforward. It serves as a universal
baseline for comparison against more sophisticated token-level or step-level 
search methods.

Usage:
    generator = BestOfNGenerator(cfg, tokenizer, base_model, blade_model)
    response, stats = generator.generate(prompt, max_new_tokens=80)
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.blades import DPOBlade

logger = logging.getLogger(__name__)


@dataclass
class BestOfNStats:
    """Statistics tracked during Best-of-N generation."""
    strategy: str = "bon"
    total_candidates: int = 0
    total_time_s: float = 0.0
    best_reward: float = 0.0
    all_rewards: list = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_candidates": self.total_candidates,
            "total_time_s": round(self.total_time_s, 3),
            "best_reward": round(self.best_reward, 6),
            "all_rewards": [round(r, 6) for r in self.all_rewards] if self.all_rewards else [],
        }


class BestOfNGenerator:
    """Best-of-N (BoN) Generator."""

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        base_model: PreTrainedModel,
        blade_model,
    ):
        """
        Initialize the BoN Generator.

        Args:
            cfg (SwissKnifeConfig): Configuration containing generation settings like `gsi_n`.
            tokenizer (PreTrainedTokenizer): Tokenizer for the base model.
            base_model (PreTrainedModel): The underlying language model to generate from.
            blade_model: The DPO adapter or reward model to score responses.
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.base_model = base_model
        
        # Build the scoring blade
        self.blade = DPOBlade(cfg, base_model, blade_model, tokenizer)
        self.device = next(iter(base_model.parameters())).device
        
        logger.info(
            "BestOfNGenerator initialized: N=%d, temp=%.2f",
            self.cfg.gsi_n, self.cfg.temperature
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ) -> str | Tuple[str, BestOfNStats]:
        """
        Generate a response using Best-of-N sampling.

        Args:
            prompt (str): The input text to condition generation on.
            max_new_tokens (int, optional): Maximum tokens to generate. Overrides config if set.
            verbose (bool): Whether to log detailed information.
            return_stats (bool): If True, returns a tuple of (generated_text, stats_object).

        Returns:
            The best generated response string, optionally along with generation statistics.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        
        t_start = time.perf_counter()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prefix_ids = inputs["input_ids"]
        prefix_len = prefix_ids.shape[1]
        
        # Step 1: Generate N candidates independently in parallel
        batch_ids = prefix_ids.expand(n, -1).contiguous()
        batch_mask = torch.ones_like(batch_ids)
        
        if verbose:
            logger.info("BoN: Generating %d candidate responses...", n)
            
        outputs = self.base_model.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Step 2: Score candidates using the alignment blade
        candidates_text = []
        candidates_ids = []
        rewards = []
        
        for i in range(n):
            new_tokens = outputs[i, prefix_len:]
            
            # Truncate at EOS if present
            eos_positions = (new_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                new_tokens = new_tokens[:eos_positions[0]]
                
            candidates_ids.append(new_tokens)
            candidates_text.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
            
            # Score this candidate sequence
            # DPOBlade.score_reasoning_steps takes a list of step sequences.
            # For BoN, the entire response is treated as one "step".
            r_blade = self.blade.score_reasoning_steps(
                prefix_ids, [new_tokens]
            )[0].item()
            rewards.append(r_blade)
            
        # Step 3: Select the candidate with the highest reward
        best_idx = rewards.index(max(rewards))
        best_response = candidates_text[best_idx]
        best_reward = rewards[best_idx]
        
        elapsed = time.perf_counter() - t_start
        
        if verbose:
            logger.info(
                "BoN complete | N=%d | best_reward=%.4f | time=%.2fs",
                n, best_reward, elapsed
            )
            
        if not return_stats:
            return best_response
            
        stats = BestOfNStats(
            total_candidates=n,
            total_time_s=elapsed,
            best_reward=best_reward,
            all_rewards=rewards,
        )
        return best_response, stats
