"""
DeAL (Decoding-time ALignment) Generation Strategy
==================================================

This module implements a simplified DeAL baseline (Huang et al., 2024).

How it works:
DeAL operates via A* search at the token level, optionally with greedy lookahead.
For this baseline comparison, we use a simplified version (lookahead=0), which 
reduces to token-level reranking.

At each generation step:
1. The base model generates the top-k candidate next tokens based on its 
   probability distribution.
2. Each of these k token extensions is scored using the reward model.
   (Here, the reward is the DPO blade's implicit reward: 
    r = beta * (log P_blade - log P_base)).
3. The candidate token with the highest combined score (or pure reward, 
   depending on the exact formulation; typically log P_base + alpha * r) 
   is selected as the next token.

Because our DPO blade can compute the reward for all vocabulary tokens in a 
single forward pass, we can compute this very efficiently without looping 
over the k candidates.

Usage:
    generator = DeALGenerator(cfg, tokenizer, base_model, blade_model)
    response, stats = generator.generate(prompt, max_new_tokens=80)
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from Model_mechanics.config import SwissKnifeConfig

logger = logging.getLogger(__name__)


@dataclass
class DeALStats:
    """Statistics tracked during DeAL generation."""
    strategy: str = "deal"
    total_tokens: int = 0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time_s, 3),
            "tokens_per_second": round(self.total_tokens / max(self.total_time_s, 1e-6), 2),
        }


class DeALGenerator:
    """Simplified DeAL Generator (lookahead=0, top-k reranking)."""

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        base_model: PreTrainedModel,
        blade_model: PreTrainedModel,
    ):
        """
        Initialize the DeAL Generator.

        Args:
            cfg (SwissKnifeConfig): Configuration containing settings like `top_k`.
            tokenizer (PreTrainedTokenizer): Tokenizer for the models.
            base_model (PreTrainedModel): The reference base language model.
            blade_model (PreTrainedModel): The adapter-loaded DPO blade model.
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.blade_model = blade_model
        
        self.device = next(iter(base_model.parameters())).device
        
        # We will use top_k for candidate selection
        self.k = self.cfg.top_k if self.cfg.top_k > 0 else 10
        
        logger.info(
            "DeALGenerator initialized: k=%d, alpha=%.2f, beta=%.2f",
            self.k, self.cfg.alpha, self.cfg.beta
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ) -> str | Tuple[str, DeALStats]:
        """
        Generate a response using DeAL top-k token reranking.

        Args:
            prompt (str): The input text to condition generation on.
            max_new_tokens (int, optional): Maximum tokens to generate.
            verbose (bool): Whether to log detailed information.
            return_stats (bool): If True, returns a tuple of (generated_text, stats_object).

        Returns:
            The generated response string, optionally along with generation statistics.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        
        # DeAL scoring weight
        steering_weight = self.cfg.alpha * self.cfg.beta
        
        t_start = time.perf_counter()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        
        if verbose:
            logger.info("DeAL: Starting generation with top-k=%d reranking...", self.k)

        past_key_values_base = None
        past_key_values_blade = None
        curr_input_ids = input_ids
        
        for step in range(max_tokens):
            # Forward pass base model
            outputs_base = self.base_model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values_base,
                use_cache=True,
            )
            past_key_values_base = outputs_base.past_key_values
            logits_base = outputs_base.logits[0, -1, :]
            
            # Forward pass blade model
            outputs_blade = self.blade_model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values_blade,
                use_cache=True,
            )
            past_key_values_blade = outputs_blade.past_key_values
            logits_blade = outputs_blade.logits[0, -1, :]
            
            logprobs_base = F.log_softmax(logits_base, dim=-1)
            logprobs_blade = F.log_softmax(logits_blade, dim=-1)
            
            # 1. Identify top-k candidate tokens according to base policy
            top_k_logprobs, top_k_indices = torch.topk(logprobs_base, self.k)
            
            # 2. Score these specific k candidates
            # Score = log P_base + alpha * (beta * (log P_blade - log P_base))
            # (or purely alpha * reward, but typically combined with base prior)
            candidate_base_lps = top_k_logprobs
            candidate_blade_lps = logprobs_blade[top_k_indices]
            
            scores = candidate_base_lps + steering_weight * (candidate_blade_lps - candidate_base_lps)
            
            # 3. Select the argmax among the top-k candidates
            best_candidate_idx = torch.argmax(scores)
            next_token_id = top_k_indices[best_candidate_idx].unsqueeze(0)
                
            token_id = next_token_id.item()
            generated_tokens.append(token_id)
            
            if token_id == self.tokenizer.eos_token_id:
                break
                
            curr_input_ids = next_token_id.unsqueeze(0)
            
        elapsed = time.perf_counter() - t_start
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if verbose:
            logger.info(
                "DeAL complete | tokens=%d | time=%.2fs | %.2f tok/s",
                len(generated_tokens), elapsed, len(generated_tokens) / max(elapsed, 1e-6)
            )
            
        if not return_stats:
            return response_text
            
        stats = DeALStats(
            total_tokens=len(generated_tokens),
            total_time_s=elapsed,
        )
        return response_text, stats
