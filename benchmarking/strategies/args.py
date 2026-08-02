"""
ARGS (Alignment as Reward-Guided Search) Generation Strategy
============================================================

This module implements a token-level steering baseline inspired by ARGS 
(Khanov et al., 2024).

How it works:
Unlike Best-of-N which generates full trajectories, or Swiss Knife which 
operates on step-level candidate chunks, ARGS operates at the individual 
token level. At each decoding step, it computes the next-token probability 
distribution from the base model, and adjusts these logits using a reward 
signal.

For a DPO blade where the implicit reward for a token `t` is defined as:
    r(x, t) = beta_dpo * (log P_blade(t|x) - log P_base(t|x))
Adding `alpha * r(x, t)` to the base logits results in:
    logit(t) = log P_base(t|x) + alpha * beta_dpo * (log P_blade(t|x) - log P_base(t|x))

This is implemented by running a forward pass on both the base model and 
the blade model at each token step, combining their logits, and then 
sampling the next token.

Usage:
    generator = ARGSGenerator(cfg, tokenizer, base_model, blade_model)
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
class ARGSStats:
    """Statistics tracked during ARGS generation."""
    strategy: str = "args"
    total_tokens: int = 0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time_s, 3),
            "tokens_per_second": round(self.total_tokens / max(self.total_time_s, 1e-6), 2),
        }


class ARGSGenerator:
    """ARGS Token-level Steering Generator."""

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        base_model: PreTrainedModel,
        blade_model: PreTrainedModel,
    ):
        """
        Initialize the ARGS Generator.

        Args:
            cfg (SwissKnifeConfig): Configuration containing steering settings.
            tokenizer (PreTrainedTokenizer): Tokenizer for both models.
            base_model (PreTrainedModel): The reference base language model.
            blade_model (PreTrainedModel): The adapter-loaded DPO blade model.
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.blade_model = blade_model
        
        self.device = next(iter(base_model.parameters())).device
        
        logger.info(
            "ARGSGenerator initialized: alpha (steering)=%.2f, dpo_beta=%.2f",
            self.cfg.alpha, self.cfg.beta
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ) -> str | Tuple[str, ARGSStats]:
        """
        Generate a response using ARGS token-level steering.

        Args:
            prompt (str): The input text to condition generation on.
            max_new_tokens (int, optional): Maximum tokens to generate. Overrides config if set.
            verbose (bool): Whether to log detailed information.
            return_stats (bool): If True, returns a tuple of (generated_text, stats_object).

        Returns:
            The generated response string, optionally along with generation statistics.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        
        # Steering multiplier (alpha * beta_dpo)
        steering_weight = self.cfg.alpha * self.cfg.beta
        
        t_start = time.perf_counter()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        
        if verbose:
            logger.info("ARGS: Starting token-by-token generation...")

        # Keep track of past key values for efficient generation
        past_key_values_base = None
        past_key_values_blade = None
        
        # Initial forward pass with full prompt
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
            
            # Convert to log probabilities to compute the reward correctly
            # log P = log_softmax(logits)
            logprobs_base = F.log_softmax(logits_base, dim=-1)
            logprobs_blade = F.log_softmax(logits_blade, dim=-1)
            
            # Compute steered logits
            # logit = log P_base + alpha * beta * (log P_blade - log P_base)
            steered_logits = logprobs_base + steering_weight * (logprobs_blade - logprobs_base)
            
            # Apply temperature
            if self.cfg.temperature != 1.0 and self.cfg.temperature > 0.0:
                steered_logits = steered_logits / self.cfg.temperature
                
            # Sample next token
            probs = F.softmax(steered_logits, dim=-1)
            
            # Top-p (nucleus) sampling
            if self.cfg.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.cfg.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Top-k sampling
            if self.cfg.top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, self.cfg.top_k)
                probs_new = torch.zeros_like(probs)
                probs_new.scatter_(-1, top_k_indices, top_k_probs)
                probs = probs_new / probs_new.sum(dim=-1, keepdim=True)
            
            # Multinomial sampling
            if self.cfg.temperature > 0.0:
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(steered_logits).unsqueeze(0)
                
            token_id = next_token_id.item()
            generated_tokens.append(token_id)
            
            if token_id == self.tokenizer.eos_token_id:
                break
                
            # Prepare next input
            curr_input_ids = next_token_id.unsqueeze(0)
            
        elapsed = time.perf_counter() - t_start
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if verbose:
            logger.info(
                "ARGS complete | tokens=%d | time=%.2fs | %.2f tok/s",
                len(generated_tokens), elapsed, len(generated_tokens) / max(elapsed, 1e-6)
            )
            
        if not return_stats:
            return response_text
            
        stats = ARGSStats(
            total_tokens=len(generated_tokens),
            total_time_s=elapsed,
        )
        return response_text, stats
