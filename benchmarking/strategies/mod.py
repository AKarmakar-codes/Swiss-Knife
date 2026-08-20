"""
Multi-Objective Decoding (MOD) Generation Strategy
==================================================

This module implements the Multi-Objective Decoding (MOD) baseline 
(Shi et al., 2024).

How it works:
MOD generates tokens by taking a linear combination of log-probability 
distributions from multiple specialized models (e.g., helpfulness and 
harmlessness). At each token step, it performs a forward pass through 
all the specified specialized models, extracts their log-probabilities 
for the next token, and combines them:

    log p_combined = w_1 * log p_model_1 + w_2 * log p_model_2 + ...

The next token is then sampled from this combined distribution.
This strategy tests the vulnerability of linear mixture methods to reward 
miscalibration compared to tournament-based selection.

Usage:
    models = [helpfulness_model, harmlessness_model]
    weights = [0.5, 0.5]
    generator = MODGenerator(cfg, tokenizer, models, weights)
    response, stats = generator.generate(prompt, max_new_tokens=80)
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from Model_mechanics.config import SwissKnifeConfig

logger = logging.getLogger(__name__)


@dataclass
class MODStats:
    """Statistics tracked during MOD generation."""
    strategy: str = "mod"
    total_tokens: int = 0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time_s, 3),
            "tokens_per_second": round(self.total_tokens / max(self.total_time_s, 1e-6), 2),
        }


class MODGenerator:
    """Multi-Objective Decoding (MOD) Generator."""

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        models: List[PreTrainedModel],
        weights: List[float],
        blade_names: Optional[List[str]] = None,
    ):
        """
        Initialize the MOD Generator.

        Args:
            cfg (SwissKnifeConfig): Configuration for generation settings.
            tokenizer (PreTrainedTokenizer): Tokenizer for the models.
            models (List[PreTrainedModel]): List of specialized models to combine.
            weights (List[float]): Weights for each model in the linear combination.
            blade_names (List[str], optional): Names of the blades corresponding to each model.
        """
        assert len(models) == len(weights), "Number of models must match number of weights."
        
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.models = models
        self.weights = weights
        self.blade_names = blade_names or ["helpfulness", "honesty", "harmlessness"]
        
        self.device = next(iter(models[0].parameters())).device
        
        logger.info(
            "MODGenerator initialized with %d models. Weights: %s | Blade names: %s",
            len(models), str(weights), str(self.blade_names)
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        blade_coefficients: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> str | Tuple[str, MODStats]:
        """
        Generate a response using MOD token-level linear combination.

        Args:
            prompt (str): The input text to condition generation on.
            max_new_tokens (int, optional): Maximum tokens to generate.
            verbose (bool): Whether to log detailed information.
            return_stats (bool): If True, returns a tuple of (generated_text, stats_object).
            The generated response string, optionally along with generation statistics.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        t_start = time.perf_counter()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        
        if verbose:
            logger.info("MOD: Starting generation combining %d models...", len(self.models))

        # Keep track of past key values for each model
        past_key_values_list = [None for _ in self.models]
        curr_input_ids = input_ids
        
        # Resolve active weights for this generation step based on blade_coefficients
        active_weights = list(self.weights)
        if blade_coefficients is not None:
            blade_names = getattr(self, "blade_names", ["helpfulness", "honesty", "harmlessness"])
            raw_w = [float(blade_coefficients.get(b, 0.0)) for b in blade_names[:len(self.models)]]
            total_w = sum(raw_w)
            if total_w > 0:
                active_weights = [w / total_w for w in raw_w]

        for step in range(max_tokens):
            combined_logits = None
            
            for i, model in enumerate(self.models):
                w_i = active_weights[i]
                if w_i <= 0.0:
                    continue

                if hasattr(model, "set_adapter") and self.blade_names and i < len(self.blade_names):
                    b_name = self.blade_names[i]
                    if hasattr(model, "peft_config") and b_name in model.peft_config:
                        model.set_adapter(b_name)

                outputs = model(
                    input_ids=curr_input_ids,
                    past_key_values=past_key_values_list[i],
                    use_cache=True,
                )
                past_key_values_list[i] = outputs.past_key_values
                logits = outputs.logits[0, -1, :]
                
                # Convert to log probabilities
                logprobs = F.log_softmax(logits, dim=-1)
                
                # Weight and accumulate
                weighted_logprobs = w_i * logprobs
                
                if combined_logits is None:
                    combined_logits = weighted_logprobs
                else:
                    combined_logits += weighted_logprobs
            
            # Apply temperature
            if self.cfg.temperature != 1.0 and self.cfg.temperature > 0.0:
                combined_logits = combined_logits / self.cfg.temperature
                
            # Sample next token
            probs = F.softmax(combined_logits, dim=-1)
            
            # Top-p (nucleus) sampling
            if self.cfg.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.cfg.top_p
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
                next_token_id = torch.argmax(combined_logits).unsqueeze(0)
                
            token_id = next_token_id.item()
            generated_tokens.append(token_id)
            
            if token_id == self.tokenizer.eos_token_id:
                break
                
            curr_input_ids = next_token_id.unsqueeze(0)
            
        elapsed = time.perf_counter() - t_start
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if verbose:
            logger.info(
                "MOD complete | tokens=%d | time=%.2fs | %.2f tok/s",
                len(generated_tokens), elapsed, len(generated_tokens) / max(elapsed, 1e-6)
            )
            
        if not return_stats:
            return response_text
            
        stats = MODStats(
            total_tokens=len(generated_tokens),
            total_time_s=elapsed,
        )
        return response_text, stats
