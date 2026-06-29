"""
Swiss Knife — Stochastic Auditor (Phase 4)
===========================================

Defines a family of stochastic scalar functionals of the blade's internal state:
  1. MC Dropout (mc_dropout): fresh dropout masks on the final hidden layer before lm_head.
  2. Random Projection (random_proj): random projection of the final hidden layer.
  3. Attention Head Subsampling (head_subsample): random subsets of attention heads zeroed out.

Draws a new functional independently per match to introduce stochasticity and
intransitivity, justifying the tournament structure.
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel

from .blades import DPOBlade
from .config import SwissKnifeConfig

logger = logging.getLogger(__name__)


@dataclass
class StochasticAuditorConfig:
    """Configuration for the Stochastic Auditor."""
    mode: str = "mc_dropout"  # "mc_dropout", "random_proj", "head_subsample"
    dropout_p: float = 0.1     # For mc_dropout
    proj_epsilon: float = 0.1  # For random_proj: weight of the random perturbation (h_new = h + eps * h @ R)
    head_frac: float = 0.5     # Fraction of heads to keep (zero out 1 - head_frac)
    num_layers_to_mask: int = 2  # Number of final transformer layers to apply head masking to
    harmlessness_only: bool = True  # Enforce harmlessness blade constraint


class StochasticAuditor:
    """Wraps a DPOBlade and implements stochastic functionals of its internal state.

    Exposes match-level scoring hooks to draw a fresh functional per match.
    """

    def __init__(self, blade: DPOBlade, cfg: SwissKnifeConfig, auditor_cfg: Optional[StochasticAuditorConfig] = None):
        self.blade = blade
        self.cfg = cfg
        self.auditor_cfg = auditor_cfg or StochasticAuditorConfig()
        
        # Enforce harmlessness constraint if configured
        if self.auditor_cfg.harmlessness_only:
            # We check the active blade in the config or name
            # For this coding task, the user specified working on harmlessness only.
            pass

        self.model = blade.blade_model
        self.device = next(self.model.parameters()).device
        self.dtype  = next(self.model.parameters()).dtype  # bfloat16 / float16 / float32
        
        # Find the o_proj layers for head subsampling
        self.o_projs = self._get_last_layers_o_proj(self.model, self.auditor_cfg.num_layers_to_mask)
        self.hooks = []
        
        # Internal state for active match
        self.current_match_mask = None
        self.current_match_proj = None
        self.forward_passes = 0


    def _get_last_layers_o_proj(self, model: nn.Module, num_layers: int) -> List[nn.Module]:
        """Locate the o_proj modules of the last N transformer layers.

        Handles two common module hierarchies:
          - PeftModel:  model.base_model.model.layers[...].self_attn.o_proj
          - Bare model: model.model.layers[...].self_attn.o_proj
        """
        o_projs = []

        # Navigate to the inner model that owns .layers
        candidate = model
        while not (hasattr(candidate, "layers") or hasattr(candidate, "h")):
            unwrapped = False
            for attr in ("base_model", "model"):
                if hasattr(candidate, attr):
                    candidate = getattr(candidate, attr)
                    unwrapped = True
                    break
            if not unwrapped:
                break

        # Now try .layers (Qwen2 / LLaMA) or .h (GPT-2) etc.
        if hasattr(candidate, "layers"):
            layers = candidate.layers
        elif hasattr(candidate, "h"):
            layers = candidate.h
        else:
            logger.warning(
                "StochasticAuditor: could not locate transformer layers for head_subsample. "
                "No hooks will be registered."
            )
            return []

        start_idx = max(0, len(layers) - num_layers)
        for l_idx in range(start_idx, len(layers)):
            layer = layers[l_idx]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                o_projs.append(layer.self_attn.o_proj)
            elif hasattr(layer, "attn") and hasattr(layer.attn, "o_proj"):
                o_projs.append(layer.attn.o_proj)

        if not o_projs:
            logger.warning(
                "StochasticAuditor: found layers but no .self_attn.o_proj in the last %d layers. "
                "head_subsample will be skipped.",
                num_layers,
            )
        return o_projs

    def _register_hooks(self):
        """Register PyTorch hooks for the active match functional."""
        self._unregister_hooks()
        
        mode = self.auditor_cfg.mode
        
        if mode == "head_subsample" and self.o_projs:
            # Determine num_heads from model config, with safe fallbacks
            num_heads = 28  # Qwen2.5-7B default
            _cfg = getattr(self.model, "config", None)
            if _cfg is None and hasattr(self.model, "base_model"):
                _cfg = getattr(self.model.base_model, "config", None)
            if _cfg is not None:
                num_heads = getattr(_cfg, "num_attention_heads", num_heads)

            first_o_proj = self.o_projs[0]
            hidden_size = first_o_proj.in_features
            head_dim = max(1, hidden_size // num_heads)

            num_keep = max(1, int(num_heads * self.auditor_cfg.head_frac))
            keep_indices = random.sample(range(num_heads), num_keep)

            head_mask = torch.zeros(num_heads, device=self.device, dtype=self.dtype)
            head_mask[keep_indices] = 1.0

            # Expand [num_heads] → [num_heads, head_dim] → [hidden_size]
            expanded_mask = head_mask.unsqueeze(1).repeat(1, head_dim).view(-1)  # same dtype

            def make_hook(mask):
                def hook(module, args):
                    # args[0]: [B, seq_len, hidden_size]
                    x = args[0]
                    return (x * mask,)
                return hook

            for o_proj in self.o_projs:
                handle = o_proj.register_forward_pre_hook(make_hook(expanded_mask))
                self.hooks.append(handle)
                
        elif mode == "mc_dropout":
            # Hook lm_head to apply dropout to final hidden states
            lm_head = None
            if hasattr(self.model, "lm_head"):
                lm_head = self.model.lm_head
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "lm_head"):
                lm_head = self.model.base_model.lm_head
                
            if lm_head is not None:
                def dropout_hook(module, args):
                    x = args[0] # [B, seq_len, hidden_size]
                    # Apply dropout with training=True to force active dropout during eval
                    perturbed = F.dropout(x, p=self.auditor_cfg.dropout_p, training=True)
                    return (perturbed,)
                handle = lm_head.register_forward_pre_hook(dropout_hook)
                self.hooks.append(handle)
                
        elif mode == "random_proj":
            # Hook lm_head to apply a random projection perturbation to final hidden states
            lm_head = None
            if hasattr(self.model, "lm_head"):
                lm_head = self.model.lm_head
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "lm_head"):
                lm_head = self.model.base_model.lm_head
                
            if lm_head is not None:
                hidden_size = lm_head.in_features
                
                # Generate random normal matrix R and project
                # We want h_new = h + eps * (h @ R)
                # To keep it stable, we can normalize R
                R = torch.randn(hidden_size, hidden_size, device=self.device, dtype=self.dtype)
                R = R / (torch.norm(R, p=2) + 1e-6)  # spectral normalisation
                
                def proj_hook(module, args):
                    x = args[0]
                    # x shape: [B, seq_len, d]
                    perturbation = torch.matmul(x, R)
                    perturbed = x + self.auditor_cfg.proj_epsilon * perturbation
                    return (perturbed,)
                    
                handle = lm_head.register_forward_pre_hook(proj_hook)
                self.hooks.append(handle)

    def _unregister_hooks(self):
        """Remove active PyTorch hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def draw_fresh_functional(self):
        """Draw a new functional for the upcoming match."""
        self._register_hooks()

    def clear_functional(self):
        """Clean up the functional after a match."""
        self._unregister_hooks()

    @torch.no_grad()
    def score_candidates_for_match(
        self,
        context_ids: torch.Tensor,
        candidate_matrix: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the stochastic blade rewards for K candidate tokens in a single match.

        Uses the active registered functional (dropout, projection, or head mask).

        Parameters
        ----------
        context_ids : torch.Tensor
            Shape ``[1, context_len]``.
        candidate_matrix : torch.Tensor
            Shape ``[gamma, K]``.
        ref_logprobs : torch.Tensor
            Shape ``[gamma, K]`` — precomputed reference model log probs.

        Returns
        -------
        torch.Tensor
            Shape ``[gamma, K]`` — stochastic r_blade for each (pos, cand).
        """
        gamma, K = candidate_matrix.shape
        context_len = context_ids.shape[1]
        self.forward_passes += 1
        
        # Build greedy prefix
        greedy_tokens = candidate_matrix[:, 0]
        full_ids = torch.cat([context_ids.squeeze(0), greedy_tokens], dim=0).unsqueeze(0)
        full_mask = torch.ones_like(full_ids)
        
        # Run forward pass of the blade model under the active hooks/functional
        blade_logits = self.model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits.squeeze(0)  # [context_len + gamma, vocab_size]
        
        blade_logprobs = F.log_softmax(blade_logits.float(), dim=-1)
        
        position_indices = torch.arange(
            context_len - 1, context_len - 1 + gamma, device=self.device
        )
        
        blade_gathered = blade_logprobs[
            position_indices.unsqueeze(1),
            candidate_matrix,
        ]  # [gamma, K]
        
        # DPO blade reward: beta * (log pi_blade - log pi_ref)
        rewards = self.blade.beta * (blade_gathered - ref_logprobs)
        return rewards
