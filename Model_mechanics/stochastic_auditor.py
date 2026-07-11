"""
Swiss Knife — Stochastic Auditor (Phase 4)
===========================================

Defines a family of stochastic scalar functionals of the blade's internal state:
  1. MC Dropout (mc_dropout): fresh dropout masks on the final hidden layer before lm_head.
  2. Random Projection (random_proj): random projection of the final hidden layer.
  3. Attention Head Subsampling (head_subsample): random subsets of attention heads zeroed out.

Draws a new functional independently per match to introduce stochasticity and
intransitivity, justifying the tournament structure.

OPTIMISATION (Phase 4 — Context §5)
--------------------------------------
mc_dropout / random_proj:
  • precompute_hidden_states() runs the blade model ONCE per outer iteration,
    caching the final-layer hidden states h_all [seq_len, D].
  • score_candidates_for_match() then applies the random perturbation directly
    to h_all in-memory and projects via lm_head — ZERO extra forward passes per match.
  • With K=8 knockout (7 matches): 1 precompute + 7×(tensor ops) vs 7 forward passes.

head_subsample:
  • Perturbs o_proj inputs in attention layers — cannot skip forward pass.
  • Continues to register per-match hooks and run 1 forward pass per match.
  • ref_logprobs are precomputed externally (passed in), so each match is 1 blade pass.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Optional

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

    Optimised paths
    ---------------
    mc_dropout / random_proj:
        Call precompute_hidden_states(context_ids, candidate_matrix) ONCE before
        the tournament bracket.  Each subsequent call to score_candidates_for_match
        applies the perturbation in pure tensor ops — no model forward pass.

    head_subsample:
        Registers o_proj pre-hooks per match and runs 1 full blade forward pass
        per match (unavoidable because the perturbation is in attention layers).
    """

    def __init__(self, blade: DPOBlade, cfg: SwissKnifeConfig, auditor_cfg: Optional[StochasticAuditorConfig] = None):
        self.blade = blade
        self.cfg = cfg
        self.auditor_cfg = auditor_cfg or StochasticAuditorConfig()

        if self.auditor_cfg.harmlessness_only:
            pass  # Harmlessness blade is active by default in the config.

        self.model = blade.blade_model
        self.device = next(self.model.parameters()).device
        self.dtype  = next(self.model.parameters()).dtype  # bfloat16 / float16 / float32

        # For head_subsample: locate o_proj layers.
        self.o_projs = self._get_last_layers_o_proj(self.model, self.auditor_cfg.num_layers_to_mask)
        self.hooks: List = []

        # Precomputed hidden states cache (mc_dropout / random_proj fast path).
        self._h_all: Optional[torch.Tensor] = None           # [seq_len, D]
        self._candidate_matrix_cache: Optional[torch.Tensor] = None  # [gamma, K]
        self._context_len_cache: int = 0

        # Cache lm_head for fast in-memory projection.
        self._lm_head: Optional[nn.Module] = self._find_lm_head()
        if self._lm_head is None:
            logger.warning(
                "StochasticAuditor: could not locate lm_head. "
                "mc_dropout / random_proj fast path will fall back to full forward passes."
            )

        # Internal state
        self.forward_passes = 0

    # ── Model introspection helpers ──────────────────────────────────────────

    def _find_lm_head(self) -> Optional[nn.Module]:
        """Locate the lm_head Linear module regardless of PEFT wrapping."""
        for candidate in (self.model,
                          getattr(self.model, "base_model", None),
                          getattr(getattr(self.model, "base_model", None), "model", None)):
            if candidate is not None and hasattr(candidate, "lm_head"):
                return candidate.lm_head
        return None

    def _get_last_layers_o_proj(self, model: nn.Module, num_layers: int) -> List[nn.Module]:
        """Locate the o_proj modules of the last N transformer layers.

        Handles two common module hierarchies:
          - PeftModel:  model.base_model.model.layers[...].self_attn.o_proj
          - Bare model: model.model.layers[...].self_attn.o_proj
        """
        o_projs = []

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

    # ── Precomputation (mc_dropout / random_proj fast path) ──────────────────

    @torch.no_grad()
    def precompute_hidden_states(
        self,
        context_ids: torch.Tensor,
        candidate_matrix: torch.Tensor,
    ) -> None:
        """Run the blade model ONCE to cache final-layer hidden states.

        Must be called once per outer iteration BEFORE the tournament bracket,
        for mc_dropout and random_proj modes.  head_subsample is a no-op here.

        Parameters
        ----------
        context_ids : torch.Tensor
            Shape ``[1, context_len]``.
        candidate_matrix : torch.Tensor
            Shape ``[gamma, K]``.

        Sets
        ----
        self._h_all : ``[seq_len, D]`` — clean hidden states (no hooks active).
        """
        if self.auditor_cfg.mode not in ("mc_dropout", "random_proj"):
            return  # head_subsample cannot exploit this precomputation.

        if self._lm_head is None:
            logger.warning("precompute_hidden_states: lm_head not found, skipping.")
            return

        gamma, K = candidate_matrix.shape
        greedy_tokens = candidate_matrix[:, 0]  # [gamma]
        full_ids = torch.cat(
            [context_ids.squeeze(0), greedy_tokens], dim=0
        ).unsqueeze(0)  # [1, ctx_len + gamma]
        full_mask = torch.ones_like(full_ids)

        # Try the standard HuggingFace output_hidden_states path first.
        # Fall back to a forward pre-hook on lm_head if the model doesn't support it.
        captured: List[Optional[torch.Tensor]] = [None]

        def _capture_hook(module, args):
            captured[0] = args[0].detach()  # [B, seq_len, D]
            return args   # pass through unchanged

        hook_handle = None
        if self._lm_head is not None:
            hook_handle = self._lm_head.register_forward_pre_hook(_capture_hook)

        try:
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
            )
            if outputs.hidden_states is not None:
                h = outputs.hidden_states[-1]   # [B, seq_len, D]
            elif captured[0] is not None:
                h = captured[0]                 # captured by hook
            else:
                logger.warning(
                    "precompute_hidden_states: neither hidden_states nor hook captured "
                    "the final hidden state. Fast path disabled for this iteration."
                )
                return
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        self._h_all = h.squeeze(0).detach()          # [seq_len, D]
        self._candidate_matrix_cache = candidate_matrix
        self._context_len_cache = context_ids.shape[1]
        self.forward_passes += 1

        logger.debug(
            "precompute_hidden_states: cached h_all %s, ctx_len=%d, gamma=%d, K=%d",
            tuple(self._h_all.shape), self._context_len_cache, gamma, K,
        )

    def clear_precomputed(self) -> None:
        """Discard cached hidden states (call after the outer iteration is done)."""
        self._h_all = None
        self._candidate_matrix_cache = None
        self._context_len_cache = 0

    # ── Fast in-memory scoring (mc_dropout / random_proj) ───────────────────

    def _fast_score_in_memory(self, ref_logprobs: torch.Tensor) -> torch.Tensor:
        """Apply a fresh random perturbation to cached h_all and project via lm_head.

        Zero model forward passes.  Returns [gamma, K] DPO blade rewards.
        """
        assert self._h_all is not None, \
            "Call precompute_hidden_states() before _fast_score_in_memory()."
        assert self._lm_head is not None, \
            "lm_head not found — cannot use fast in-memory scoring."

        h = self._h_all.clone()  # [seq_len, D] — clone to avoid aliasing

        mode = self.auditor_cfg.mode
        if mode == "mc_dropout":
            # Fresh dropout mask applied to the full hidden-state sequence.
            h = F.dropout(h, p=self.auditor_cfg.dropout_p, training=True)

        elif mode == "random_proj":
            # Fresh spectrally-normalised random projection matrix R.
            D = h.shape[-1]
            R = torch.randn(D, D, device=self.device, dtype=self.dtype)
            R = R / (torch.norm(R, p=2) + 1e-6)
            h = h + self.auditor_cfg.proj_epsilon * (h @ R)

        # Project [seq_len, D] → [seq_len, vocab_size] using lm_head weights.
        # Unsqueeze/squeeze the batch dim so nn.Linear is happy.
        logits = self._lm_head(h.unsqueeze(0)).squeeze(0)  # [seq_len, vocab_size]

        blade_logprobs = F.log_softmax(logits.float(), dim=-1)

        gamma, K = self._candidate_matrix_cache.shape
        ctx_len   = self._context_len_cache
        position_indices = torch.arange(
            ctx_len - 1, ctx_len - 1 + gamma, device=self.device
        )

        blade_gathered = blade_logprobs[
            position_indices.unsqueeze(1),
            self._candidate_matrix_cache,
        ]  # [gamma, K]

        rewards = self.blade.beta * (blade_gathered - ref_logprobs)
        return rewards  # [gamma, K]

    # ── Hook management (head_subsample) ────────────────────────────────────

    def _register_hooks(self) -> None:
        """Register PyTorch hooks for the active match functional (head_subsample only)."""
        self._unregister_hooks()

        mode = self.auditor_cfg.mode

        if mode == "head_subsample" and self.o_projs:
            num_heads = 28  # Qwen2.5-7B default
            _cfg = getattr(self.model, "config", None)
            if _cfg is None and hasattr(self.model, "base_model"):
                _cfg = getattr(self.model.base_model, "config", None)
            if _cfg is not None:
                num_heads = getattr(_cfg, "num_attention_heads", num_heads)

            first_o_proj = self.o_projs[0]
            hidden_size  = first_o_proj.in_features
            head_dim     = max(1, hidden_size // num_heads)

            num_keep     = max(1, int(num_heads * self.auditor_cfg.head_frac))
            keep_indices = random.sample(range(num_heads), num_keep)

            head_mask = torch.zeros(num_heads, device=self.device, dtype=self.dtype)
            head_mask[keep_indices] = 1.0
            expanded_mask = head_mask.unsqueeze(1).repeat(1, head_dim).view(-1)

            def make_hook(mask):
                def hook(module, args):
                    x = args[0]  # [B, seq_len, hidden_size]
                    return (x * mask,)
                return hook

            for o_proj in self.o_projs:
                handle = o_proj.register_forward_pre_hook(make_hook(expanded_mask))
                self.hooks.append(handle)

        # mc_dropout / random_proj: no hooks needed — fast path handles them.

    def _unregister_hooks(self) -> None:
        """Remove all active PyTorch hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    # ── Public match-level API ───────────────────────────────────────────────

    def draw_fresh_functional(self) -> None:
        """Draw a new functional for the upcoming match.

        mc_dropout / random_proj:  no-op (randomness is injected inside
            _fast_score_in_memory at score time — every call is fresh).
        head_subsample:  registers a new random head mask via forward pre-hooks.
        """
        if self.auditor_cfg.mode == "head_subsample":
            self._register_hooks()
        # else: freshness is implicit in the per-call random ops in _fast_score_in_memory.

    def clear_functional(self) -> None:
        """Clean up the functional after a match."""
        self._unregister_hooks()

    # ── Scoring ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_candidates_for_match(
        self,
        context_ids: torch.Tensor,
        candidate_matrix: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stochastic blade rewards for all K candidates.

        Fast path (mc_dropout / random_proj):
            Requires precompute_hidden_states() to have been called first.
            Applies a fresh random perturbation to the cached hidden states
            and projects via lm_head.  ZERO model forward passes.

        Standard path (head_subsample, or fast-path fallback):
            Runs 1 blade forward pass with the active o_proj hooks.

        Parameters
        ----------
        context_ids : torch.Tensor
            Shape ``[1, context_len]``.
        candidate_matrix : torch.Tensor
            Shape ``[gamma, K]``.
        ref_logprobs : torch.Tensor
            Shape ``[gamma, K]`` — precomputed reference model log-probs.

        Returns
        -------
        torch.Tensor
            Shape ``[gamma, K]`` — stochastic r_blade for each (pos, cand).
        """
        mode = self.auditor_cfg.mode

        # ── Fast path: mc_dropout / random_proj ──────────────────────────
        if mode in ("mc_dropout", "random_proj") and self._h_all is not None:
            return self._fast_score_in_memory(ref_logprobs)

        # ── Standard path: head_subsample (1 forward pass with hooks) ────
        # Also used as a fallback if precompute_hidden_states() was not called.
        self.forward_passes += 1

        gamma, K   = candidate_matrix.shape
        context_len = context_ids.shape[1]

        greedy_tokens = candidate_matrix[:, 0]
        full_ids  = torch.cat([context_ids.squeeze(0), greedy_tokens], dim=0).unsqueeze(0)
        full_mask = torch.ones_like(full_ids)

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

        rewards = self.blade.beta * (blade_gathered - ref_logprobs)
        return rewards
