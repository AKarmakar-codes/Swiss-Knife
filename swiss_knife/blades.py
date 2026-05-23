"""
Swiss Knife — DPO Blade Reward Computation

Implements the DPO implicit reward used in the match score function:

    r_blade(y | x)  =  β · [ log π_blade(y | x)  -  log π_ref(y | x) ]

where π_blade is the LoRA-adapted model and π_ref is the bare base model.

Both log-probabilities are computed per-token, then summed over the span
to produce a single scalar score per candidate.
"""

import logging
from typing import List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from .config import SwissKnifeConfig

logger = logging.getLogger(__name__)


class DPOBlade:
    """Wraps a DPO-trained LoRA adapter and the reference model to produce
    blade rewards via the implicit DPO reward formulation.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Pipeline configuration (β, device, etc.).
    base_model : PreTrainedModel
        The frozen base model acting as π_ref.
    blade_model : PeftModel
        The LoRA-adapted model acting as π_blade.
    tokenizer : PreTrainedTokenizer
        Shared tokenizer.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        base_model: PreTrainedModel,
        blade_model: PeftModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.cfg = cfg
        self.base_model = base_model
        self.blade_model = blade_model
        self.tokenizer = tokenizer
        self.beta = cfg.beta

    # ── Core computation ───────────────────────────────────────────────

    @torch.no_grad()
    def _logprobs_over_span(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_start: int,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities over the span portion.

        Parameters
        ----------
        model : PreTrainedModel
            Either base (π_ref) or blade (π_blade).
        input_ids : torch.Tensor
            Shape ``[B, seq_len]`` — full sequence (prompt + span).
        attention_mask : torch.Tensor
            Shape ``[B, seq_len]``.
        span_start : int
            Index where the span begins (i.e., prompt length).

        Returns
        -------
        torch.Tensor
            Shape ``[B]`` — sum of log-probs over span tokens for each batch.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits shape: [B, seq_len, vocab_size]
        logits = outputs.logits

        # Shift: predict token t from position t-1
        # We want log p(token_t | tokens_<t) for each span position
        shift_logits = logits[:, span_start - 1:-1, :]   # [B, span_len, V]
        shift_labels = input_ids[:, span_start:]          # [B, span_len]

        log_probs = F.log_softmax(shift_logits, dim=-1)   # [B, span_len, V]

        # Gather the log-prob of the actual token at each position
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [B, span_len]

        # Mask out padding positions in the span
        span_mask = attention_mask[:, span_start:].float()  # [B, span_len]
        token_log_probs = token_log_probs * span_mask

        # Sum over span to get a single score per candidate
        return token_log_probs.sum(dim=-1)  # [B]

    # ── Public API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def score_candidates(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        candidate_ids_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute blade rewards for K candidate spans.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Shape ``[1, prompt_len]`` — the tokenized prompt.
        prompt_mask : torch.Tensor
            Shape ``[1, prompt_len]`` — attention mask for the prompt.
        candidate_ids_list : list of torch.Tensor
            K tensors each of shape ``[span_len]`` — the candidate span tokens.

        Returns
        -------
        torch.Tensor
            Shape ``[K]`` — r_blade for each candidate.
            r_blade = β · (log π_blade - log π_ref)  summed over the span.
        """
        K = len(candidate_ids_list)
        prompt_len = prompt_ids.shape[1]
        device = prompt_ids.device

        # Build batched inputs: [prompt ⊕ candidate_k] for each k
        full_ids_list = []
        full_mask_list = []
        max_len = 0

        for cand_ids in candidate_ids_list:
            cand_ids = cand_ids.to(device)
            full = torch.cat([prompt_ids.squeeze(0), cand_ids], dim=0)
            mask = torch.ones(full.shape[0], dtype=torch.long, device=device)
            full_ids_list.append(full)
            full_mask_list.append(mask)
            max_len = max(max_len, full.shape[0])

        # Pad to uniform length (left-padded since tokenizer is left-pad)
        padded_ids = torch.full(
            (K, max_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        padded_mask = torch.zeros(K, max_len, dtype=torch.long, device=device)

        for i, (ids, mask) in enumerate(zip(full_ids_list, full_mask_list)):
            # Right-align (left-pad)
            offset = max_len - ids.shape[0]
            padded_ids[i, offset:] = ids
            padded_mask[i, offset:] = mask

        # Compute span_start accounting for left padding
        # Each candidate may have different padding, but span_start is relative
        # to the actual content. For simplicity, since all candidates have the
        # same prompt, span_start in the padded tensor is:
        #   max_len - (prompt_len + span_len_k)  +  prompt_len
        # But span lengths may differ due to EOS. We use a uniform span_start
        # = max_len - max_span_len  (conservative).
        # Actually, since all candidates start with the same prompt, the span
        # always starts at position (padding_offset + prompt_len).
        # For the log-prob computation, we use per-row computation.

        # Simpler approach: compute per-candidate to handle variable lengths
        ref_scores = self._logprobs_over_span(
            self.base_model, padded_ids, padded_mask, span_start=max_len - (max_len - prompt_len),
        )
        blade_scores = self._logprobs_over_span(
            self.blade_model, padded_ids, padded_mask, span_start=max_len - (max_len - prompt_len),
        )

        # r_blade = β * (log π_blade - log π_ref)
        rewards = self.beta * (blade_scores - ref_scores)  # [K]
        return rewards

    @torch.no_grad()
    def compute_draft_logprobs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        candidate_ids_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute draft (base model) span-level log-probabilities.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Shape ``[1, prompt_len]``.
        prompt_mask : torch.Tensor
            Shape ``[1, prompt_len]``.
        candidate_ids_list : list of torch.Tensor
            K tensors each of shape ``[span_len]``.

        Returns
        -------
        torch.Tensor
            Shape ``[K]`` — log π_draft(span | prompt)  for each candidate.
        """
        K = len(candidate_ids_list)
        prompt_len = prompt_ids.shape[1]
        device = prompt_ids.device

        full_ids_list = []
        full_mask_list = []
        max_len = 0

        for cand_ids in candidate_ids_list:
            cand_ids = cand_ids.to(device)
            full = torch.cat([prompt_ids.squeeze(0), cand_ids], dim=0)
            mask = torch.ones(full.shape[0], dtype=torch.long, device=device)
            full_ids_list.append(full)
            full_mask_list.append(mask)
            max_len = max(max_len, full.shape[0])

        padded_ids = torch.full(
            (K, max_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        padded_mask = torch.zeros(K, max_len, dtype=torch.long, device=device)

        for i, (ids, mask) in enumerate(zip(full_ids_list, full_mask_list)):
            offset = max_len - ids.shape[0]
            padded_ids[i, offset:] = ids
            padded_mask[i, offset:] = mask

        draft_scores = self._logprobs_over_span(
            self.base_model, padded_ids, padded_mask, span_start=prompt_len,
        )
        return draft_scores  # [K]
