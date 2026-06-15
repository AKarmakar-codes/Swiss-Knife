"""
Swiss Knife — DPO Blade Reward Computation

Implements the DPO implicit reward used in the match score function:

    r_blade(y | x)  =  β · [ log π_blade(y | x)  -  log π_ref(y | x) ]

where π_blade is the LoRA-adapted model and π_ref is the bare base model.

Both log-probabilities are computed per-token, then summed over the span
to produce a single scalar score per candidate.

Option B adds two key methods:
  - score_parallel([gamma, K]):  score all candidate tokens at all gamma
    positions in ONE forward pass each (blade + ref). Returns [gamma, K] tensor.
  - target_logprob_parallel([gamma, K]): extract target log-probs for all
    candidates from a single forward pass. Returns [gamma, K] tensor.
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

    # ── Option B: parallel [gamma, K] scoring ───────────────────────────────

    @torch.no_grad()
    def score_parallel(
        self,
        context_ids: torch.Tensor,
        candidate_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Score all [gamma, K] candidate tokens in ONE forward pass per model.

        This is the key efficiency method for Option B. The draft produces a
        [gamma, K] tensor of candidate token IDs (top-K at each of the gamma
        draft positions). We need the blade reward r_blade(D[i,k] | prefix_i)
        for every (position i, candidate k) pair.

        Strategy:
          1. Run the blade model on the shared greedy prefix
             (context + draft_greedy_path = context + D[:,0]).
          2. At each position i, the model's output logit at step (context_len + i - 1)
             gives the distribution over next tokens conditioned on the prefix up to i.
          3. Index into the K candidates for position i to get K log-probs.
          4. Repeat for the ref (base) model.
          5. blade_reward[i,k] = beta * (log_pi_blade[i,k] - log_pi_ref[i,k]).

        This runs exactly 2 forward passes (blade + ref) regardless of gamma and K.

        Parameters
        ----------
        context_ids : torch.Tensor
            Shape ``[1, context_len]`` — the prompt + generated prefix so far.
        candidate_matrix : torch.Tensor
            Shape ``[gamma, K]`` — candidate token IDs.
            candidate_matrix[i, 0] is the draft's greedy token at position i.
            candidate_matrix[i, k] for k>0 are alternative top-K tokens.

        Returns
        -------
        torch.Tensor
            Shape ``[gamma, K]`` — r_blade(D[i,k] | prefix_i) for all (i,k).
        """
        gamma, K = candidate_matrix.shape
        context_len = context_ids.shape[1]
        device = context_ids.device

        # Build the greedy sequence: [context, D[0,0], D[1,0], ..., D[gamma-1,0]]
        greedy_tokens = candidate_matrix[:, 0]  # [gamma]
        full_ids = torch.cat([
            context_ids.squeeze(0),
            greedy_tokens,
        ], dim=0).unsqueeze(0)  # [1, context_len + gamma]
        full_mask = torch.ones_like(full_ids)

        # Forward pass: blade and ref (base) model in sequence
        blade_logits = self.blade_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits.squeeze(0)  # [context_len + gamma, vocab_size]

        ref_logits = self.base_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits.squeeze(0)  # [context_len + gamma, vocab_size]

        # Extract position-specific log-probs
        # At position i (0-indexed from 0 to gamma-1), the logit to look at is
        # at sequence index (context_len + i - 1) because the model output at
        # position t predicts token t+1.
        blade_logprobs = F.log_softmax(blade_logits.float(), dim=-1)  # [T, V]
        ref_logprobs   = F.log_softmax(ref_logits.float(), dim=-1)    # [T, V]

        # Gather [gamma, K] log-probs
        # position_indices[i] = context_len + i - 1
        position_indices = torch.arange(
            context_len - 1, context_len - 1 + gamma, device=device
        )  # [gamma]

        # candidate_matrix: [gamma, K] — token ids to gather
        blade_gathered = blade_logprobs[
            position_indices.unsqueeze(1),   # [gamma, 1]
            candidate_matrix,                # [gamma, K]
        ]  # [gamma, K]

        ref_gathered = ref_logprobs[
            position_indices.unsqueeze(1),
            candidate_matrix,
        ]  # [gamma, K]

        # DPO blade reward: beta * (log pi_blade - log pi_ref)
        rewards = self.beta * (blade_gathered - ref_gathered)  # [gamma, K]
        return rewards

    @torch.no_grad()
    def target_logprob_parallel(
        self,
        context_ids: torch.Tensor,
        candidate_matrix: torch.Tensor,
        target_model,
    ) -> torch.Tensor:
        """Compute target model log-probabilities for all [gamma, K] candidates.

        In Option B, the target model provides calibrated fluency likelihoods
        (replacing the draft log-prob used in Option A). This method computes:

            log_pi_target[i, k] = log pi_target(D[i,k] | context + D[:i, 0])

        for all positions i=0..gamma-1 and candidates k=0..K-1 in ONE forward pass.

        Parameters
        ----------
        context_ids : torch.Tensor
            Shape ``[1, context_len]``.
        candidate_matrix : torch.Tensor
            Shape ``[gamma, K]``.
        target_model : PreTrainedModel
            The target (verifier) model — in Swiss Knife this is the same as the
            base model (frozen draft = target for the speculative decoding loop).

        Returns
        -------
        torch.Tensor
            Shape ``[gamma, K]`` — log pi_target(D[i,k] | prefix) for all (i,k).
        """
        gamma, K = candidate_matrix.shape
        context_len = context_ids.shape[1]
        device = context_ids.device

        greedy_tokens = candidate_matrix[:, 0]  # [gamma]
        full_ids = torch.cat([
            context_ids.squeeze(0),
            greedy_tokens,
        ], dim=0).unsqueeze(0)  # [1, context_len + gamma]
        full_mask = torch.ones_like(full_ids)

        logits = target_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits.squeeze(0)  # [context_len + gamma, vocab_size]

        log_probs = F.log_softmax(logits.float(), dim=-1)  # [T, V]

        position_indices = torch.arange(
            context_len - 1, context_len - 1 + gamma, device=device
        )  # [gamma]

        gathered = log_probs[
            position_indices.unsqueeze(1),
            candidate_matrix,
        ]  # [gamma, K]

        return gathered

    # ── GSI: Step-level reward scoring ──────────────────────────────────

    @torch.no_grad()
    def score_reasoning_steps(
        self,
        prefix_ids: torch.Tensor,
        step_token_ids_list: list,
    ) -> torch.Tensor:
        """Compute blade rewards for n complete reasoning steps.

        This is the bridge between Swiss Knife blades and GSI:
        it scores entire reasoning steps (not individual tokens), returning
        one scalar reward per step that serves as r(x, y_i) in GSI.

        Parameters
        ----------
        prefix_ids : torch.Tensor
            Shape ``[1, prefix_len]`` — tokenized prompt + previously accepted steps.
        step_token_ids_list : list of torch.Tensor
            n tensors, each of shape ``[step_len_i]`` — token IDs for each
            candidate reasoning step (variable lengths allowed).

        Returns
        -------
        torch.Tensor
            Shape ``[n]`` — r_blade for each step.
            r_blade = β · Σ_t [log π_blade(tok_t | prefix, tok_<t) - log π_ref(tok_t | prefix, tok_<t)]
        """
        n = len(step_token_ids_list)
        if n == 0:
            return torch.tensor([], device=prefix_ids.device)

        prefix_len = prefix_ids.shape[1]
        device = prefix_ids.device

        # Build [prefix ⊕ step_i] for each candidate step
        full_ids_list = []
        full_mask_list = []
        max_len = 0

        for step_ids in step_token_ids_list:
            step_ids = step_ids.to(device)
            full = torch.cat([prefix_ids.squeeze(0), step_ids], dim=0)
            mask = torch.ones(full.shape[0], dtype=torch.long, device=device)
            full_ids_list.append(full)
            full_mask_list.append(mask)
            max_len = max(max_len, full.shape[0])

        # Pad to uniform length (right-pad for simplicity; mask handles it)
        padded_ids = torch.full(
            (n, max_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        padded_mask = torch.zeros(n, max_len, dtype=torch.long, device=device)

        for i, (ids, mask) in enumerate(zip(full_ids_list, full_mask_list)):
            padded_ids[i, :ids.shape[0]] = ids
            padded_mask[i, :mask.shape[0]] = mask

        # Forward pass: blade and ref models
        blade_logits = self.blade_model(
            input_ids=padded_ids, attention_mask=padded_mask
        ).logits  # [n, max_len, V]

        ref_logits = self.base_model(
            input_ids=padded_ids, attention_mask=padded_mask
        ).logits  # [n, max_len, V]

        # Compute per-token log-probs over the step portion only
        blade_logprobs = F.log_softmax(blade_logits.float(), dim=-1)
        ref_logprobs = F.log_softmax(ref_logits.float(), dim=-1)

        rewards = torch.zeros(n, device=device)
        for i, step_ids in enumerate(step_token_ids_list):
            step_len = step_ids.shape[0]
            if step_len == 0:
                continue
            # Positions: predict step tokens from positions [prefix_len-1, prefix_len+step_len-2]
            # Labels: step tokens at positions [prefix_len, prefix_len+step_len-1]
            pred_positions = torch.arange(
                prefix_len - 1, prefix_len - 1 + step_len, device=device
            )
            label_tokens = padded_ids[i, prefix_len:prefix_len + step_len]

            blade_lp = blade_logprobs[i, pred_positions, label_tokens].sum()
            ref_lp = ref_logprobs[i, pred_positions, label_tokens].sum()

            rewards[i] = self.beta * (blade_lp - ref_lp)

        return rewards

    @torch.no_grad()
    def compute_step_draft_logprobs(
        self,
        prefix_ids: torch.Tensor,
        step_token_ids_list: list,
    ) -> torch.Tensor:
        """Compute base model (draft) log-probabilities for n reasoning steps.

        Used as the fluency signal (log π_draft) in the match function.

        Parameters
        ----------
        prefix_ids : torch.Tensor
            Shape ``[1, prefix_len]``.
        step_token_ids_list : list of torch.Tensor
            n tensors, each of shape ``[step_len_i]``.

        Returns
        -------
        torch.Tensor
            Shape ``[n]`` — log π_draft(step_i | prefix) for each step.
        """
        n = len(step_token_ids_list)
        if n == 0:
            return torch.tensor([], device=prefix_ids.device)

        prefix_len = prefix_ids.shape[1]
        device = prefix_ids.device

        full_ids_list = []
        full_mask_list = []
        max_len = 0

        for step_ids in step_token_ids_list:
            step_ids = step_ids.to(device)
            full = torch.cat([prefix_ids.squeeze(0), step_ids], dim=0)
            mask = torch.ones(full.shape[0], dtype=torch.long, device=device)
            full_ids_list.append(full)
            full_mask_list.append(mask)
            max_len = max(max_len, full.shape[0])

        padded_ids = torch.full(
            (n, max_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        padded_mask = torch.zeros(n, max_len, dtype=torch.long, device=device)

        for i, (ids, mask) in enumerate(zip(full_ids_list, full_mask_list)):
            padded_ids[i, :ids.shape[0]] = ids
            padded_mask[i, :mask.shape[0]] = mask

        draft_logits = self.base_model(
            input_ids=padded_ids, attention_mask=padded_mask
        ).logits
        draft_logprobs = F.log_softmax(draft_logits.float(), dim=-1)

        scores = torch.zeros(n, device=device)
        for i, step_ids in enumerate(step_token_ids_list):
            step_len = step_ids.shape[0]
            if step_len == 0:
                continue
            pred_positions = torch.arange(
                prefix_len - 1, prefix_len - 1 + step_len, device=device
            )
            label_tokens = padded_ids[i, prefix_len:prefix_len + step_len]
            scores[i] = draft_logprobs[i, pred_positions, label_tokens].sum()

        return scores

