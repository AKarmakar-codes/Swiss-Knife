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
from contextlib import nullcontext
from typing import List, Optional

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
        blade_name: str = None,
    ):
        self.cfg = cfg
        self.base_model = base_model
        self.blade_model = blade_model
        self.tokenizer = tokenizer
        self.beta = cfg.beta
        
        if blade_name is None and isinstance(blade_model, PeftModel):
            if hasattr(blade_model, "active_adapter") and isinstance(blade_model.active_adapter, str):
                blade_name = blade_model.active_adapter
            elif hasattr(blade_model, "peft_config") and blade_model.peft_config:
                blade_name = next(iter(blade_model.peft_config.keys()))
        self.blade_name = blade_name

    def _disable_adapter_context(self):
        """Context manager to disable PEFT LoRA adapter for reference (π_ref) forward passes."""
        if isinstance(self.base_model, PeftModel):
            return self.base_model.disable_adapter()
        elif isinstance(self.blade_model, PeftModel):
            return self.blade_model.disable_adapter()
        return nullcontext()

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

        # Compute ref scores
        with self._disable_adapter_context():
            ref_scores = self._logprobs_over_span(
                self.base_model, padded_ids, padded_mask, span_start=max_len - (max_len - prompt_len),
            )

        # Compute blade scores
        if isinstance(self.blade_model, PeftModel) and self.blade_name:
            self.blade_model.set_adapter(self.blade_name)
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

        with self._disable_adapter_context():
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
        if isinstance(self.blade_model, PeftModel) and self.blade_name:
            self.blade_model.set_adapter(self.blade_name)
        blade_logits = self.blade_model(
            input_ids=full_ids, attention_mask=full_mask
        ).logits.squeeze(0)  # [context_len + gamma, vocab_size]

        with self._disable_adapter_context():
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

    @staticmethod
    def _step_logprob_sum(
        logits: torch.Tensor,
        padded_ids: torch.Tensor,
        prefix_len: int,
        step_token_ids_list: list,
        device: torch.device,
        return_entropy: bool = False,
    ):
        """Sum log p(tok_t | tok_<t) over each candidate's step span.

        Deliberately avoids ``F.log_softmax(logits, dim=-1)`` over the full
        ``[batch, seq_len, vocab]`` tensor: that materializes a dense
        float32 tensor covering *every* prefix position (which is never
        used — only the last `step_len` positions per row are read) times
        the *full* vocab (~152k for Qwen2.5), for every candidate. For
        n=11 candidates and a few hundred prefix tokens that is multiple
        GB, computed twice (blade + ref) and held live simultaneously —
        that is exactly what threw the CUDA OOM.

        Instead we slice to just the needed (short) `pred_positions` rows
        *before* touching the vocab dimension, and use a numerically
        stable gather + logsumexp per row instead of a dense log-softmax.
        Peak memory here is O(step_len · V) per row, not O(seq_len · V)
        per whole batch — typically a >90% reduction since step_len
        (a handful to a few dozen tokens) is usually << seq_len (prefix
        + step, which grows over the course of generation).

        If return_entropy=True, also computes the per-token lower-bound entropy
        H_t = logsumexp(z_t) - max(z_t) = -log p_max (Rényi Min-Entropy)
        and returns the mean token entropy per candidate step.
        """
        m = logits.shape[0]
        out = torch.zeros(m, device=device)
        entropies = torch.zeros(m, device=device) if return_entropy else None
        for i, step_ids in enumerate(step_token_ids_list):
            step_len = step_ids.shape[0]
            if step_len == 0:
                continue
            pred_positions = torch.arange(
                prefix_len - 1, prefix_len - 1 + step_len, device=device
            )
            label_tokens = padded_ids[i, prefix_len:prefix_len + step_len]

            row_logits = logits[i, pred_positions, :].float()          # [step_len, V]
            row_logsumexp = torch.logsumexp(row_logits, dim=-1)         # [step_len]
            row_target = row_logits.gather(
                -1, label_tokens.unsqueeze(-1)
            ).squeeze(-1)                                                # [step_len]
            out[i] = (row_target - row_logsumexp).sum()

            if return_entropy:
                row_max = row_logits.max(dim=-1).values                 # [step_len]
                row_min_entropy = row_logsumexp - row_max               # [step_len] (-log p_max)
                entropies[i] = row_min_entropy.mean()                   # mean token min-entropy per step

        return (out, entropies) if return_entropy else out

    @torch.no_grad()
    def score_reasoning_steps(
        self,
        prefix_ids: torch.Tensor,
        step_token_ids_list: list,
        micro_batch_size: Optional[int] = None,
        return_entropy: bool = False,
    ):
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
        micro_batch_size : int, optional
            If set, candidates are scored in chunks of this size instead of
            all ``n`` at once, bounding peak activation memory regardless of
            how large ``n`` or the prefix gets. Defaults to scoring all ``n``
            candidates in one batch (previous behaviour).
        return_entropy : bool, optional
            If True, returns (rewards, entropies) where entropies is a 1D tensor [n]
            containing mean per-token min-entropy H_t = logsumexp(z) - max(z).

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Shape ``[n]`` — r_blade for each step.
            If return_entropy=True, returns (rewards, entropies).
        """
        n = len(step_token_ids_list)
        if n == 0:
            empty = torch.tensor([], device=prefix_ids.device)
            return (empty, empty) if return_entropy else empty

        prefix_len = prefix_ids.shape[1]
        device = prefix_ids.device
        mbs = micro_batch_size or n
        rewards = torch.zeros(n, device=device)
        entropies = torch.zeros(n, device=device) if return_entropy else None

        for start in range(0, n, mbs):
            chunk = step_token_ids_list[start:start + mbs]
            m = len(chunk)

            full_ids_list, full_mask_list, max_len = [], [], 0
            for step_ids in chunk:
                step_ids = step_ids.to(device)
                full = torch.cat([prefix_ids.squeeze(0), step_ids], dim=0)
                mask = torch.ones(full.shape[0], dtype=torch.long, device=device)
                full_ids_list.append(full)
                full_mask_list.append(mask)
                max_len = max(max_len, full.shape[0])

            padded_ids = torch.full(
                (m, max_len), self.tokenizer.pad_token_id,
                dtype=torch.long, device=device,
            )
            padded_mask = torch.zeros(m, max_len, dtype=torch.long, device=device)
            for i, (ids, mask) in enumerate(zip(full_ids_list, full_mask_list)):
                padded_ids[i, :ids.shape[0]] = ids
                padded_mask[i, :mask.shape[0]] = mask

            # Score blade first, then free its activations *before* running
            # the ref forward pass — the two [m, max_len, V] logit tensors
            # never need to be alive at the same time.
            blade_logits = self.blade_model(
                input_ids=padded_ids, attention_mask=padded_mask
            ).logits  # [m, max_len, V]
            if return_entropy:
                blade_lp_sum, chunk_entropies = self._step_logprob_sum(
                    blade_logits, padded_ids, prefix_len, chunk, device, return_entropy=True
                )
                entropies[start:start + m] = chunk_entropies
            else:
                blade_lp_sum = self._step_logprob_sum(
                    blade_logits, padded_ids, prefix_len, chunk, device
                )
            del blade_logits

            with self._disable_adapter_context():
                ref_logits = self.base_model(
                    input_ids=padded_ids, attention_mask=padded_mask
                ).logits  # [m, max_len, V]
            ref_lp_sum = self._step_logprob_sum(
                ref_logits, padded_ids, prefix_len, chunk, device
            )
            del ref_logits

            # Normalize by step length so scale matches compute_logprobs_batched
            # (which returns mean per-token log-prob). This ensures mu, draft_logprobs
            # and verifier_logprobs are all on the same O(1) per-token scale.
            for i, step_ids in enumerate(chunk):
                step_len = max(step_ids.shape[0], 1)
                rewards[start + i] = self.beta * (blade_lp_sum[i] - ref_lp_sum[i]) / step_len

        return (rewards, entropies) if return_entropy else rewards

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

        with self._disable_adapter_context():
            draft_logits = self.base_model(
                input_ids=padded_ids, attention_mask=padded_mask
            ).logits
        scores = self._step_logprob_sum(
            draft_logits, padded_ids, prefix_len, step_token_ids_list, device
        )
        del draft_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return scores

