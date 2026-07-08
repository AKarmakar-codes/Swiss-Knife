"""
Swiss Knife — Span-Level Generation Loop

Implements Algorithm 1 from the proposal (Section 5):

    y ← []
    while not done:
        C  ← draft.sample(x ⊕ y, num=K, len=L)       # K independent spans
        s_d ← draft.logprob(C | x ⊕ y)                # batched
        s_b ← blade.logratio(C | x ⊕ y)               # batched; β·log(π_b/π_ref)
        w  ← KnockoutBracket(C, s_d, s_b, α)
        y  ← y ⊕ C[w]

The loop terminates when:
    - max_new_tokens is reached, or
    - every candidate in a round contains EOS (nothing useful to append).
"""

import json
import logging
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from .config import SwissKnifeConfig
from .blades import DPOBlade
from .tournament import knockout_bracket
from .swiss_system import swiss_system_bracket
from .elo_system import elo_bracket

logger = logging.getLogger(__name__)


class SwissKnifeGenerator:
    """Option A span-level tournament generator.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline configuration.
    tokenizer : PreTrainedTokenizer
        Shared tokenizer.
    base_model : PreTrainedModel
        Frozen draft / reference model.
    blade_model : PeftModel
        Active DPO blade adapter.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        base_model: PreTrainedModel,
        blade_model: PeftModel,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.blade = DPOBlade(cfg, base_model, blade_model, tokenizer)

        if cfg.tournament_mode == "knockout":
            self._run_tournament = lambda ts, bs: knockout_bracket(ts, bs, self.cfg.alpha)
        elif cfg.tournament_mode == "swiss":
            self._run_tournament = lambda ts, bs: swiss_system_bracket(ts, bs, self.cfg.alpha, rounds=self.cfg.swiss_rounds)
        else:
            self._run_tournament = lambda ts, bs: elo_bracket(ts, bs, self.cfg.alpha, normalize=self.cfg.normalize_scores, temperature=self.cfg.elo_temperature, rounds=self.cfg.elo_rounds)

    # ── Candidate sampling ─────────────────────────────────────────────

    @torch.no_grad()
    def _sample_candidates(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Sample K independent spans of length L from the draft model.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Shape ``[1, prompt_len]``.
        attention_mask : torch.Tensor
            Shape ``[1, prompt_len]``.

        Returns
        -------
        list of torch.Tensor
            K tensors, each of shape ``[≤ L]`` (may be shorter if EOS hit).
        """
        K = self.cfg.K
        L = self.cfg.L
        device = prompt_ids.device

        # Expand prompt for batched generation:  [1, P] → [K, P]
        batch_ids = prompt_ids.expand(K, -1).contiguous()
        batch_mask = attention_mask.expand(K, -1).contiguous()

        # Generate K spans in parallel
        outputs = self.base_model.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=L,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # outputs shape: [K, prompt_len + generated_len]
        prompt_len = prompt_ids.shape[1]

        candidates = []
        for k in range(K):
            span = outputs[k, prompt_len:]  # just the new tokens
            # Truncate at first EOS if present
            eos_positions = (span == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                span = span[:eos_positions[0]]  # exclude EOS itself
            candidates.append(span)

        return candidates

    # ── Main generation loop ───────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Run the Swiss Knife Option A generation loop.

        Parameters
        ----------
        prompt : str
            The input prompt text.
        max_new_tokens : int, optional
            Override ``cfg.max_new_tokens``.
        verbose : bool
            If True, log per-round tournament details.

        Returns
        -------
        str
            The generated text (prompt + aligned completion).
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens

        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        device = next(self.base_model.parameters()).device
        prompt_ids = encoded["input_ids"].to(device)        # [1, P]
        prompt_mask = encoded["attention_mask"].to(device)  # [1, P]

        # ── Algorithm 1: y ← [] ────────────────────────────────────────
        generated_ids: List[int] = []
        total_generated = 0
        round_count = 0

        while total_generated < max_tokens:
            round_count += 1

            # Current context = prompt ⊕ generated-so-far
            if generated_ids:
                gen_tensor = torch.tensor(
                    generated_ids, dtype=torch.long, device=device,
                ).unsqueeze(0)
                current_ids = torch.cat([prompt_ids, gen_tensor], dim=1)
                current_mask = torch.ones_like(current_ids)
            else:
                current_ids = prompt_ids
                current_mask = prompt_mask

            # ── Step 1: C ← draft.sample(x ⊕ y, num=K, len=L) ────────
            candidates = self._sample_candidates(current_ids, current_mask)

            # Check termination: if all candidates are empty, we're done
            if all(len(c) == 0 for c in candidates):
                logger.info("All candidates empty (EOS). Stopping.")
                break

            # Filter out empty candidates (replace with shortest non-empty)
            non_empty = [c for c in candidates if len(c) > 0]
            if len(non_empty) < len(candidates):
                # Pad empty slots by repeating a non-empty candidate
                fallback = non_empty[0]
                candidates = [c if len(c) > 0 else fallback for c in candidates]

            # ── Step 2: s_d ← draft.logprob(C | x ⊕ y) ──────────────
            draft_scores = self.blade.compute_draft_logprobs(
                current_ids, current_mask, candidates,
            )

            # ── Step 3: s_b ← blade.logratio(C | x ⊕ y) ────────────
            blade_scores = self.blade.score_candidates(
                current_ids, current_mask, candidates,
            )

            # Snapshot raw (pre-bias, pre-normalisation) scores for the
            # scores-log dump. Used by make_plots.py to visualise the
            # raw scale mismatch between draft and blade.
            draft_raw = draft_scores.detach().float().cpu().tolist()
            blade_raw = blade_scores.detach().float().cpu().tolist()

            # Diagnostic: log the per-round scale of both score tensors.
            # If draft std ≫ blade std without normalisation, the bracket
            # is effectively running on draft-argmax — a bug, not a feature.
            logger.debug(
                "scales | draft μ=%.3f σ=%.3f range=[%.3f,%.3f] | "
                "blade μ=%.5f σ=%.5f range=[%.5f,%.5f]",
                draft_scores.mean().item(), draft_scores.std().item(),
                draft_scores.min().item(), draft_scores.max().item(),
                blade_scores.mean().item(), blade_scores.std().item(),
                blade_scores.min().item(), blade_scores.max().item(),
            )

            # Calibration-invariance probe: add a constant offset to all
            # blade scores. Pairwise differences cancel it, so the winner
            # should be unchanged. Use --blade-bias from CLI to verify.
            if self.cfg.blade_bias != 0.0:
                blade_scores = blade_scores + self.cfg.blade_bias

            # Per-round z-score normalisation. Brings draft and blade
            # tensors onto a comparable scale so α actually controls the
            # mix instead of being dominated by the larger-magnitude
            # tensor. Z-scoring is itself shift-invariant, so the
            # calibration-invariance property of pairwise selection is
            # preserved at every α. Disable with --no-normalize to test
            # the kernel-level invariance on raw scores.
            if self.cfg.normalize_scores:
                draft_scores = (draft_scores - draft_scores.mean()) / (draft_scores.std() + 1e-6)
                blade_scores = (blade_scores - blade_scores.mean()) / (blade_scores.std() + 1e-6)

            # ── Step 4: w ← Tournament(C, s_d, s_b, α) ─────────
            winner_idx = self._run_tournament(
                draft_scores, blade_scores
            )

            # Optional JSONL dump of per-round score vectors for plotting.
            if self.cfg.scores_log:
                with open(self.cfg.scores_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "round":      round_count,
                        "alpha":      self.cfg.alpha,
                        "normalized": bool(self.cfg.normalize_scores),
                        "draft_raw":  draft_raw,
                        "blade_raw":  blade_raw,
                        "draft_used": draft_scores.detach().float().cpu().tolist(),
                        "blade_used": blade_scores.detach().float().cpu().tolist(),
                        "winner":     int(winner_idx),
                    }) + "\n")

            # ── Step 5: y ← y ⊕ C[w] ────────────────────────────────
            winning_span = candidates[winner_idx]
            generated_ids.extend(winning_span.tolist())
            total_generated += len(winning_span)

            if verbose:
                span_text = self.tokenizer.decode(winning_span, skip_special_tokens=True)
                logger.info(
                    "Round %d | Winner: c%d | Span (%d tokens): '%s' | "
                    "draft=%.3f blade=%.3f | Total: %d/%d",
                    round_count, winner_idx, len(winning_span),
                    span_text,
                    draft_scores[winner_idx].item(),
                    blade_scores[winner_idx].item(),
                    total_generated, max_tokens,
                )

            # Check for EOS in winning span
            if self.tokenizer.eos_token_id in winning_span.tolist():
                logger.info("EOS in winning span. Stopping.")
                break

        # Decode final output
        all_ids = prompt_ids.squeeze(0).tolist() + generated_ids
        output = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        return output
