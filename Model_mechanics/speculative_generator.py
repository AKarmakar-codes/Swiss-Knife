"""
Swiss Knife — Option B: Speculative-Decoding-Integrated Tournament (Phase 2)
=============================================================================

Implements Algorithm 2 from swiss_knife_analysis.pdf (Section 6):

    y ← []
    while not done:
        D  ← draft.propose(x ⊕ y, lookahead=γ, topK=K)    # shape [γ, K]
        ℓt ← target.logprob_parallel(D | x ⊕ y)            # ONE forward pass, shape [γ, K]
        rb ← blade.score_parallel(D | x ⊕ y)               # shape [γ, K]
        for i = 1 to γ:
            wi ← Tournament(D[i,:], ℓt[i,:], rb[i,:], α)
            if D[i, wi] ≠ D[i, 0]:                          # winner differs from draft top-1
                y ← y ⊕ D[1:i-1, 0] ⊕ D[i, wi]
                break                                        # discard tail, restart draft
        if no rejection:
            y ← y ⊕ D[:, 0]                                # accept all γ tokens
    return y

What makes this "actually speculative" (§6.3):
  1. ONE target forward pass per γ positions → throughput gain.
  2. Positional acceptance propagation: first rejection discards tail.
  3. Calibrated likelihoods: target model provides π_target (not a proxy).
  4. Alignment socket: blade plugs into the verifier slot; hot-swap = pointer swap.

Compute profile per generated span of expected length ℓ̄:
  • Draft forward passes:  γ   (autoregressive proposal)
  • Target forward passes: 1   (parallel verification)
  • Blade forward passes:  1   (parallel scoring over [γ, K])
  • Tournament matches:    γ·(K-1) if knockout, γ·(K/2)·R if Swiss
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from .config import SwissKnifeConfig
from .blades import DPOBlade
from .tournament import knockout_bracket, stochastic_knockout_bracket
from .swiss_system import swiss_system_bracket, stochastic_swiss_bracket

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generation statistics (for evaluation harnesses)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeculativeStats:
    """Collected statistics from one Option B generation run."""

    total_rounds: int = 0
    """Total speculative draft-verify cycles."""

    total_tokens_accepted: int = 0
    """Tokens committed to y (excluding EOS)."""

    full_accept_rounds: int = 0
    """Rounds where all γ draft tokens were accepted (no rejection)."""

    partial_accept_rounds: int = 0
    """Rounds where rejection occurred mid-sequence."""

    acceptance_positions: List[int] = field(default_factory=list)
    """List of accept-prefix lengths (how many greedy tokens were accepted
    before the first rejection in each partial round)."""

    target_forward_passes: int = 0
    """Total target model forward passes."""

    blade_forward_passes: int = 0
    """Total blade model forward passes."""

    tournament_calls: int = 0
    """Total per-position tournament() calls."""

    total_time_s: float = 0.0
    """Wall-clock generation time in seconds."""

    @property
    def acceptance_rate(self) -> float:
        """Fraction of speculative rounds that accepted all γ tokens."""
        if self.total_rounds == 0:
            return 0.0
        return self.full_accept_rounds / self.total_rounds

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s < 1e-6:
            return 0.0
        return self.total_tokens_accepted / self.total_time_s

    @property
    def auditor_calls_per_token(self) -> float:
        if self.total_tokens_accepted == 0:
            return 0.0
        return self.blade_forward_passes / self.total_tokens_accepted

    def to_dict(self) -> dict:
        return {
            "total_rounds": self.total_rounds,
            "total_tokens_accepted": self.total_tokens_accepted,
            "full_accept_rounds": self.full_accept_rounds,
            "partial_accept_rounds": self.partial_accept_rounds,
            "acceptance_rate": self.acceptance_rate,
            "target_forward_passes": self.target_forward_passes,
            "blade_forward_passes": self.blade_forward_passes,
            "tournament_calls": self.tournament_calls,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "auditor_calls_per_token": round(self.auditor_calls_per_token, 4),
            "total_time_s": round(self.total_time_s, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Option B generator
# ─────────────────────────────────────────────────────────────────────────────

class SwissKnifeSpeculativeGenerator:
    """Option B: Speculative-Decoding-Integrated Tournament Verifier.

    The draft model proposes γ future tokens, keeping the top-K alternatives
    at each position → candidate matrix D of shape [γ, K]. The target model
    (same as base model in our setup, acting as the "calibrated verifier") and
    the blade each do ONE forward pass over D. Per-position tournaments select
    the best token. Positional acceptance propagates: first divergence from
    draft top-1 triggers a tail discard and draft restart.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline config (gamma, K, alpha, beta, tournament_mode, etc.)
    tokenizer : PreTrainedTokenizer
        Shared tokenizer.
    base_model : PreTrainedModel
        Frozen base model — serves as BOTH the draft model and the target
        verifier (π_ref for DPO, π_target for speculative accept/reject).
    blade_model : PeftModel
        Active DPO LoRA adapter (π_blade).
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
        self.blade_model = blade_model
        self.blade = DPOBlade(cfg, base_model, blade_model, tokenizer)

        if cfg.use_stochastic_auditor:
            from .stochastic_auditor import StochasticAuditor, StochasticAuditorConfig
            auditor_cfg = StochasticAuditorConfig(
                mode=cfg.stochastic_mode,
                dropout_p=cfg.stochastic_dropout_p,
                proj_epsilon=cfg.stochastic_proj_epsilon,
                head_frac=cfg.stochastic_head_frac,
                num_layers_to_mask=cfg.stochastic_num_layers_to_mask,
            )
            self.auditor = StochasticAuditor(self.blade, cfg, auditor_cfg)
        else:
            self.auditor = None

        # Select tournament function based on config
        self._run_tournament = (
            self._knockout_at_position
            if cfg.tournament_mode == "knockout"
            else self._swiss_at_position
        )

        logger.info(
            "SwissKnifeSpeculativeGenerator initialized: "
            "γ=%d, K=%d, α=%.2f, β=%.3f, tournament=%s, stochastic=%s",
            cfg.gamma, cfg.K, cfg.alpha, cfg.beta, cfg.tournament_mode,
            cfg.use_stochastic_auditor,
        )


    # ── Draft proposal: [γ, K] candidate tensor ──────────────────────────

    @torch.no_grad()
    def _draft_propose(
        self,
        context_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the draft model autoregressively for γ steps, capturing top-K
        token IDs at each step.

        This implements:
            D ← draft.propose(x ⊕ y, lookahead=γ, topK=K)  # shape [γ, K]

        The draft runs greedily step-by-step. At each step t:
          1. Forward pass to get logit distribution over vocab.
          2. Top-K tokens by logit value → D[t, :].
          3. Greedy token D[t, 0] is appended to the draft context.

        Parameters
        ----------
        context_ids : torch.Tensor
            Shape ``[1, context_len]`` — prompt + y generated so far.

        Returns
        -------
        candidate_matrix : torch.Tensor
            Shape ``[γ, K]`` — top-K token IDs per position.
            Column 0 is always the greedy (argmax) token.
        draft_topk_logits : torch.Tensor
            Shape ``[γ, K]`` — raw logits for the K candidates per position.
            Retained for diagnostic/logging purposes.
        """
        gamma = self.cfg.gamma
        K = self.cfg.K
        device = context_ids.device

        candidate_ids = torch.zeros(gamma, K, dtype=torch.long, device=device)
        draft_topk_logits = torch.zeros(gamma, K, device=device)

        # Running context for autoregressive draft
        running_ids = context_ids.clone()  # [1, T]

        for t in range(gamma):
            outputs = self.base_model(
                input_ids=running_ids,
                attention_mask=torch.ones_like(running_ids),
            )
            logits_t = outputs.logits[0, -1, :]  # [vocab_size]

            # Top-K by logit value (not probability — avoids redundant softmax)
            topk_logits, topk_ids = torch.topk(logits_t, K)
            candidate_ids[t] = topk_ids
            draft_topk_logits[t] = topk_logits

            # Append greedy token to running context
            greedy_token = topk_ids[0:1].unsqueeze(0)  # [1, 1]
            running_ids = torch.cat([running_ids, greedy_token], dim=1)

        return candidate_ids, draft_topk_logits  # [γ, K], [γ, K]

    # ── Per-position tournament helpers ─────────────────────────────────

    def _knockout_at_position(
        self,
        target_scores_i: torch.Tensor,
        blade_scores_i: torch.Tensor,
    ) -> int:
        """Run knockout bracket at a single position."""
        return knockout_bracket(target_scores_i, blade_scores_i, self.cfg.alpha)

    def _swiss_at_position(
        self,
        target_scores_i: torch.Tensor,
        blade_scores_i: torch.Tensor,
    ) -> int:
        """Run Swiss-system bracket at a single position."""
        return swiss_system_bracket(
            target_scores_i,
            blade_scores_i,
            self.cfg.alpha,
            rounds=self.cfg.swiss_rounds,
        )

    # ── Normalize scores for a single position ───────────────────────────

    @staticmethod
    def _znorm(t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / (t.std() + 1e-6)

    # ── Main Option B generation loop ────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        """Run Algorithm 2: speculative-decoding-integrated tournament.

        Parameters
        ----------
        prompt : str
            Input prompt text.
        max_new_tokens : int, optional
            Override cfg.max_new_tokens.
        verbose : bool
            Log per-round details.
        return_stats : bool
            If True, return (text, SpeculativeStats) tuple instead of just text.

        Returns
        -------
        str | (str, SpeculativeStats)
            Generated text, optionally paired with statistics.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        gamma = self.cfg.gamma
        K = self.cfg.K

        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        device = next(self.base_model.parameters()).device
        prompt_ids = encoded["input_ids"].to(device)   # [1, P]

        # ── Algorithm 2: y ← [] ─────────────────────────────────────────
        generated_ids: List[int] = []
        stats = SpeculativeStats()
        t_start = time.perf_counter()

        while len(generated_ids) < max_tokens:
            stats.total_rounds += 1

            # Current context = prompt ⊕ y
            if generated_ids:
                gen_tensor = torch.tensor(
                    generated_ids, dtype=torch.long, device=device,
                ).unsqueeze(0)
                context_ids = torch.cat([prompt_ids, gen_tensor], dim=1)
            else:
                context_ids = prompt_ids  # [1, P]

            # ── Step 1: D ← draft.propose(...) → [γ, K] ────────────────
            candidate_matrix, draft_topk_logits = self._draft_propose(context_ids)
            # candidate_matrix[i, 0] = greedy token at position i
            # candidate_matrix[i, k>0] = top-K alternatives

            # ── Step 2: ℓt ← target.logprob_parallel(D | context) [γ,K] ─
            target_logprobs = self.blade.target_logprob_parallel(
                context_ids, candidate_matrix, self.base_model,
            )  # [γ, K]
            stats.target_forward_passes += 1

            # ── Step 3: rb ← blade.score_parallel(D | context) [γ, K] ───
            if not self.cfg.use_stochastic_auditor:
                blade_scores = self.blade.score_parallel(
                    context_ids, candidate_matrix,
                )  # [γ, K]
                # Apply blade bias if set (calibration invariance probe)
                if self.cfg.blade_bias != 0.0:
                    blade_scores = blade_scores + self.cfg.blade_bias
                stats.blade_forward_passes += 1
            else:
                blade_scores = None
                self.auditor.forward_passes = 0

            # ── Steps 6-12: per-position tournament + acceptance ─────────
            accepted_prefix: List[int] = []
            rejection_occurred = False
            rejection_position = -1

            for i in range(gamma):
                # Remaining budget check
                if len(generated_ids) + len(accepted_prefix) >= max_tokens:
                    break

                ts_i = target_logprobs[i]  # [K]

                if self.cfg.use_stochastic_auditor:
                    if self.cfg.tournament_mode == "knockout":
                        wi = stochastic_knockout_bracket(
                            draft_scores=ts_i,
                            auditor=self.auditor,
                            context_ids=context_ids,
                            candidate_matrix=candidate_matrix,
                            ref_logprobs=target_logprobs,
                            position_idx=i,
                            alpha=self.cfg.alpha,
                            normalize=self.cfg.normalize_scores,
                        )
                    else:
                        wi = stochastic_swiss_bracket(
                            target_scores=ts_i,
                            auditor=self.auditor,
                            context_ids=context_ids,
                            candidate_matrix=candidate_matrix,
                            ref_logprobs=target_logprobs,
                            position_idx=i,
                            alpha=self.cfg.alpha,
                            rounds=self.cfg.swiss_rounds,
                            normalize=self.cfg.normalize_scores,
                        )
                else:
                    bs_i = blade_scores[i]     # [K]

                    # Z-score normalize if configured
                    if self.cfg.normalize_scores:
                        ts_i = self._znorm(ts_i)
                        bs_i = self._znorm(bs_i)

                    # wi ← Tournament(D[i,:], ℓt[i,:], rb[i,:], α)
                    wi = self._run_tournament(ts_i, bs_i)

                stats.tournament_calls += 1


                logger.debug(
                    "Position %d: tournament_winner=c%d (greedy=c0, token_id=%d→%d)",
                    i, wi,
                    candidate_matrix[i, 0].item(),
                    candidate_matrix[i, wi].item(),
                )

                # Deterministic: tournament winner is final (no probabilistic gate)
                # if D[i, wi] ≠ D[i, 0]:  rejection event
                if wi != 0:
                    winner_token = int(candidate_matrix[i, wi].item())
                    # Accept D[0:i, 0] (greedy prefix) + winner token
                    greedy_prefix = candidate_matrix[:i, 0].tolist()
                    accepted_prefix.extend(greedy_prefix)
                    accepted_prefix.append(winner_token)
                    rejection_occurred = True
                    rejection_position = i
                    stats.partial_accept_rounds += 1
                    stats.acceptance_positions.append(i)
                    if verbose:
                        greedy_text = self.tokenizer.decode(
                            candidate_matrix[:i, 0].tolist(), skip_special_tokens=True
                        )
                        winner_text = self.tokenizer.decode(
                            [winner_token], skip_special_tokens=True
                        )
                        logger.info(
                            "Round %d | Rejection at pos %d | "
                            "Greedy prefix (%d tok): '%s' | Winner: '%s'",
                            stats.total_rounds, i, i, greedy_text, winner_text,
                        )
                    break  # discard tail, restart draft

            if not rejection_occurred:
                # if no rejection: accept all γ tokens (greedy path D[:,0])
                all_greedy = candidate_matrix[:, 0].tolist()
                accepted_prefix.extend(all_greedy)
                stats.full_accept_rounds += 1
                if verbose:
                    span_text = self.tokenizer.decode(
                        all_greedy, skip_special_tokens=True
                    )
                    logger.info(
                        "Round %d | Full accept (%d tok): '%s'",
                        stats.total_rounds, gamma, span_text,
                    )

            if self.cfg.use_stochastic_auditor:
                stats.blade_forward_passes += self.auditor.forward_passes

            # ── Commit accepted tokens ───────────────────────────────────
            # Filter to budget
            remaining = max_tokens - len(generated_ids)
            accepted_prefix = accepted_prefix[:remaining]

            # Check for EOS in accepted tokens
            eos_id = self.tokenizer.eos_token_id
            eos_hit = False
            clean_tokens = []
            for tok in accepted_prefix:
                if tok == eos_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_ids.extend(clean_tokens)
            stats.total_tokens_accepted += len(clean_tokens)

            if eos_hit:
                logger.info("EOS encountered. Stopping.")
                break

            if len(generated_ids) >= max_tokens:
                break

        # ── Finalize ─────────────────────────────────────────────────────
        stats.total_time_s = time.perf_counter() - t_start

        all_ids = prompt_ids.squeeze(0).tolist() + generated_ids
        output_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "Generation complete | %d tokens | %.2fs | "
                "%d rounds (%.1f%% full-accept) | %.2f tok/s",
                stats.total_tokens_accepted,
                stats.total_time_s,
                stats.total_rounds,
                100 * stats.acceptance_rate,
                stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text

    # ── Blade hot-swap support (Phase 3) ────────────────────────────────

    def swap_blade(self, blade_name: str, blade_rack: "BladeRack") -> "ReconfigurationProfile":
        """Hot-swap the active alignment blade without reloading the draft model.

        Parameters
        ----------
        blade_name : str
            Name of the blade to activate (must be loaded in blade_rack).
        blade_rack : BladeRack
            The pre-loaded blade cache from Model_mechanics/blade_rack.py.

        Returns
        -------
        ReconfigurationProfile
            Profiling data for this swap (time, memory, adapter params).
        """
        from .blade_rack import BladeRack  # avoid circular import at module level
        new_blade, profile = blade_rack.swap(blade_name)
        self.blade_model = new_blade.blade_model
        self.blade = new_blade
        logger.info(
            "Blade swapped to '%s' in %.1f ms (%.1f MB delta)",
            blade_name, profile.swap_time_ms, profile.memory_delta_mb,
        )
        return profile
