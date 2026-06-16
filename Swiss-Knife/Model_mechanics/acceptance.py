"""
Swiss Knife — Phase 1: Reward-Shifted Acceptance Math
======================================================

Implements the localized acceptance probability gate from the reward-shifted
speculative sampling framework:

    P_accept(x) = min(1,  exp(β · S_auditor(x)) / Z_local )

where Z_local approximates the true partition function by evaluating the auditor
score only over the top-K draft candidates (the "local vocabulary" V_K).

─────────────────────────────────────────────────────────────────────────────
THEORETICAL DERIVATION — Why Draft Probability Cancels
─────────────────────────────────────────────────────────────────────────────

Goal: sample from the reward-shifted aligned distribution

    q(x) ∝ p_draft(x) · exp(β · S_auditor(x))                   ... (1)

using the draft model p_draft as a proposal, without running the full target.

Standard speculative sampling (Leviathan et al., 2023) accepts a proposed token x
with probability:

    P_accept(x) = min(1,  q(x) / p_draft(x) )                   ... (2)

Substituting (1) into (2):

    P_accept(x) = min(1,   p_draft(x) · exp(β·S(x)) / Z
                          ──────────────────────────────── )
                             p_draft(x)

                = min(1,  exp(β · S_auditor(x)) / Z )            ... (3)

The p_draft(x) term cancels exactly from numerator and denominator.
The normalizing constant Z = Σ_v p_draft(v) · exp(β·S(v)) over all v ∈ V.

─────────────────────────────────────────────────────────────────────────────
Z_local APPROXIMATION
─────────────────────────────────────────────────────────────────────────────

Computing Z over the full vocabulary V (|V| ~ 150k for Qwen) requires calling
the auditor for every token — prohibitively expensive.

Instead we approximate over the top-K draft candidates V_K ⊂ V:

    Z_local = Σ_{v ∈ V_K}  p_draft(v) · exp(β · S_auditor(v))   ... (4)

This is exact if the aligned target mass is concentrated in V_K (true when β
is moderate and the draft is reasonable). The approximation error is bounded by
the total draft probability mass outside V_K, which shrinks with larger K.

─────────────────────────────────────────────────────────────────────────────
RELATIONSHIP TO OPTION B
─────────────────────────────────────────────────────────────────────────────

In Option B (speculative_generator.py), the *tournament itself* acts as the
acceptance decision — a candidate wins the tournament iff it is pairwise-
preferred to the draft's top-1. The Z_local formulation (this file) gives an
alternative, theoretically grounded acceptance probability that can be applied
as an additional coin-flip gate after the tournament winner is chosen
(config.use_acceptance_gate = True), or used standalone for a single-auditor
baseline without tournaments.

─────────────────────────────────────────────────────────────────────────────
"""

import math
import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core mathematical primitives
# ─────────────────────────────────────────────────────────────────────────────


def compute_z_local(
    draft_logits_topk: torch.Tensor,
    auditor_scores_topk: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute the localized partition function Z_local over the top-K vocabulary.

    Implements Equation (4) from the module docstring:

        Z_local = Σ_{v ∈ V_K}  p_draft(v) · exp(β · S_auditor(v))

    where V_K is the set of top-K draft candidates at a single token position.

    Parameters
    ----------
    draft_logits_topk : torch.Tensor
        Shape ``[K]``.  Raw (unnormalized) logits from the draft model for the
        top-K candidate tokens at a single position.
    auditor_scores_topk : torch.Tensor
        Shape ``[K]``.  Scalar auditor scores S_auditor(v) for each of the K
        candidates. In Swiss Knife, this is the DPO blade reward r_blade(v).
    beta : float
        Temperature scaling for the reward.  β > 0 sharpens alignment; β → 0
        collapses to pure draft sampling.

    Returns
    -------
    torch.Tensor
        Scalar — Z_local value.  Always positive (guaranteed by exp).

    Notes
    -----
    We use log-sum-exp for numerical stability:

        Z_local = Σ_k  softmax(logits)_k · exp(β · S_k)
                = Σ_k  exp(log_softmax(logits)_k + β · S_k)
                = exp( logsumexp(log_softmax(logits) + β·S) )
    """
    log_p_draft = F.log_softmax(draft_logits_topk.float(), dim=-1)  # [K]
    exponent = log_p_draft + beta * auditor_scores_topk.float()      # [K]
    # Use logsumexp for stability; clamp to avoid inf on extreme inputs
    log_z = torch.logsumexp(exponent.clamp(min=-500, max=500), dim=0)
    z_local = torch.exp(log_z.clamp(max=88.0))  # exp(88) ≈ 1.65e38, safe in float32
    return z_local


def acceptance_prob(
    auditor_score_winner: torch.Tensor,
    z_local: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute the reward-shifted acceptance probability for the tournament winner.

    Implements Equation (3):

        P_accept = min(1,  exp(β · S_auditor(winner)) / Z_local )

    Parameters
    ----------
    auditor_score_winner : torch.Tensor
        Scalar — S_auditor for the chosen winner token.
    z_local : torch.Tensor
        Scalar — Z_local computed by ``compute_z_local``.
    beta : float
        Reward temperature.

    Returns
    -------
    torch.Tensor
        Scalar in [0, 1] — acceptance probability.
    """
    log_numerator = beta * auditor_score_winner.float()
    log_z = torch.log(z_local.clamp(min=1e-12))
    log_ratio = log_numerator - log_z
    # Clamp ratio to [0, 1]
    p_accept = torch.exp(log_ratio).clamp(max=1.0)
    return p_accept


def speculative_coin_flip(p_accept: torch.Tensor) -> bool:
    """Perform the speculative decoding Bernoulli acceptance coin flip.

    Returns True (accept) with probability p_accept, False (reject) otherwise.

    Parameters
    ----------
    p_accept : torch.Tensor
        Scalar in [0, 1].

    Returns
    -------
    bool
        True → accept the proposed token.
        False → reject; caller should resample from the residual distribution.
    """
    u = torch.rand(1, device=p_accept.device)
    return bool(u.item() < p_accept.item())


def resample_from_residual(
    draft_logits_topk: torch.Tensor,
    auditor_scores_topk: torch.Tensor,
    beta: float,
    winner_idx: int,
) -> int:
    """Sample a fallback token from the residual (q - p_draft)+ distribution.

    Called when the acceptance coin flip rejects the tournament winner.
    Standard speculative decoding resamples from:

        (q - p_draft)+ / Z_residual

    where q is the aligned target distribution and (·)+ = max(0, ·).

    In the localized setting, this is approximated over V_K.

    Parameters
    ----------
    draft_logits_topk : torch.Tensor
        Shape ``[K]`` — raw draft logits for top-K candidates.
    auditor_scores_topk : torch.Tensor
        Shape ``[K]`` — auditor scores for top-K candidates.
    beta : float
        Reward temperature.
    winner_idx : int
        The index (in top-K) of the rejected tournament winner.

    Returns
    -------
    int
        Index into the top-K array for the fallback token.
    """
    K = draft_logits_topk.shape[0]
    log_p_draft = F.log_softmax(draft_logits_topk.float(), dim=-1)  # [K]
    p_draft = torch.exp(log_p_draft)                                 # [K]

    # Aligned target probabilities (unnormalized)
    q_unnorm = p_draft * torch.exp(beta * auditor_scores_topk.float())  # [K]
    q = q_unnorm / q_unnorm.sum().clamp(min=1e-12)

    # Residual: (q - p_draft)+
    residual = (q - p_draft).clamp(min=0.0)

    # If all residual mass is zero (degenerate case), fall back to q
    if residual.sum().item() < 1e-9:
        residual = q

    residual = residual / residual.sum()

    # Sample from residual
    fallback_idx = int(torch.multinomial(residual, num_samples=1).item())
    return fallback_idx


# ─────────────────────────────────────────────────────────────────────────────
# Single-auditor baseline loop (standalone, no tournament)
# ─────────────────────────────────────────────────────────────────────────────

class SingleAuditorBaseline:
    """Single-candidate token-level speculative sampler using the Z_local gate.

    This is a standalone baseline that does NOT use the tournament.
    At each token position:
      1. Draft proposes top-K tokens with their logits.
      2. Auditor scores all K tokens.
      3. Z_local is computed over the K candidates.
      4. The highest-auditor-scored token is the proposed winner.
      5. Acceptance coin flip: P_accept = min(1, exp(β·S_winner)/Z_local).
      6. If rejected, resample from residual.

    Use this to compare against the tournament approach in ablations.

    Parameters
    ----------
    beta : float
        Reward temperature.
    top_k : int
        Number of top draft candidates to evaluate.
    """

    def __init__(self, beta: float, top_k: int = 8):
        self.beta = beta
        self.top_k = top_k

    def select_token(
        self,
        draft_logits: torch.Tensor,
        auditor_scores_topk: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> int:
        """Select a token for one position using the acceptance gate.

        Parameters
        ----------
        draft_logits : torch.Tensor
            Shape ``[vocab_size]`` — full draft logit distribution.
        auditor_scores_topk : torch.Tensor
            Shape ``[K]`` — auditor scores for the top-K tokens.
        topk_indices : torch.Tensor
            Shape ``[K]`` — vocabulary indices of the top-K tokens.

        Returns
        -------
        int
            Vocabulary index of the selected token.
        """
        # Extract top-K logits from full logit vector
        K = self.top_k
        draft_topk_logits, topk_idx = torch.topk(draft_logits, K)

        # Greedy auditor winner = highest auditor score
        winner_in_topk = int(auditor_scores_topk.argmax().item())

        z_local = compute_z_local(draft_topk_logits, auditor_scores_topk, self.beta)
        p_acc = acceptance_prob(
            auditor_scores_topk[winner_in_topk], z_local, self.beta
        )

        if speculative_coin_flip(p_acc):
            return int(topk_indices[winner_in_topk].item())
        else:
            fallback_in_topk = resample_from_residual(
                draft_topk_logits, auditor_scores_topk, self.beta, winner_in_topk
            )
            return int(topk_indices[fallback_in_topk].item())
