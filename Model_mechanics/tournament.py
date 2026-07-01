"""
Swiss Knife — Tournament Engine

Implements the knockout bracket from Listing 1 of the proposal:

    match(A, B)  =  α · [log π_draft(A) − log π_draft(B)]
                  + (1 − α) · [r_blade(A) − r_blade(B)]

    A wins  iff  match(A, B) > 0.

Single-elimination bracket: K candidates → log₂(K) rounds → 1 winner.
"""

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def knockout_bracket(
    draft_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
) -> int:
    """Run a single-elimination knockout bracket over K candidates.

    This is the *exact* translation of Listing 1 from the Swiss Knife
    proposal (Section 5, Option A).

    Parameters
    ----------
    draft_scores : torch.Tensor
        Shape ``[K]``.  log π_draft(span_k | prompt)  for each candidate.
    blade_scores : torch.Tensor
        Shape ``[K]``.  r_blade(span_k | prompt) = β·log(π_blade/π_ref)
        for each candidate.
    alpha : float
        Mixing coefficient  α ∈ [0, 1].
        α → 1  favors draft fluency.
        α → 0  favors blade alignment.

    Returns
    -------
    int
        Index (into the original K candidates) of the tournament winner.

    Notes
    -----
    The match score between candidates A and B is:

        score = α · (draft_A − draft_B) + (1 − α) · (blade_A − blade_B)

    A wins iff score > 0.  Ties (score == 0) go to the first candidate
    (deterministic tie-break; can be randomized later).

    If K is odd (non-power-of-2), the last unpaired candidate receives
    a bye and advances automatically.
    """
    K = draft_scores.shape[0]
    assert blade_scores.shape[0] == K, \
        f"Score tensor shapes must match: draft={draft_scores.shape}, blade={blade_scores.shape}"

    active: List[int] = list(range(K))

    round_num = 0
    while len(active) > 1:
        round_num += 1
        next_round: List[int] = []

        for i in range(0, len(active) - 1, 2):
            a, b = active[i], active[i + 1]

            # ── THE MATCH FUNCTION (Eq. from §4.4) ─────────────────────
            delta_draft = draft_scores[a] - draft_scores[b]
            delta_blade = blade_scores[a] - blade_scores[b]
            score = alpha * delta_draft + (1.0 - alpha) * delta_blade
            # ───────────────────────────────────────────────────────────

            winner = a if score > 0 else b
            next_round.append(winner)

            logger.debug(
                "Round %d | Match: c%d vs c%d  "
                "Δdraft=%.4f  Δblade=%.4f  score=%.4f → winner=c%d",
                round_num, a, b,
                delta_draft.item(), delta_blade.item(), score.item(),
                winner,
            )

        # Bye: if odd number, last candidate advances without a match
        if len(active) % 2 == 1:
            bye_idx = active[-1]
            next_round.append(bye_idx)
            logger.debug("Round %d | Bye: c%d advances", round_num, bye_idx)

        active = next_round

    champion = active[0]
    logger.debug("Tournament champion: c%d", champion)
    return champion


def stochastic_knockout_bracket(
    draft_scores: torch.Tensor,
    auditor,
    context_ids: torch.Tensor,
    candidate_matrix: torch.Tensor,
    ref_logprobs: torch.Tensor,
    position_idx: int,
    alpha: float,
    normalize: bool = True,
) -> int:
    """Run a single-elimination knockout bracket over K candidates with a stochastic auditor.

    Draws a new stochastic functional of the blade's internal state independently
    for each match.

    Optimisation
    ------------
    mc_dropout / random_proj:
        The CALLER (speculative_generator.py) must call
        auditor.precompute_hidden_states(context_ids, candidate_matrix) ONCE
        per outer iteration before iterating over positions.  Each match then
        applies the perturbation in-memory — ZERO model forward passes.
    head_subsample:
        1 forward pass per match (hooks are applied per-match as before).
    """
    K = draft_scores.shape[0]

    active: List[int] = list(range(K))

    if normalize:
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            return (t - t.mean()) / (t.std() + 1e-6)
        ts_i_norm = _znorm(draft_scores)
    else:
        ts_i_norm = draft_scores

    round_num = 0
    while len(active) > 1:
        round_num += 1
        next_round: List[int] = []

        for i in range(0, len(active) - 1, 2):
            a, b = active[i], active[i + 1]

            # Draw a fresh functional per match (registers hooks for head_subsample;
            # randomness is injected inside score_candidates_for_match for mc_dropout/random_proj).
            auditor.draw_fresh_functional()
            stochastic_rewards = auditor.score_candidates_for_match(
                context_ids, candidate_matrix, ref_logprobs
            )
            auditor.clear_functional()

            # Extract and optionally normalise blade scores for this position.
            bs_i = stochastic_rewards[position_idx]
            if normalize:
                bs_i = _znorm(bs_i)

            delta_draft = ts_i_norm[a] - ts_i_norm[b]
            delta_blade = bs_i[a] - bs_i[b]
            score = alpha * delta_draft + (1.0 - alpha) * delta_blade

            winner = a if score > 0 else b
            next_round.append(winner)

            logger.debug(
                "Round %d | Match: c%d vs c%d (stochastic)  "
                "Δdraft=%.4f  Δblade=%.4f  score=%.4f → winner=c%d",
                round_num, a, b,
                delta_draft.item(), delta_blade.item(), score.item(),
                winner,
            )

        # Bye
        if len(active) % 2 == 1:
            bye_idx = active[-1]
            next_round.append(bye_idx)
            logger.debug("Round %d | Bye: c%d advances", round_num, bye_idx)

        active = next_round

    champion = active[0]
    logger.debug("Stochastic tournament champion: c%d", champion)
    return champion



def tournament_score_matrix(
    draft_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute the full pairwise match-score matrix (diagnostic utility).

    Parameters
    ----------
    draft_scores, blade_scores : torch.Tensor
        Shape ``[K]`` each.
    alpha : float
        Mixing coefficient.

    Returns
    -------
    torch.Tensor
        Shape ``[K, K]`` where ``M[i, j]`` = match(i, j).
        Positive → i beats j.  Anti-symmetric: M[i,j] = −M[j,i].
    """
    K = draft_scores.shape[0]
    # Pairwise differences via broadcasting
    delta_draft = draft_scores.unsqueeze(1) - draft_scores.unsqueeze(0)  # [K, K]
    delta_blade = blade_scores.unsqueeze(1) - blade_scores.unsqueeze(0)  # [K, K]
    return alpha * delta_draft + (1.0 - alpha) * delta_blade
