"""
Swiss Knife — Swiss-System Tournament (Phase 2)
================================================

Implements the Swiss-system pairing schedule as described in Section 4.3 of
swiss_knife_analysis.pdf:

    • Every candidate plays every round (no elimination after a single loss).
    • In each round, pair candidates by their current cumulative score.
    • After R = ceil(log2(K)) rounds, the candidate with the highest
      cumulative score wins. Ties are broken by secondary draft-likelihood rank.

Complexity: R · K/2 matches total (~2× knockout for R = ceil(log2(K))).

Why Swiss over Knockout:
  • Robust to auditor noise: a strong candidate that loses one close match
    can still win the overall tournament.
  • No bracket luck: pairing adapts round-by-round to actual performance.
  • Recommended for K ≥ 16 or noisy auditors; knockout is fine for small K.

This module exposes the same interface as knockout_bracket() so both can be
used interchangeably by SwissKnifeSpeculativeGenerator.
"""

import logging
import math
from typing import List

import torch

logger = logging.getLogger(__name__)


def swiss_system_bracket(
    target_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    rounds: int = 0,
    normalize: bool = True,
) -> int:
    """Run a Swiss-system tournament over K candidates at a single token position.

    Parameters
    ----------
    target_scores : torch.Tensor
        Shape ``[K]``.  log π_target(token_k | context) for each candidate.
        In Option B, this is the target model's log-probability (calibrated
        fluency signal), NOT the draft log-prob used in Option A.
    blade_scores : torch.Tensor
        Shape ``[K]``.  r_blade(token_k) = β·log(π_blade/π_ref) for each candidate.
    alpha : float
        Mixing coefficient α ∈ [0, 1].
        α → 1  favors target log-likelihood (fluency).
        α → 0  favors blade reward (alignment).
    rounds : int
        Number of Swiss rounds.  0 → automatically use ceil(log2(K)).
    normalize : bool
        If True, z-score normalize target_scores and blade_scores before
        matching to prevent scale dominance. Default True.

    Returns
    -------
    int
        Index (into the original K candidates) of the tournament winner.

    Notes
    -----
    Match function (same as knockout, §4.4 of the PDF):

        match(A, B) = α · (target_A − target_B) + (1−α) · (blade_A − blade_B)

        A wins iff match(A, B) > 0.

    Pairings per round:
      1. Group candidates by current cumulative win count.
      2. Within each group, sort by original draft-rank (secondary criterion)
         and pair adjacent candidates.
      3. If odd group size, the unpaired candidate "floats" down to pair with
         the top of the next score group.
      4. Avoid repeat pairings when possible.
    """
    K = target_scores.shape[0]
    assert blade_scores.shape[0] == K, "Score tensor shapes must match"

    if rounds == 0:
        rounds = max(1, math.ceil(math.log2(K)))

    # Z-score normalization (optional but default ON)
    if normalize:
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            return (t - t.mean()) / (t.std() + 1e-6)
        target_scores = _znorm(target_scores)
        blade_scores  = _znorm(blade_scores)

    # Cumulative wins for each candidate (float for future weighted extension)
    cum_wins = [0.0] * K
    # Track past pairings to avoid rematches
    paired_before = set()
    # Original index mapping (never changes)
    indices = list(range(K))

    round_num = 0
    for _ in range(rounds):
        round_num += 1

        # ── Build pairings ──────────────────────────────────────────────
        # Sort by (cumulative wins DESC, original index ASC for tie-break)
        sorted_by_score = sorted(
            indices,
            key=lambda i: (-cum_wins[i], i),
        )

        pairs: List[tuple] = []
        unpaired = list(sorted_by_score)

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)

            # Find best pairing partner: prefer same score, avoid rematch
            best_partner_pos = None
            for pos, b in enumerate(unpaired):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in paired_before:
                    best_partner_pos = pos
                    break

            if best_partner_pos is None:
                # All potential partners were already paired — allow rematch
                best_partner_pos = 0

            b = unpaired.pop(best_partner_pos)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        # Bye for the last unpaired candidate (if K is odd)
        if unpaired:
            bye_idx = unpaired[0]
            cum_wins[bye_idx] += 0.5  # half-point bye (standard Swiss rule)
            logger.debug("Round %d | Bye: c%d", round_num, bye_idx)

        # ── Execute matches ─────────────────────────────────────────────
        for a, b in pairs:
            delta_target = target_scores[a] - target_scores[b]
            delta_blade  = blade_scores[a]  - blade_scores[b]
            score = alpha * delta_target + (1.0 - alpha) * delta_blade

            if score > 0:
                winner, loser = a, b
            else:
                winner, loser = b, a

            cum_wins[winner] += 1.0
            logger.debug(
                "Round %d | c%d vs c%d → winner=c%d "
                "(Δtgt=%.4f Δblade=%.4f score=%.4f)",
                round_num, a, b, winner,
                delta_target.item(), delta_blade.item(), score.item(),
            )

    # Champion: highest cumulative wins; ties broken by target log-prob
    champion = max(
        indices,
        key=lambda i: (cum_wins[i], target_scores[i].item()),
    )
    logger.debug(
        "Swiss champion: c%d  (wins=%.1f)", champion, cum_wins[champion]
    )
    return champion


def swiss_score_summary(
    target_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    rounds: int = 0,
) -> dict:
    """Run Swiss-system and return full per-candidate win counts (diagnostic).

    Returns
    -------
    dict with keys:
        'champion': int
        'cum_wins': list[float]  — cumulative wins per candidate
        'rounds': int
    """
    K = target_scores.shape[0]
    if rounds == 0:
        rounds = max(1, math.ceil(math.log2(K)))

    def _znorm(t):
        return (t - t.mean()) / (t.std() + 1e-6)

    ts = _znorm(target_scores)
    bs = _znorm(blade_scores)

    cum_wins = [0.0] * K
    paired_before = set()
    indices = list(range(K))

    for rnd in range(rounds):
        sorted_by_score = sorted(indices, key=lambda i: (-cum_wins[i], i))
        unpaired = list(sorted_by_score)
        pairs = []

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)
            best = None
            for pos, b in enumerate(unpaired):
                if (min(a, b), max(a, b)) not in paired_before:
                    best = pos
                    break
            if best is None:
                best = 0
            b = unpaired.pop(best)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        if unpaired:
            cum_wins[unpaired[0]] += 0.5

        for a, b in pairs:
            score = (alpha * (ts[a] - ts[b]) + (1 - alpha) * (bs[a] - bs[b]))
            if score > 0:
                cum_wins[a] += 1.0
            else:
                cum_wins[b] += 1.0

    champion = max(indices, key=lambda i: (cum_wins[i], ts[i].item()))
    return {"champion": champion, "cum_wins": cum_wins, "rounds": rounds}
