"""
Swiss Knife — Elo Rating System Tournament
===========================================

Implements the Elo rating tournament strategy:
  • Configurable rounds (default 6) with decaying K-factors.
  • The candidate pool size stays constant across all rounds (no elimination).
  • Matches are decided using the blended match function:
      match(A, B) = α · (target_A − target_B) + (1−α) · (blade_A − blade_B)
  • Expected score calculation uses a numerically stable sigmoid to prevent overflow.
  • Champion selection uses zero-centered ratings scaled by beta:
      logits_i = (R_i − 1500) · β
    This matches gsi_swiss scale of softmax(β · points), making the two strategies
    directly comparable under the same β.
"""

import logging
import math
from typing import List, Tuple, Dict, Any

import torch

logger = logging.getLogger(__name__)


def stable_sigmoid(x: float) -> float:
    """Compute a numerically stable sigmoid function to prevent math overflow."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def elo_bracket(
    target_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    normalize: bool = True,
    temperature: float = 1.0,
    rounds: int = 6,
    beta: float = 1.0,
) -> int:
    """Run an Elo rating system tournament over candidates to select a champion.

    Tournament mechanics (Swiss-system pairing + continuous Elo ratings):
      1. Initializes Elo ratings for all candidates to 1500.0.
      2. In each round:
         - Candidates are sorted by current rating (continuous, unlike discrete Swiss points).
         - Paired greedily in rating order, avoiding rematches when possible.
         - Match outcome decided by blended score:
             score = alpha * (target_A - target_B) + (1 - alpha) * (blade_A - blade_B)
         - Actual outcome: s_a=1/s_b=0 (win), s_a=0/s_b=1 (loss), 0.5/0.5 (draw).
         - Expected outcome: e_a = stable_sigmoid((R_a - R_b) * ln10/400)
         - Rating update: R_new = R + K * (actual - expected)
      3. Champion selection (β-scaled, matches gsi_swiss scale):
         - If temperature < 1e-5: greedy argmax of ratings.
         - Otherwise: logits_i = (R_i − 1500) · β  →  softmax → multinomial.
           Zero-centering means only tournament-earned deltas drive the probability,
           making this directly comparable to gsi_swiss's softmax(β · points).

    Parameters
    ----------
    target_scores : torch.Tensor
        Shape ``[N]``. Log-probability under the draft (or verifier) model.
    blade_scores : torch.Tensor
        Shape ``[N]``. DPO blade rewards r_blade for each candidate.
    alpha : float
        Mixing coefficient α ∈ [0, 1].
    normalize : bool
        If True, z-score normalize scores prior to matching.
    temperature : float
        If < 1e-5, use greedy selection; otherwise probabilistic.
    rounds : int
        Number of Elo rounds. Default 6 (K-factors: 40,32,24,16,12,10).
    beta : float
        Scales champion selection logits: logits = (ratings - 1500) * beta.
        Should match the beta used elsewhere in the pipeline.

    Returns
    -------
    int
        Index of the tournament champion.
    """
    N = target_scores.shape[0]
    assert blade_scores.shape[0] == N, "Score tensor shapes must match"

    # Z-score normalization
    if normalize:
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            if t.numel() <= 1:
                return torch.zeros_like(t)
            std = t.std()
            if std < 1e-8:
                return t - t.mean()
            return (t - t.mean()) / (std + 1e-6)
        target_scores = _znorm(target_scores)
        blade_scores  = _znorm(blade_scores)

    # Initialize ratings
    ratings = [1500.0] * N
    paired_before = set()
    indices = list(range(N))

    # Decaying K-factors for each of the rounds
    if rounds == 3:
        k_factors = [40.0, 20.0, 10.0]
    elif rounds == 6:
        k_factors = [40.0, 32.0, 24.0, 16.0, 12.0, 10.0]
    else:
        k_factors = [40.0 * (0.5 ** i) for i in range(rounds)]

    for round_idx in range(rounds):
        k_factor = k_factors[round_idx]

        # Pair candidates based on current ratings DESC, breaking ties with target_scores DESC
        sorted_by_rating = sorted(
            indices,
            key=lambda i: (-ratings[i], -target_scores[i].item()),
        )

        pairs: List[Tuple[int, int]] = []
        unpaired = list(sorted_by_rating)

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)

            # Find best partner: similar rating, avoid rematch when possible
            best_partner_pos = None
            for pos, b in enumerate(unpaired):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in paired_before:
                    best_partner_pos = pos
                    break

            if best_partner_pos is None:
                best_partner_pos = 0

            b = unpaired.pop(best_partner_pos)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        # Bye candidate: rating remains unchanged
        if unpaired:
            bye_idx = unpaired[0]
            logger.debug("Elo Round %d | Bye: c%d (rating=%.1f unchanged)", round_idx + 1, bye_idx, ratings[bye_idx])

        # Execute matches and update ratings
        for a, b in pairs:
            delta_target = target_scores[a] - target_scores[b]
            delta_blade  = blade_scores[a]  - blade_scores[b]
            score = alpha * delta_target + (1.0 - alpha) * delta_blade

            # Determine actual outcome
            if score > 1e-6:
                sa, sb = 1.0, 0.0
                winner = a
            elif score < -1e-6:
                sa, sb = 0.0, 1.0
                winner = b
            else:
                sa, sb = 0.5, 0.5
                winner = None

            # Calculate expected outcomes using stable sigmoid
            diff_ratings = (ratings[a] - ratings[b]) * math.log(10.0) / 400.0
            ea = stable_sigmoid(diff_ratings)
            eb = 1.0 - ea

            # Update ratings
            ratings[a] += k_factor * (sa - ea)
            ratings[b] += k_factor * (sb - eb)

            logger.debug(
                "Elo Round %d (K=%d) | c%d (rating=%.1f) vs c%d (rating=%.1f) → winner=%s | new_ratings: c%d=%.1f, c%d=%.1f",
                round_idx + 1, k_factor, a, ratings[a] - k_factor * (sa - ea),
                b, ratings[b] - k_factor * (sb - eb),
                f"c{winner}" if winner is not None else "draw",
                a, ratings[a], b, ratings[b]
            )

    # Determine champion based on temperature
    if temperature < 1e-5:
        champion = max(
            indices,
            key=lambda i: (ratings[i], target_scores[i].item()),
        )
        logger.debug(
            "Elo champion (Greedy, T=0): c%d (rating=%.1f)", champion, ratings[champion]
        )
    else:
        # Zero-center ratings then scale by beta.
        # logits_i = (R_i - 1500) * beta  →  same scale as gsi_swiss softmax(beta * points).
        ratings_tensor = torch.tensor(ratings, dtype=torch.float, device=target_scores.device)
        logits = (ratings_tensor - 1500.0) * beta
        logits = logits - torch.max(logits)  # numerical stability
        probs = torch.softmax(logits, dim=0)
        champion = int(torch.multinomial(probs, num_samples=1).item())
        logger.debug(
            "Elo champion (Probabilistic, β=%.2f): c%d (rating=%.1f, prob=%.3f)",
            beta, champion, ratings[champion], probs[champion].item()
        )

    return champion


def stochastic_elo_bracket(
    target_scores: torch.Tensor,
    auditor,
    context_ids: torch.Tensor,
    candidate_matrix: torch.Tensor,
    ref_logprobs: torch.Tensor,
    position_idx: int,
    alpha: float,
    normalize: bool = True,
    temperature: float = 1.0,
    rounds: int = 6,
    beta: float = 1.0,
) -> int:
    """Run an Elo tournament over candidates using a stochastic auditor.

    Draws a new stochastic functional of the blade's internal state independently
    for each match.
    """
    N = target_scores.shape[0]
    ratings = [1500.0] * N
    paired_before = set()
    indices = list(range(N))

    if rounds == 3:
        k_factors = [40.0, 20.0, 10.0]
    elif rounds == 6:
        k_factors = [40.0, 32.0, 24.0, 16.0, 12.0, 10.0]
    else:
        k_factors = [40.0 * (0.5 ** i) for i in range(rounds)]

    for round_idx in range(rounds):
        k_factor = k_factors[round_idx]

        sorted_by_rating = sorted(
            indices,
            key=lambda i: (-ratings[i], -target_scores[i].item()),
        )

        pairs: List[Tuple[int, int]] = []
        unpaired = list(sorted_by_rating)

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)

            best_partner_pos = None
            for pos, b in enumerate(unpaired):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in paired_before:
                    best_partner_pos = pos
                    break

            if best_partner_pos is None:
                best_partner_pos = 0

            b = unpaired.pop(best_partner_pos)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        # Execute matches
        for a, b in pairs:
            # Draw a fresh functional per match
            auditor.draw_fresh_functional()
            stochastic_rewards = auditor.score_candidates_for_match(
                context_ids, candidate_matrix, ref_logprobs
            )
            auditor.clear_functional()

            bs_i = stochastic_rewards[position_idx]

            ts_i = target_scores
            if normalize:
                def _znorm(t: torch.Tensor) -> torch.Tensor:
                    if t.numel() <= 1:
                        return torch.zeros_like(t)
                    std = t.std()
                    if std < 1e-8:
                        return t - t.mean()
                    return (t - t.mean()) / (std + 1e-6)
                ts_i = _znorm(ts_i)
                bs_i = _znorm(bs_i)

            delta_target = ts_i[a] - ts_i[b]
            delta_blade  = bs_i[a]  - bs_i[b]
            score = alpha * delta_target + (1.0 - alpha) * delta_blade

            # Determine actual outcome
            if score > 1e-6:
                sa, sb = 1.0, 0.0
            elif score < -1e-6:
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            # Calculate expected outcome using stable sigmoid
            diff_ratings = (ratings[a] - ratings[b]) * math.log(10.0) / 400.0
            ea = stable_sigmoid(diff_ratings)
            eb = 1.0 - ea

            # Update ratings
            ratings[a] += k_factor * (sa - ea)
            ratings[b] += k_factor * (sb - eb)

    # Determine champion based on temperature
    if temperature < 1e-5:
        champion = max(
            indices,
            key=lambda i: (ratings[i], target_scores[i].item()),
        )
    else:
        ratings_tensor = torch.tensor(ratings, dtype=torch.float, device=target_scores.device)
        logits = (ratings_tensor - 1500.0) * beta
        logits = logits - torch.max(logits)
        probs = torch.softmax(logits, dim=0)
        champion = int(torch.multinomial(probs, num_samples=1).item())

    return champion


def elo_score_summary(
    target_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    rounds: int = 3,
) -> Dict[str, Any]:
    """Run Elo system and return diagnostic summary.

    Returns
    -------
    dict with keys:
        'champion_greedy': int
        'ratings': list[float]  — final rating per candidate
        'rounds': int
    """
    N = target_scores.shape[0]

    def _znorm(t):
        if t.numel() <= 1:
            return torch.zeros_like(t)
        std = t.std()
        if std < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / (std + 1e-6)

    ts = _znorm(target_scores)
    bs = _znorm(blade_scores)

    ratings = [1500.0] * N
    paired_before = set()
    indices = list(range(N))

    if rounds == 3:
        k_factors = [40.0, 20.0, 10.0]
    elif rounds == 6:
        k_factors = [40.0, 32.0, 24.0, 16.0, 12.0, 10.0]
    else:
        k_factors = [40.0 * (0.5 ** i) for i in range(rounds)]

    for round_idx in range(rounds):
        k_factor = k_factors[round_idx]

        sorted_by_rating = sorted(
            indices,
            key=lambda i: (-ratings[i], -ts[i].item()),
        )

        pairs = []
        unpaired = list(sorted_by_rating)

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

        for a, b in pairs:
            score = (alpha * (ts[a] - ts[b]) + (1 - alpha) * (bs[a] - bs[b]))
            if score > 1e-6:
                sa, sb = 1.0, 0.0
            elif score < -1e-6:
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            diff_ratings = (ratings[a] - ratings[b]) * math.log(10.0) / 400.0
            ea = stable_sigmoid(diff_ratings)
            eb = 1.0 - ea
            ratings[a] += k_factor * (sa - ea)
            ratings[b] += k_factor * (sb - eb)

    champion = max(indices, key=lambda i: (ratings[i], ts[i].item()))
    return {
        "champion_greedy": champion,
        "ratings": ratings,
        "rounds": rounds,
    }
