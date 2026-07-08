"""
Swiss Knife — Elo Rating System Tournament
===========================================

Implements the Elo rating tournament strategy:
  • Fixed to exactly 3 rounds.
  • Decaying K-factors: Round 1: K-factor = 40, Round 2: K-factor = 20, Round 3: K-factor = 10.
  • The candidate pool size stays constant across all rounds (no elimination).
  • Matches are decided using the blended match function:
      match(A, B) = α · (target_A − target_B) + (1−α) · (blade_A − blade_B)
  • Expected score calculation uses a numerically stable sigmoid to prevent overflow.
  • Selection of the champion uses stable temperature-scaled relative strengths:
      P(i) = 10^(R_i / (400 * T)) / sum(10^(R_j / (400 * T)))
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
    rounds: int = 3,
) -> int:
    """Run an Elo rating system tournament over candidates to select a champion.

    This implements a 3-round Elo-based tournament to rank N candidates and select
    the overall winner:
      1. Initializes Elo ratings for all candidates to 1500.0.
      2. In each of the 3 rounds:
         - Candidates are sorted by current rating (and target_scores as tie-breaker).
         - They are paired using a greedy Swiss-system-like algorithm: pairing adjacent
           ranked candidates, avoiding rematches from previous rounds when possible.
         - The outcome of each matchup (A vs B) is decided by a blended score function:
             score = alpha * (target_A - target_B) + (1 - alpha) * (blade_A - blade_B)
           where target scores (fluency/base log-probability) and blade scores
           (implicit DPO reward) are optionally z-score normalized first.
         - Actual matchup outcome is:
             s_a = 1.0, s_b = 0.0 (if score > 1e-6)
             s_a = 0.0, s_b = 1.0 (if score < -1e-6)
             s_a = 0.5, s_b = 0.5 (otherwise, a draw)
         - Expected matchup outcome is computed using a numerically stable sigmoid:
             e_a = 1 / (1 + 10^((R_b - R_a)/400)) = stable_sigmoid( (R_a - R_b) * ln(10)/400 )
             e_b = 1.0 - e_a
         - Ratings are updated using decaying K-factors (Round 1: 40, Round 2: 20, Round 3: 10):
             R_a_new = R_a + K * (s_a - e_a)
             R_b_new = R_b + K * (s_b - e_b)
      3. At the end of the tournament, the champion is chosen:
         - If temperature < 1e-5: greedily select candidate with the highest rating.
         - Otherwise: sample probabilistically from the Boltzmann distribution:
             P(i) = exp( R_i * ln(10) / (400 * temperature) ) / sum_j( exp( R_j * ln(10) / (400 * temperature) ) )

    Parameters
    ----------
    target_scores : torch.Tensor
        Shape ``[N]``. The target model (fluency) scores for the N candidates.
    blade_scores : torch.Tensor
        Shape ``[N]``. The alignment/blade rewards for the N candidates.
    alpha : float
        Mixing coefficient alpha in [0, 1] weighting target fluency vs blade reward.
    normalize : bool
        If True, z-score normalize scores prior to tournament matching.
    temperature : float
        Temperature parameter for relative strength selection of the champion.

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
        # Probabilistic selection based on Elo strengths
        ratings_tensor = torch.tensor(ratings, dtype=torch.float, device=target_scores.device)
        ln10 = math.log(10.0)
        logits = (ratings_tensor * ln10) / (400.0 * temperature)
        logits = logits - torch.max(logits)  # Prevent overflow
        probs = torch.softmax(logits, dim=0)
        champion = int(torch.multinomial(probs, num_samples=1).item())
        logger.debug(
            "Elo champion (Probabilistic, T=%.2f): c%d (rating=%.1f, prob=%.3f)",
            temperature, champion, ratings[champion], probs[champion].item()
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
    rounds: int = 3,
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
        ln10 = math.log(10.0)
        logits = (ratings_tensor * ln10) / (400.0 * temperature)
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
