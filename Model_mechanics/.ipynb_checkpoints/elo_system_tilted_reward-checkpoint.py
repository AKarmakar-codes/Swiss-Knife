"""
Swiss Knife — Elo Rating System Tournament
===========================================

Implements the Elo rating tournament strategy:
  • Configurable rounds (default 6) with decaying K-factors.
  • The candidate pool size stays constant across all rounds (no elimination).
  • Matches are decided using the single provided score (e.g., Tilted Reward).
  • Expected score calculation uses a numerically stable sigmoid to prevent overflow.
  • Champion selection uses zero-centered ratings scaled by beta:
      logits_i = (R_i − 1500) · β
    This matches gsi_swiss scale of softmax(β · points).
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
    scores: torch.Tensor,
    normalize: bool = True,
    temperature: float = 1.0,
    rounds: int = 6,
    beta: float = 1.0,
) -> int:
    """Run an Elo rating system tournament over candidates to select a champion.

    Parameters
    ----------
    scores : torch.Tensor
        Shape ``[N]``. The combined reward for each candidate (e.g., Tilted Reward).
    normalize : bool
        If True, z-score normalize scores prior to matching.
    temperature : float
        If < 1e-5, use greedy selection; otherwise probabilistic.
    rounds : int
        Number of Elo rounds. Default 6 (K-factors: 40,32,24,16,12,10).
    beta : float
        Scales champion selection logits: logits = (ratings - 1500) * beta.

    Returns
    -------
    int
        Index of the tournament champion.
    """
    N = scores.shape[0]

    # Z-score normalization
    if normalize:
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            if t.numel() <= 1:
                return torch.zeros_like(t)
            std = t.std()
            if std < 1e-8:
                return t - t.mean()
            return (t - t.mean()) / (std + 1e-6)
        scores = _znorm(scores)

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

        # Pair candidates based on current ratings DESC, breaking ties with scores DESC
        sorted_by_rating = sorted(
            indices,
            key=lambda i: (-ratings[i], -scores[i].item()),
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
            # Deterministic win outcome based directly on tilted score differences
            score_diff = scores[a] - scores[b]

            if score_diff > 1e-6:
                sa, sb = 1.0, 0.0
                winner = a
            elif score_diff < -1e-6:
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
            key=lambda i: (ratings[i], scores[i].item()),
        )
        logger.debug(
            "Elo champion (Greedy, T=0): c%d (rating=%.1f)", champion, ratings[champion]
        )
    else:
        # Zero-center ratings then scale by temperature.
        ratings_tensor = torch.tensor(ratings, dtype=torch.float, device=scores.device)
        logits = (ratings_tensor - 1500.0) / temperature
        logits = logits - torch.max(logits)  # numerical stability
        probs = torch.softmax(logits, dim=0)
        champion = int(torch.multinomial(probs, num_samples=1).item())
        logger.debug(
            "Elo champion (Probabilistic, T=%.2f): c%d (rating=%.1f, prob=%.3f)",
            temperature, champion, ratings[champion], probs[champion].item()
        )

    return champion


def elo_score_summary(
    scores: torch.Tensor,
    rounds: int = 3,
) -> Dict[str, Any]:
    """Run Elo system and return diagnostic summary."""
    N = scores.shape[0]

    def _znorm(t):
        if t.numel() <= 1:
            return torch.zeros_like(t)
        std = t.std()
        if std < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / (std + 1e-6)

    normed_scores = _znorm(scores)

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
            key=lambda i: (-ratings[i], -normed_scores[i].item()),
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
            score_diff = normed_scores[a] - normed_scores[b]
            if score_diff > 1e-6:
                sa, sb = 1.0, 0.0
            elif score_diff < -1e-6:
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            diff_ratings = (ratings[a] - ratings[b]) * math.log(10.0) / 400.0
            ea = stable_sigmoid(diff_ratings)
            eb = 1.0 - ea
            ratings[a] += k_factor * (sa - ea)
            ratings[b] += k_factor * (sb - eb)

    champion = max(indices, key=lambda i: (ratings[i], normed_scores[i].item()))
    return {
        "champion_greedy": champion,
        "ratings": ratings,
        "rounds": rounds,
    }