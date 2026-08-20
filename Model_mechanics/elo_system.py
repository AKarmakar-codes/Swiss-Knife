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
from typing import List, Tuple, Dict, Any, Optional

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
    tilted_rewards: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    hard_draw: bool = False,
    w_tournament: float = 1.0,
    w_blade: float = 0.0,
    uwo_lambda: float = 0.5,
    probabilistic: bool = False,
    return_details: bool = False,
) -> int | Tuple[int, Dict[str, float]]:
    """Run an Elo rating system tournament over candidates to select a champion.

    Tournament mechanics (rating-based pairing + continuous Elo ratings):
      1. Initializes Elo ratings for all candidates to 1500.0.
      2. In each round:
         - Candidates are sorted by current rating (continuous, unlike discrete Swiss points).
         - Paired greedily in rating order, avoiding rematches when possible.
         - Match outcome decided by one of two mechanisms:

           (a) Thurstonian CDF — enabled by ``probabilistic=True`` OR when ``sigmas`` are
               provided.  P(A beats B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B² + ε)).  This gives
               every candidate a non-zero chance of winning, even the weaker one.

           (b) Deterministic step-function match — used when ``probabilistic=False`` and no ``sigmas``
               are supplied. P(A beats B) = 1.0 if score_A > score_B else (0.0 if score_A < score_B else 0.5).
               The candidate with the higher score always wins the match deterministically.

         - Actual outcome: s_a = P, s_b = 1 - P (soft), or Bernoulli draw (hard).
         - Expected outcome: e_a = stable_sigmoid((R_a - R_b) * ln10/400)
         - Rating update: R_new = R + K * (actual - expected)
      3. Champion selection (β-scaled, matches swiss scale):
         - If temperature < 1e-5: greedy argmax of ratings.
         - Otherwise: combined logits from tournament rating + UWO blade signal
           → softmax → multinomial.

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
        Scales champion selection logits.
    tilted_rewards : torch.Tensor, optional
        Shape ``[N]``. Precomputed tilted rewards for all candidates.
    sigmas : torch.Tensor, optional
        Shape ``[N]``. Standard deviation of the blade rewards (uncertainty).
    hard_draw : bool
        If True, sample actual outcome from Bernoulli(P). If False, use continuous P.
    w_tournament : float
        Weight for tournament score in champion selection.
    w_blade : float
        Weight for UWO score (mu - uwo_lambda*sigma) in champion selection.
    uwo_lambda : float
        Uncertainty penalty factor λ for the UWO blade term in champion selection.
        Does NOT gate/reject candidates; penalises high-uncertainty candidates at
        the *softmax selection* stage only.
    probabilistic : bool
        If True, forces Thurstonian CDF P(A beats B) = Φ(z) for every match,
        even when sigmas are all zero (degenerates to a step-function CDF but
        keeps the same code path).  This means a lower-scoring candidate always
        has a positive probability of winning any given match.
        If False (default), deterministic step-function matching is used unless sigmas are
        explicitly provided.

    Returns
    -------
    int
        Index of the tournament champion.
    """
    N = target_scores.shape[0]
    assert blade_scores.shape[0] == N, "Score tensor shapes must match"
    import random
    raw_blade_scores = blade_scores.clone() if isinstance(blade_scores, torch.Tensor) else torch.tensor(blade_scores)

    def _znorm(t: torch.Tensor) -> torch.Tensor:
        if t.numel() <= 1:
            return torch.zeros_like(t)
        std = t.std()
        if std < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / (std + 1e-6)

    # Z-score normalization
    if normalize:
        
        # Calculate standard deviations before normalization
        std_target = target_scores.std().item() if target_scores.numel() > 1 else 1.0
        std_blade = blade_scores.std().item() if blade_scores.numel() > 1 else 1.0
        std_tilted = tilted_rewards.std().item() if (tilted_rewards is not None and tilted_rewards.numel() > 1) else 1.0
        
        target_scores = _znorm(target_scores)
        blade_scores  = _znorm(blade_scores)
        if tilted_rewards is not None:
            tilted_rewards = _znorm(tilted_rewards)
            
        if sigmas is not None:
            if tilted_rewards is not None:
                sigmas_normed = sigmas / (std_tilted + 1e-8)
            else:
                sigmas_normed = sigmas / (std_blade + 1e-8)
        else:
            sigmas_normed = None
    else:
        sigmas_normed = sigmas

    # Initialize ratings
    ratings = [1500.0] * N
    paired_before = set()
    indices = list(range(N))

    # Decaying K-factors: geometric interpolation from K_max=40 down to K_min=10
    # across all R rounds. K_i = 40 * (10/40)^(i/(R-1)), giving K_0=40 and
    # K_{R-1}=10 always, with every round contributing a meaningful update.
    # Eliminates the old special cases for R=3 and R=6 in favour of one
    # consistent formula valid for any R in the search range [2, 10].
    _K_MAX, _K_MIN = 40.0, 10.0
    k_factors = [
        _K_MAX * (_K_MIN / _K_MAX) ** (i / max(rounds - 1, 1))
        for i in range(rounds)
    ]

    for round_idx in range(rounds):
        k_factor = k_factors[round_idx]

        # Pair candidates based on current ratings DESC, breaking ties with blade_scores/tilted_rewards DESC
        tie_breaker = tilted_rewards if tilted_rewards is not None else blade_scores
        sorted_by_rating = sorted(
            indices,
            key=lambda i: (-ratings[i], -tie_breaker[i].item()),
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
            # Determine the raw score difference for this match
            if tilted_rewards is not None:
                diff = tilted_rewards[a] - tilted_rewards[b]
            else:
                diff = (alpha * (target_scores[a] - target_scores[b]) +
                        (1.0 - alpha) * (blade_scores[a] - blade_scores[b]))

            if probabilistic or sigmas_normed is not None:
                # ── Thurstonian Case-V match ──────────────────────────────
                # P(A beats B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B² + ε))
                # When probabilistic=True and sigmas_normed is None, sigma is treated
                # as 0, so the denominator is √ε ≈ 0.0032 — gives a very sharp CDF
                # (essentially deterministic) but remains in the Thurstonian code path.
                if sigmas_normed is not None:
                    if tilted_rewards is not None:
                        var_match = sigmas_normed[a]**2 + sigmas_normed[b]**2
                    else:
                        var_match = ((1.0 - alpha) ** 2) * (sigmas_normed[a]**2 + sigmas_normed[b]**2)
                else:
                    var_match = torch.tensor(0.0, dtype=diff.dtype, device=diff.device)
                denom = torch.sqrt(var_match + 1e-8)
                P_A_beats_B = 0.5 * (1.0 + torch.erf((diff / denom) / math.sqrt(2.0))).item()
            else:
                # ── Deterministic match scoring for elo_baseline ──────────
                # Candidate with higher score always wins the match (1.0 vs 0.0)
                if diff.item() > 1e-6:
                    P_A_beats_B = 1.0
                elif diff.item() < -1e-6:
                    P_A_beats_B = 0.0
                else:
                    P_A_beats_B = 0.5

            # Determine actual outcome
            if hard_draw:
                if random.random() < P_A_beats_B:
                    sa, sb = 1.0, 0.0
                    winner = a
                else:
                    sa, sb = 0.0, 1.0
                    winner = b
            else:
                sa, sb = P_A_beats_B, 1.0 - P_A_beats_B
                winner = a if P_A_beats_B > 0.5 else b

            # Calculate expected outcomes using stable sigmoid
            diff_ratings = (ratings[a] - ratings[b]) * math.log(10.0) / 400.0
            ea = stable_sigmoid(diff_ratings)
            eb = 1.0 - ea

            # Update ratings
            ratings[a] += k_factor * (sa - ea)
            ratings[b] += k_factor * (sb - eb)

            logger.debug(
                "Elo Round %d (K=%d) | c%d (rating=%.1f) vs c%d (rating=%.1f) → winner=%s (prob=%.3f) | new_ratings: c%d=%.1f, c%d=%.1f",
                round_idx + 1, k_factor, a, ratings[a] - k_factor * (sa - ea),
                b, ratings[b] - k_factor * (sb - eb),
                f"c{winner}" if sa != sb else "draw", P_A_beats_B,
                a, ratings[a], b, ratings[b]
            )

    # ── Step C: Uncertainty-Weighted Objective (UWO) Logit ──────────────────
    # Formula: logit_i = w_tournament * znorm(R_i-1500) + w_blade * (znorm(μ_i) - λ · znorm(σ_i))
    ratings_tensor = torch.tensor(ratings, dtype=torch.float, device=target_scores.device)

    def _znorm_term(t: torch.Tensor) -> torch.Tensor:
        """Z-normalize a 1-D tensor across candidates; returns zeros if constant."""
        if t.numel() <= 1:
            return torch.zeros_like(t)
        std = t.std()
        if std < 1e-8:
            return t - t.mean()   # zero-center but don't divide by near-zero std
        return (t - t.mean()) / (std + 1e-6)

    if sigmas is not None:
        if not isinstance(sigmas, torch.Tensor):
            sigmas_tensor = torch.tensor(sigmas, dtype=torch.float, device=target_scores.device)
        else:
            sigmas_tensor = sigmas.to(target_scores.device)
    else:
        sigmas_tensor = None

    if normalize:
        norm_blade = _znorm_term(raw_blade_scores)
        if sigmas_tensor is not None:
            norm_sigma = _znorm_term(sigmas_tensor)
            uwo_normed = norm_blade - uwo_lambda * norm_sigma
        else:
            uwo_normed = norm_blade
        tournament_term = _znorm_term(ratings_tensor - 1500.0)
    else:
        if sigmas_tensor is not None:
            uwo_normed = raw_blade_scores - uwo_lambda * sigmas_tensor
        else:
            uwo_normed = raw_blade_scores
        tournament_term = ratings_tensor - 1500.0

    std_tourn = float(tournament_term.std().item()) if tournament_term.numel() > 1 else 0.0
    std_blade = float(uwo_normed.std().item()) if uwo_normed.numel() > 1 else 0.0
    logit_scale_ratio = round(std_tourn / (std_blade + 1e-8), 4)

    diags = {
        "tournament_term_std": round(std_tourn, 4),
        "blade_term_std": round(std_blade, 6),
        "logit_scale_ratio": logit_scale_ratio,
    }

    if temperature < 1e-5:
        # Greedy (T→0): pick argmax of combined scores
        scores = w_tournament * tournament_term + w_blade * uwo_normed
        champion = int(torch.argmax(scores).item())
        logger.debug(
            "Elo champion (Greedy, T=0): c%d (rating=%.1f)", champion, ratings[champion]
        )
    else:
        # Probabilistic: scale combined logits by 1 / temperature
        logits = (w_tournament * tournament_term + w_blade * uwo_normed) / temperature
        probs = torch.softmax(logits, dim=0)
        champion = int(torch.multinomial(probs, num_samples=1).item())
        logger.debug(
            "Elo champion (Probabilistic, T=%.2f): c%d (rating=%.1f, prob=%.3f)",
            temperature, champion, ratings[champion], probs[champion].item()
        )

    return (champion, diags) if return_details else champion


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

    # Geometric K-factor schedule: K_i = 40*(10/40)^(i/(R-1)), K_0=40, K_{R-1}=10.
    _K_MAX, _K_MIN = 40.0, 10.0
    k_factors = [
        _K_MAX * (_K_MIN / _K_MAX) ** (i / max(rounds - 1, 1))
        for i in range(rounds)
    ]

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
        logits = (ratings_tensor - 1500.0) / temperature
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

    # Geometric K-factor schedule: K_i = 40*(10/40)^(i/(R-1)), K_0=40, K_{R-1}=10.
    _K_MAX, _K_MIN = 40.0, 10.0
    k_factors = [
        _K_MAX * (_K_MIN / _K_MAX) ** (i / max(rounds - 1, 1))
        for i in range(rounds)
    ]

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
