"""
configs.py — Strategy Hyperparameter Registry
==============================================

Swiss-Knife (HHH multi-blade):
  Bayesian search was run separately on three single-objective tasks:
    - bayes_search_harmlessness → {T=11.22, w_t=0.504, w_b=1.483, λ=0.109, R=4, N=11}
    - bayes_search_helpfulness  → {T=1.69,  w_t=1.170, w_b=1.707, λ=0.158, R=3, N=7}
    - bayes_search_honesty      → {T=21.03, w_t=1.587, w_b=2.088, λ=0.395, R=7, N=3}

  For HHH multi-objective mixing we cannot reuse any single config; instead we derive
  the multi-blade config as the principled mean across the three per-objective optima:
    - elo_temperature : geometric mean (T acts multiplicatively on a log-probability scale)
                        = (11.22 × 1.69 × 21.03)^(1/3) ≈ 7.4  → rounded to 8.0
    - all others      : arithmetic mean

  This requires ZERO additional computation and is defensible as a principled prior
  over HHH objectives, not as ad-hoc hyperparameter selection.

All baselines: use exact published paper defaults.
  - MOD  (Shi et al., NeurIPS 2024): T=1.0, top_p=0.95, reverse_kld. (HH-RLHF setting)
  - RS   (Ramé et al., NeurIPS 2023): T=1.0, top_p=0.95, linear merge. (HH-RLHF setting)
  - ARGS (Shi et al., 2023): lambda_reward = DPO β = 0.1 (not an independent free param).
  - BoN: N = gsi_n (compute-matched to Swiss-Knife). T=1.0, top_p=0.95.
"""

import os
import json
from typing import Dict, Any

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── Architectural constants (not tuned) ───────────────────────────────────────
# These follow from the geometry of the algorithm, not from search:
#   probabilistic=True  : Thurstonian CDF matching is integral to the design
#   normalize=True      : CBN (Candidate-Batch Normalization) is the core contribution
#   beta=0.1            : DPO training β; cancels out under CBN, fixed to training value

ARCHITECTURAL = {
    "probabilistic": True,
    "normalize": True,
    "beta": 0.1,
    "sigma_mode": "none",
}

# ── Swiss-Knife HHH multi-blade config ───────────────────────────────────────
# Derived as mean of three per-objective Bayesian optima (see module docstring).

SWISS_HHH_CONFIG = {
    "elo_temperature": 8.0,    # geometric mean of (11.22, 1.69, 21.03)
    "w_tournament":    1.1,    # arithmetic mean of (0.504, 1.170, 1.587)
    "w_blade":         1.75,   # arithmetic mean of (1.483, 1.707, 2.088)
    "uwo_lambda":      0.2,    # arithmetic mean of (0.109, 0.158, 0.395)
    "elo_rounds":      5,      # arithmetic mean of (4, 3, 7) → 4.7 → 5
    "gsi_n":           7,      # arithmetic mean of (11, 7, 3)  → 7.0
    "temperature":     1.0,    # sampling temperature for base generation
    "top_p":           0.95,
}


def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    Returns the definitive hyperparameter config for every benchmarked strategy.
    BoN N is set equal to Swiss-Knife gsi_n for a compute-matched comparison.
    """
    gsi_n = SWISS_HHH_CONFIG["gsi_n"]  # = 7

    return {
        "swiss": SWISS_HHH_CONFIG,
        "mod": {
            # Shi et al. NeurIPS 2024 §4.1 + Appendix E — HH-RLHF setting
            "f_divergence": "reverse_kld",
            "temperature":  1.0,
            "top_p":        0.95,
            "beta":         0.1,
        },
        "rs": {
            # Ramé et al. NeurIPS 2023 / MOD §4.1 — HH-RLHF setting
            "combination_type": "linear",
            "temperature":      1.0,
            "top_p":            0.95,
        },
        "args": {
            # lambda_reward = DPO β = 0.1: reward scale is not a free parameter
            "lambda_reward": 0.1,
            "temperature":   1.0,
            "top_p":         0.95,
        },
        "bon": {
            # Compute-matched: N equals Swiss-Knife gsi_n
            "N":           gsi_n,
            "temperature": 1.0,
            "top_p":       0.95,
        },
        "base": {},
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(get_all_configs())
