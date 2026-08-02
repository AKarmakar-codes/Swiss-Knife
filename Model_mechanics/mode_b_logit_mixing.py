"""
Swiss Knife — Mode-B Multi-Blade Logit Mixing Engine
=====================================================

This module implements candidate step-level logit mixing and tournament-based
champion selection for Swiss-Knife Mode B using multi-blade DPO alignment objectives
(e.g., Helpfulness + Harmlessness + Truthfulness).

───────────────────────────────────────────────────────────────────────────────
1. CONCEPTUAL PARADIGM: MULTI-OBJECTIVE DECODING (MOD) VS. SWISS-KNIFE MODE B
───────────────────────────────────────────────────────────────────────────────

+-------------------------+------------------------------------------+----------------------------------------------------------------------------------------------------+
| Feature                 | Multi-Objective Decoding (MOD)           | Swiss-Knife Mode-B (Elo-Swiss Logit Mixing)                                                        |
+-------------------------+------------------------------------------+----------------------------------------------------------------------------------------------------+
| Mixing Granularity      | Token-level linear log-probability sum   | Step/Span-level candidate scoring & tournament                                                     |
| Logit / Score Formula   | log P_MOD(y_t) = Σ w_k log P_k(y_t)      | logit_i = w_tournament · Z-norm((R_i - 1500)/T) + w_blade · Z-norm(μ_composite_i - λ σ_composite_i)|
| Robustness to Spikes    | Low (single rogue token logit degrades)  | High (Thurstonian Elo ranking flattens extreme outliers via variance normalization)               |
| Uncertainty Awareness   | None (assumes calibrated log-probs)      | Explicit UWO (penalizes candidate uncertainty σ_i via Log-Ratio Proxy)                             |
| Acceptance Policy       | Greedy / Nucleus token sampling          | Unconditional Acceptance of Elo Champion (low latency, zero verifier fallback gating overhead)     |
+-------------------------+------------------------------------------+----------------------------------------------------------------------------------------------------+

───────────────────────────────────────────────────────────────────────────────
2. MULTI-BLADE LOGIT MIXING MECHANICS IN MODE B
───────────────────────────────────────────────────────────────────────────────

Given prompt prefix x:

Step A: Draft Candidate Generation
-----------------------------------
The base/drafter model π_draft samples N candidate reasoning steps: S_1, S_2, ..., S_N.

Step B: Multi-Blade Scoring & Uncertainty Estimation
---------------------------------------------------
For each candidate step S_i, rewards and uncertainties are evaluated for all active DPO blades
(e.g., Helpfulness Blade and Harmlessness Blade):

1. Helpfulness Blade:
   • Reward:      μ_help_i = r_help(x, S_i)
   • Uncertainty: σ_help_i = | μ_help_i - (1 / β) · (log π_help(S_i) - log π_draft(S_i)) |

2. Harmlessness Blade:
   • Reward:      μ_harm_i = r_harm(x, S_i)
   • Uncertainty: σ_harm_i = | μ_harm_i - (1 / β) · (log π_harm(S_i) - log π_draft(S_i)) |

3. Composite Multi-Blade Reward & Uncertainty:
   Given Pareto weight γ_help ∈ [0, 1] (with γ_harm = 1.0 - γ_help):
   • Composite Reward:      μ_composite_i = γ_help · μ_help_i + γ_harm · μ_harm_i
   • Composite Uncertainty: σ_composite_i = γ_help · σ_help_i + γ_harm · σ_harm_i

   For general multi-blade sets {Blade_k} with normalized weights w_k (Σ w_k = 1):
   • μ_composite_i = Σ_k w_k · μ_{k, i}
   • σ_composite_i = Σ_k w_k · σ_{k, i}

Step C: Thurstonian Elo Tournament
----------------------------------
Candidates enter a K-round Elo tournament with initial rating R_i = 1500.0 for all i.
Pairwise matches between candidate A and B are resolved using the Thurstonian Case-V Model:

   P(A beats B) = Normal_CDF( (μ_composite_A - μ_composite_B) / √(σ_composite_A² + σ_composite_B² + ε) )

Elo updates follow match outcomes across rounds with decaying K-factors (40, 32, 24, 16, 12, 10):
   R_A_new = R_A + K_round · ( P(A beats B) - Sigmoid( (R_A - R_B) · ln(10) / 400 ) )

Step D: Final Z-Normalized Champion Logit Synthesis
----------------------------------------------------
To compute final selection probabilities across candidates, Mode B synthesizes the
Tournament Rating logit and the Uncertainty-Weighted Objective (UWO) Blade logit:

   logit_i = w_tournament · Z-norm( (R_i - 1500) / T_elo ) + w_blade · Z-norm( μ_composite_i - λ_uwo · σ_composite_i )

where Z-normalization across the candidate pool N is defined as:
   Z-norm(X_i) = (X_i - mean(X)) / (std(X) + 1e-6)

Why Z-Normalization is Essential:
----------------------------------
Raw Elo ratings operate on a scale of [1400, 1600] (divided by temperature T_elo, producing values ~ ±50),
whereas DPO rewards μ operate in [-1.0, 1.0]. Without Z-normalization, the tournament rating term
completely overwhelms the blade reward term regardless of parameter settings. Z-normalization maps both
components to zero mean and unit variance across candidate options, making w_tournament and w_blade
true, controllable hyperparameters.

Step E: Softmax Selection & Unconditional Acceptance
----------------------------------------------------
The champion step index is drawn via a softmax probability distribution over the synthesized logits:

   P(Select Step i) = exp(logit_i) / Σ_j exp(logit_j)

The selected champion step is unconditionally accepted (Mode B hallmark), eliminating verifier
rejection gating overhead and fallback latency.

───────────────────────────────────────────────────────────────────────────────
3. KEY ADVANTAGES FOR AAAI PARETO BENCHMARKS VS. MOD
───────────────────────────────────────────────────────────────────────────────

1. Non-Convex Pareto Control: Adjusting γ_help in [0, 1] smoothly traces out superior
   Pareto frontiers on Helpfulness vs. Harmlessness compared to token-level MOD.
2. Elimination of Syntax Corruption: Generating multi-token candidate steps from π_draft
   guarantees syntactic coherence, whereas MOD's token-level logit addition often mixes
   contradictory token probabilities.
3. Robustness via Thurstonian CDF: In uncertainty-heavy regions (where blades disagree or
   predict high variance), the Thurstonian match outcome probability flattens toward 0.5,
   protecting generation from over-exploiting noisy reward estimates.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F

from .config import SwissKnifeConfig
from .elo_system import elo_bracket
from .sigma_estimator import estimate_mu_sigma, RunningPercentileThreshold
from .elo_swiss import EloSwissGenerator, EloSwissStats
from evaluation.logprob_utilities import compute_logprobs_batched, compute_logprob

logger = logging.getLogger(__name__)


def compute_dual_blade_mode_b_step(
    prefix_ids: torch.Tensor,
    candidate_step_ids: List[torch.Tensor],
    draft_logprobs: torch.Tensor,
    helpfulness_blade: Any,
    harmlessness_blade: Any,
    gamma_help: float = 0.5,
    beta: float = 0.1,
    cfg: Optional[Any] = None,
    w_tournament: float = 1.0,
    w_blade: float = 1.0,
    uwo_lambda: float = 0.5,
    elo_temperature: float = 15.0,
    elo_rounds: int = 6,
    sigma_mode: str = "log_ratio_proxy",
    probabilistic: bool = True,
    hard_draw: bool = False,
    alpha: float = 0.5,
) -> int:
    """Computes step-level candidate selection for Mode B using dual DPO blades (Helpfulness + Harmlessness).

    Parameters
    ----------
    prefix_ids : torch.Tensor
        Tokenized prompt prefix / history.
    candidate_step_ids : list of torch.Tensor
        List of candidate step token ID tensors.
    draft_logprobs : torch.Tensor
        Precomputed log-probabilities of candidate steps under the draft model.
    helpfulness_blade : DPOBlade
        Helpfulness alignment blade.
    harmlessness_blade : DPOBlade
        Harmlessness alignment blade.
    gamma_help : float
        Pareto weight for helpfulness in [0, 1]. Harmlessness weight is (1.0 - gamma_help).
    beta : float
        DPO reward scaling factor beta.
    cfg : SwissKnifeConfig, optional
        Optional config object overriding hyperparameter defaults.
    w_tournament : float
        Weight of tournament rating logit term.
    w_blade : float
        Weight of UWO composite blade logit term.
    uwo_lambda : float
        Uncertainty penalty coefficient lambda in (mu - lambda * sigma).
    elo_temperature : float
        Temperature for Elo champion softmax selection.
    elo_rounds : int
        Number of Elo rating tournament rounds.
    sigma_mode : str
        Uncertainty mode: 'none', 'log_ratio_proxy', or 'mc_dropout'.
    probabilistic : bool
        If True, forces Thurstonian CDF matching probability.
    hard_draw : bool
        If True, uses hard Bernoulli match outcome draws.
    alpha : float
        Likelihood vs reward mixing weight inside elo_bracket.

    Returns
    -------
    int
        Index of the selected champion step candidate.
    """
    if cfg is not None:
        sigma_mode = getattr(cfg, "sigma_mode", sigma_mode)
        alpha = getattr(cfg, "alpha", alpha)
        beta = getattr(cfg, "beta", beta)
        w_tournament = getattr(cfg, "w_tournament", w_tournament)
        w_blade = getattr(cfg, "w_blade", w_blade)
        uwo_lambda = getattr(cfg, "uwo_lambda", uwo_lambda)
        elo_temperature = getattr(cfg, "elo_temperature", elo_temperature)
        elo_rounds = getattr(cfg, "elo_rounds", elo_rounds)
        probabilistic = getattr(cfg, "probabilistic", probabilistic)
        hard_draw = getattr(cfg, "hard_draw", hard_draw)

    # Ensure prefix_ids is 2D [1, seq_len]
    if prefix_ids.ndim == 1:
        prefix_ids = prefix_ids.unsqueeze(0)

    gamma_help = max(0.0, min(1.0, float(gamma_help)))
    gamma_harm = 1.0 - gamma_help

    # 1. Score Helpfulness Blade & Uncertainty
    mu_help, sigma_help = estimate_mu_sigma(
        prefix_ids=prefix_ids,
        step_token_ids_list=candidate_step_ids,
        blade=helpfulness_blade,
        sigma_mode=sigma_mode,
        draft_logprobs=draft_logprobs,
        beta=beta,
    )

    # 2. Score Harmlessness Blade & Uncertainty
    mu_harm, sigma_harm = estimate_mu_sigma(
        prefix_ids=prefix_ids,
        step_token_ids_list=candidate_step_ids,
        blade=harmlessness_blade,
        sigma_mode=sigma_mode,
        draft_logprobs=draft_logprobs,
        beta=beta,
    )

    # 3. Composite Multi-Blade Reward & Uncertainty
    mu_composite = gamma_help * mu_help + gamma_harm * mu_harm
    sigma_composite = gamma_help * sigma_help + gamma_harm * sigma_harm

    # 4. Perform Thurstonian Elo Tournament & Z-Norm Champion Selection
    champion_idx = elo_bracket(
        target_scores=draft_logprobs,
        blade_scores=mu_composite,
        alpha=alpha,
        normalize=True,
        temperature=elo_temperature,
        rounds=elo_rounds,
        beta=beta,
        sigmas=sigma_composite if sigma_mode != "none" else None,
        hard_draw=hard_draw,
        w_tournament=w_tournament,
        w_blade=w_blade,
        uwo_lambda=uwo_lambda,
        probabilistic=probabilistic,
    )

    return champion_idx


def compute_multi_blade_mode_b_step(
    prefix_ids: torch.Tensor,
    candidate_step_ids: List[torch.Tensor],
    draft_logprobs: torch.Tensor,
    blades: Dict[str, Any],
    weights: Dict[str, float],
    beta: float = 0.1,
    cfg: Optional[Any] = None,
    w_tournament: float = 1.0,
    w_blade: float = 1.0,
    uwo_lambda: float = 0.5,
    elo_temperature: float = 15.0,
    elo_rounds: int = 6,
    sigma_mode: str = "log_ratio_proxy",
    probabilistic: bool = True,
    hard_draw: bool = False,
    alpha: float = 0.5,
) -> int:
    """Computes step-level candidate selection for Mode B using arbitrary M >= 2 DPO blades.

    Parameters
    ----------
    prefix_ids : torch.Tensor
        Tokenized prompt prefix / history.
    candidate_step_ids : list of torch.Tensor
        List of candidate step token ID tensors.
    draft_logprobs : torch.Tensor
        Precomputed log-probabilities under draft model.
    blades : dict of {str: DPOBlade}
        Mapping of blade names to DPOBlade instances.
    weights : dict of {str: float}
        Mapping of blade names to Pareto scalar weights.
    beta : float
        DPO reward scaling factor beta.
    cfg : SwissKnifeConfig, optional
        Configuration object overriding parameters.

    Returns
    -------
    int
        Index of selected champion candidate.
    """
    if cfg is not None:
        sigma_mode = getattr(cfg, "sigma_mode", sigma_mode)
        alpha = getattr(cfg, "alpha", alpha)
        beta = getattr(cfg, "beta", beta)
        w_tournament = getattr(cfg, "w_tournament", w_tournament)
        w_blade = getattr(cfg, "w_blade", w_blade)
        uwo_lambda = getattr(cfg, "uwo_lambda", uwo_lambda)
        elo_temperature = getattr(cfg, "elo_temperature", elo_temperature)
        elo_rounds = getattr(cfg, "elo_rounds", elo_rounds)
        probabilistic = getattr(cfg, "probabilistic", probabilistic)
        hard_draw = getattr(cfg, "hard_draw", hard_draw)

    if prefix_ids.ndim == 1:
        prefix_ids = prefix_ids.unsqueeze(0)

    # Normalize weights across active blades
    total_weight = sum(weights.get(name, 0.0) for name in blades)
    if total_weight <= 1e-8:
        norm_weights = {name: 1.0 / len(blades) for name in blades}
    else:
        norm_weights = {name: weights.get(name, 0.0) / total_weight for name in blades}

    mu_composite = None
    sigma_composite = None

    for name, blade in blades.items():
        w_k = norm_weights[name]
        if w_k <= 1e-8:
            continue

        mu_k, sigma_k = estimate_mu_sigma(
            prefix_ids=prefix_ids,
            step_token_ids_list=candidate_step_ids,
            blade=blade,
            sigma_mode=sigma_mode,
            draft_logprobs=draft_logprobs,
            beta=beta,
        )

        if mu_composite is None:
            mu_composite = w_k * mu_k
            sigma_composite = w_k * sigma_k
        else:
            mu_composite = mu_composite + w_k * mu_k
            sigma_composite = sigma_composite + w_k * sigma_k

    if mu_composite is None:
        # Fallback if no blades had positive weight
        n = len(candidate_step_ids)
        mu_composite = torch.zeros(n, device=prefix_ids.device)
        sigma_composite = torch.zeros(n, device=prefix_ids.device)

    champion_idx = elo_bracket(
        target_scores=draft_logprobs,
        blade_scores=mu_composite,
        alpha=alpha,
        normalize=True,
        temperature=elo_temperature,
        rounds=elo_rounds,
        beta=beta,
        sigmas=sigma_composite if sigma_mode != "none" else None,
        hard_draw=hard_draw,
        w_tournament=w_tournament,
        w_blade=w_blade,
        uwo_lambda=uwo_lambda,
        probabilistic=probabilistic,
    )

    return champion_idx


class MultiBladeModeBGenerator(EloSwissGenerator):
    """Multi-Blade Swiss-Knife Mode-B Generator (Unconditional Acceptance Logit Mixing).

    Evaluates dual or multi-blade alignment rewards (e.g. Helpfulness + Harmlessness),
    runs a Thurstonian Case-V Elo tournament on composite scores, synthesizes
    Z-normalized champion logits, and commits candidate steps unconditionally.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model,
        drafter_tokenizer,
        verifier_model,
        verifier_tokenizer,
        helpfulness_blade=None,
        harmlessness_blade=None,
        blade_dict: Optional[Dict[str, Any]] = None,
    ):
        primary_blade = helpfulness_blade if helpfulness_blade is not None else (
            next(iter(blade_dict.values())) if blade_dict else None
        )
        super().__init__(
            cfg=cfg,
            drafter_model=drafter_model,
            drafter_tokenizer=drafter_tokenizer,
            verifier_model=verifier_model,
            verifier_tokenizer=verifier_tokenizer,
            blade_model=primary_blade.blade_model if primary_blade else None,
            blade=primary_blade,
        )

        self.helpfulness_blade = helpfulness_blade
        self.harmlessness_blade = harmlessness_blade
        self.blades = blade_dict or {}

        if helpfulness_blade is not None and "helpfulness" not in self.blades:
            self.blades["helpfulness"] = helpfulness_blade
        if harmlessness_blade is not None and "harmlessness" not in self.blades:
            self.blades["harmlessness"] = harmlessness_blade

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        gamma_help: float = 0.5,
        blade_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        """Generates text using Multi-Blade Mode B Logit Mixing with unconditional acceptance.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        max_new_tokens : int, optional
            Maximum tokens to generate.
        gamma_help : float
            Helpfulness Pareto weight in [0, 1] when using dual blades (helpfulness + harmlessness).
        blade_weights : dict of {str: float}, optional
            Custom blade weight mapping for multi-blade mode.
        verbose : bool
            Logging verbosity.
        return_stats : bool
            If True, returns (output_text, stats).

        Returns
        -------
        str or (str, EloSwissStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        beta = self.cfg.beta
        stats = EloSwissStats()
        t_start = time.perf_counter()

        prefix_text = prompt
        generated_tokens: List[int] = []

        initial_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_prefix_ids = initial_encoded["input_ids"].squeeze(0).tolist()

        while len(generated_tokens) < max_tokens:
            stats.total_steps += 1

            encoded = self.verifier_tokenizer(
                prefix_text, return_tensors="pt", padding=False, truncation=True
            )
            prefix_ids_verifier = encoded["input_ids"].squeeze(0).to(self.verifier_device)
            prefix_ids_drafter = prefix_ids_verifier.to(self.drafter_device)

            # Step 1: Draft candidate reasoning step generation
            draft_step_ids_list, step_texts = self._sample_reasoning_steps(
                self.drafter_model,
                self.drafter_tokenizer,
                prefix_ids_drafter.unsqueeze(0),
                n,
                self.drafter_device,
            )
            stats.total_candidates_scored += n

            non_empty = [
                (ids, txt)
                for ids, txt in zip(draft_step_ids_list, step_texts)
                if len(ids) > 0
            ]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            draft_step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]

            # Compute draft log-probabilities
            draft_logprobs_list = compute_logprobs_batched(
                self.drafter_model, prefix_ids_drafter, draft_step_ids_list
            )
            verifier_step_ids_list = [ids.to(self.verifier_device) for ids in draft_step_ids_list]

            if not draft_logprobs_list:
                break

            draft_logprobs = torch.tensor(
                draft_logprobs_list, dtype=torch.float, device=self.verifier_device
            )

            # Step 2: Multi-Blade Scoring & Logit Synthesis
            if blade_weights is not None or len(self.blades) > 2:
                weights = blade_weights or {"helpfulness": gamma_help, "harmlessness": 1.0 - gamma_help}
                selected_idx = compute_multi_blade_mode_b_step(
                    prefix_ids=prefix_ids_verifier.unsqueeze(0),
                    candidate_step_ids=verifier_step_ids_list,
                    draft_logprobs=draft_logprobs,
                    blades=self.blades,
                    weights=weights,
                    beta=beta,
                    cfg=self.cfg,
                )
            else:
                help_blade = self.helpfulness_blade or self.blades.get("helpfulness")
                harm_blade = self.harmlessness_blade or self.blades.get("harmlessness")
                selected_idx = compute_dual_blade_mode_b_step(
                    prefix_ids=prefix_ids_verifier.unsqueeze(0),
                    candidate_step_ids=verifier_step_ids_list,
                    draft_logprobs=draft_logprobs,
                    helpfulness_blade=help_blade,
                    harmlessness_blade=harm_blade,
                    gamma_help=gamma_help,
                    beta=beta,
                    cfg=self.cfg,
                )

            # Step 3: Unconditional Acceptance
            stats.accepted_steps += 1
            winner_text = step_texts[selected_idx]
            winner_verifier_step_ids = verifier_step_ids_list[selected_idx]
            winner_draft_lp = draft_logprobs_list[selected_idx]

            # Track threshold calibrator diagnostics
            winner_target_lp = compute_logprob(
                self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids
            )
            kl_term = (1.0 / beta) * (winner_target_lp - winner_draft_lp)
            self.threshold_calibrator.update(0.0, kl_term)

            if verbose:
                logger.info(
                    "Step %d (Multi-Blade Mode B) accepted champion: '%s' (kl=%.4f)",
                    stats.total_steps,
                    winner_text.strip()[:60],
                    kl_term,
                )

            # Commit step tokens
            prefix_text += winner_text
            step_tokens_list = winner_verifier_step_ids.tolist()
            remaining = max_tokens - len(generated_tokens)
            step_tokens_list = step_tokens_list[:remaining]

            eos_hit = False
            clean_tokens = []
            for tok in step_tokens_list:
                if tok == self.verifier_tokenizer.eos_token_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_tokens.extend(clean_tokens)
            stats.total_tokens += len(clean_tokens)

            if eos_hit:
                break

        stats.total_time_s = time.perf_counter() - t_start
        all_ids = initial_prefix_ids + generated_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        return (output_text, stats) if return_stats else output_text
