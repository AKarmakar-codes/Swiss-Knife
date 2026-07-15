"""
Swiss Knife — GSI Strategy 4 (Mode B): Elo Tournament Selection, Unconditional Acceptance
==========================================================================================

This module implements step-level GSI generation using a simulated Elo-rating tournament
to select the best candidate reasoning step from the Drafter model.  Unlike Mode A
(:mod:`elo_swiss`), there is **no verifier threshold check and no fallback resampling**.
The tournament winner is accepted unconditionally every step.

For Mode A (with verifier fallback), see :mod:`elo_swiss`.

─────────────────────────────────────────────────────────────────────────────
Why Mode B?
─────────────────────────────────────────────────────────────────────────────

In Mode A the acceptance gate (r_tilted ≥ u) acts as a hard quality floor that
triggers expensive Verifier resampling when the Drafter's best step looks weak.
Mode B removes this gate entirely:

  • FASTER  — no secondary Verifier forward passes; latency is purely the Drafter
    + Blade scoring cost.
  • SIMPLER — the only quality signal is the tournament selection itself.

In Mode B, candidate quality is controlled exclusively by the **tournament match
mechanics** and the **champion selection logits**.  High-uncertainty or low-reward
candidates are naturally penalised by the softmax selection step (via uwo_lambda).
This makes Mode B a clean ablation condition: it tests whether the tournament alone
is a sufficient selector without any hard acceptance gate.

─────────────────────────────────────────────────────────────────────────────
Pipeline — one decoding step
─────────────────────────────────────────────────────────────────────────────

  1. DRAFT CANDIDATE GENERATION
     • Sample n candidate reasoning steps from the fast Drafter model.
     • Compute Drafter log-probabilities log π_draft(step_i) for each candidate.
     • If ``use_tilted_elo=True`` or ``sigma_mode='log_ratio_proxy'``, also compute
       Verifier log-probabilities log π_verifier(step_i).
     • Estimate DPO-Blade rewards μ_i (and optionally σ_i) via ``estimate_mu_sigma``:
         - ``sigma_mode='none'``            → σ_i = 0 for all i
         - ``sigma_mode='log_ratio_proxy'`` → σ_i ≈ |r_blade − (1/β)(log π_V − log π_D)|
         - ``sigma_mode='mc_dropout'``      → σ_i = std over K stochastic forward passes
     • If ``use_tilted_elo=True``, compute tilted rewards:
           tilted_i = μ_i + (1/β)(log π_verifier − log π_draft)

  2. ELO TOURNAMENT MATCH MECHANICS
     Match outcome between candidates A and B is decided by one of two methods:

     (a) PROBABILISTIC (Thurstonian Case-V) — enabled by ``--probabilistic`` flag
         or automatically when ``sigma_mode`` produces non-zero σ values:

             P(A beats B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B² + ε))

         where Φ is the standard normal CDF.  Crucially, the LOWER-SCORING
         candidate can still WIN with probability 1 − P.  When σ is large
         (uncertain candidates), the win probability is pulled toward 0.5,
         making upset wins more likely — this is the justification for using a
         tournament rather than just sorting by μ.

     (b) BRADLEY-TERRY sigmoid — default when ``--probabilistic`` is NOT set and
         ``sigma_mode='none'``:

             P(A beats B) = σ(score_A − score_B)

         Higher score always means P > 0.5.  With many rounds, this degenerates
         into a soft sorting mechanism and offers little stochasticity.

     Elo ratings are updated after each match:
         R_new = R + K · (actual_outcome − expected_outcome)
     where actual_outcome = P (soft) or Bernoulli(P) (hard, via ``--hard-draw``).

  3. CHAMPION SELECTION (softmax draw, not hard argmax)
     After all rounds, the champion is chosen via a softmax draw over combined logits:

         logit_i = w_tournament · (R_i − 1500) / T  +  w_blade · (μ_i − λ · σ_i)

     Parameters:
       • w_tournament  — weight of the Elo tournament rating  (``--w-tournament``)
       • w_blade       — weight of the blade reward term       (``--w-blade``)
       • λ (uwo_lambda)— uncertainty SELECTION penalty         (``--uwo-lambda``)

     ┌─────────────────────────────────────────────────────────────────────┐
     │  About uwo_lambda in Mode B                                         │
     │                                                                     │
     │  In Mode A, the system has TWO places where uncertainty matters:    │
     │    (i)  champion selection  (μ − λσ in softmax logits)              │
     │    (ii) acceptance gate     (r_tilted ≥ u hard threshold)           │
     │                                                                     │
     │  In Mode B there is NO acceptance gate.  uwo_lambda controls ONLY  │
     │  place (i): it down-weights high-σ candidates at selection time so  │
     │  that the softmax assigns lower probability to uncertain steps.     │
     │                                                                     │
     │  Set uwo_lambda=0.0 to ignore uncertainty in selection entirely.    │
     │  Set uwo_lambda=1.0–2.0 to be very conservative: uncertain steps   │
     │  need much higher μ to be selected.                                 │
     └─────────────────────────────────────────────────────────────────────┘

  4. UNCONDITIONAL ACCEPTANCE
     • The tournament champion is accepted and its tokens appended.
     • The running threshold calibrator is updated for diagnostic tracking
       (so you can monitor what the threshold *would have been* in Mode A).
     • No fallback resampling. No rejection.

─────────────────────────────────────────────────────────────────────────────
Key CLI flags
─────────────────────────────────────────────────────────────────────────────
  --probabilistic       Force Thurstonian CDF for all matches (recommended).
  --sigma-mode          'none' | 'log_ratio_proxy' | 'mc_dropout'
  --hard-draw           Use hard Bernoulli flip instead of soft-P Elo update.
  --w-tournament        Weight of tournament rating in champion selection logits.
  --w-blade             Weight of blade UWO term in champion selection logits.
  --uwo-lambda          Uncertainty penalty λ in μ − λσ (SELECTION only, no gating).
  --elo-temperature     Temperature T for the softmax over final Elo ratings.
  --elo-rounds          Number of Elo rating rounds (default 6).
"""

import logging
import time
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SwissKnifeConfig
from .elo_swiss import EloSwissGenerator, EloSwissStats
from .elo_system import elo_bracket
from .sigma_estimator import estimate_mu_sigma
from evaluation.retokenisation_llama_to_qwen import compute_logprob

logger = logging.getLogger(__name__)


class EloSwissModeBGenerator(EloSwissGenerator):
    """GSI Elo-Swiss Tournament Generator — Mode B (unconditional acceptance, no fallback).

    Inherits the __init__ and _sample_reasoning_steps from EloSwissGenerator.
    Overrides generate() to remove all threshold checks and verifier fallback.
    """

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        use_tilted_elo: Optional[bool] = None,
    ):
        """Run Elo-Swiss tournament selection in Mode B (unconditional acceptance).

        Each step:
          1. Sample n candidates from the Drafter.
          2. Score them with the DPO Blade (and estimate σ if sigma_mode != 'none').
          3. Run a probabilistic Elo tournament (Thurstonian or Bradley-Terry depending
             on ``--probabilistic``).
          4. Select champion via softmax over combined tournament + UWO logits.
          5. Accept unconditionally — no threshold, no rejection, no fallback.

        Parameters
        ----------
        prompt : str
            Input text to continue from.
        max_new_tokens : int, optional
            Override cfg.max_new_tokens.
        verbose : bool
            Log per-step details via the logger.
        return_stats : bool
            If True, return ``(text, EloSwissStats)``; otherwise return text only.
        use_tilted_elo : bool, optional
            Override cfg.use_tilted_elo.  When True, match scores are computed from
            the tilted reward r_tilted = μ + (1/β)(log π_V − log π_D).

        Returns
        -------
        str | (str, EloSwissStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        elo_rounds = self.cfg.elo_rounds
        elo_temp = self.cfg.elo_temperature
        active_use_tilted_elo = (
            use_tilted_elo if use_tilted_elo is not None
            else getattr(self.cfg, "use_tilted_elo", False)
        )
        is_probabilistic = getattr(self.cfg, "probabilistic", False)
        w_tournament = getattr(self.cfg, "w_tournament", 1.0)
        w_blade = getattr(self.cfg, "w_blade", 1.0)
        uwo_lambda = getattr(self.cfg, "uwo_lambda", 0.5)

        prefix_text = prompt
        generated_tokens: List[int] = []
        stats = EloSwissStats()
        t_start = __import__("time").perf_counter()

        initial_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_prefix_ids = initial_encoded["input_ids"].squeeze(0).tolist()

        while len(generated_tokens) < max_tokens:
            stats.total_steps += 1

            # ── Tokenise prefix ──────────────────────────────────────────────
            encoded = self.verifier_tokenizer(
                prefix_text, return_tensors="pt", padding=False, truncation=True
            )
            prefix_ids_verifier = encoded["input_ids"].squeeze(0).to(self.verifier_device)
            prefix_ids_drafter = prefix_ids_verifier.to(self.drafter_device)

            # ── Step 1: Sample n reasoning steps from Drafter ────────────────
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
            n_actual = len(step_texts)

            # ── Drafter log-probabilities ────────────────────────────────────
            draft_logprobs_list = []
            verifier_step_ids_list = []
            for i in range(n_actual):
                draft_step_ids = draft_step_ids_list[i]
                draft_lp = compute_logprob(
                    self.drafter_model, prefix_ids_drafter, draft_step_ids
                )
                draft_logprobs_list.append(draft_lp)
                verifier_step_ids_list.append(draft_step_ids.to(self.verifier_device))

            if not draft_logprobs_list:
                logger.info("All candidate steps empty after logprob computation. Stopping.")
                break

            draft_logprobs = torch.tensor(
                draft_logprobs_list, dtype=torch.float, device=self.verifier_device
            )

            # ── Verifier log-probabilities (if needed) ───────────────────────
            # Needed for: tilted reward computation OR log_ratio_proxy sigma estimation.
            if active_use_tilted_elo or self.cfg.sigma_mode == "log_ratio_proxy":
                verifier_logprobs_list = [
                    compute_logprob(
                        self.verifier_model, prefix_ids_verifier, step_ids
                    )
                    for step_ids in verifier_step_ids_list
                ]
                verifier_logprobs = torch.tensor(
                    verifier_logprobs_list, dtype=torch.float, device=self.verifier_device
                )
            else:
                verifier_logprobs = None

            # ── Uncertainty estimation (μ, σ) ────────────────────────────────
            mu, sigma = estimate_mu_sigma(
                prefix_ids=prefix_ids_verifier.unsqueeze(0),
                step_token_ids_list=verifier_step_ids_list,
                blade=self.blade,
                sigma_mode=self.cfg.sigma_mode,
                K=self.cfg.sigma_mc_samples,
                dropout_p=self.cfg.sigma_dropout_p,
                draft_logprobs=draft_logprobs,
                verifier_logprobs=verifier_logprobs,
                beta=beta,
            )
            blade_rewards = mu

            # ── Tilted rewards (optional) ────────────────────────────────────
            if active_use_tilted_elo:
                tilted_rewards = blade_rewards + (1.0 / beta) * (
                    verifier_logprobs - draft_logprobs
                )
            else:
                tilted_rewards = None

            # ── Step 2: Elo tournament ───────────────────────────────────────
            # is_probabilistic=True  → Thurstonian CDF, lower scorer CAN win
            # is_probabilistic=False → Bradley-Terry sigmoid (default soft sort)
            selected_idx = elo_bracket(
                draft_logprobs,
                blade_rewards,
                alpha,
                normalize=self.cfg.normalize_scores,
                temperature=elo_temp,
                rounds=elo_rounds,
                beta=beta,
                tilted_rewards=tilted_rewards,
                sigmas=sigma if self.cfg.sigma_mode != "none" else None,
                hard_draw=self.cfg.hard_draw,
                w_tournament=w_tournament,
                w_blade=w_blade,
                uwo_lambda=uwo_lambda,
                probabilistic=is_probabilistic,
            )

            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_verifier_step_ids = verifier_step_ids_list[selected_idx]

            # ── Step 3: Unconditional acceptance (Mode B) ────────────────────
            # No threshold check. No rejection. No fallback.
            stats.accepted_steps += 1
            winner_text = step_texts[selected_idx]

            # Update calibrator diagnostics (tracks what threshold *would be* in Mode A)
            if active_use_tilted_elo:
                winner_target_lp = verifier_logprobs[selected_idx].item()
            else:
                winner_target_lp = compute_logprob(
                    self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids
                )
            kl_term = (1.0 / beta) * (winner_target_lp - winner_draft_lp)
            self.threshold_calibrator.update(selected_reward, kl_term)

            if verbose:
                logger.info(
                    "Step %d (Mode B Elo | probabilistic=%s) accepted: '%s' "
                    "(μ=%.4f, σ=%.4f, kl=%.4f)",
                    stats.total_steps,
                    is_probabilistic,
                    winner_text.strip()[:60],
                    selected_reward,
                    sigma[selected_idx].item(),
                    kl_term,
                )

            # ── Commit step tokens ───────────────────────────────────────────
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
                logger.info("EOS token generated. Stopping.")
                break

        stats.total_time_s = __import__("time").perf_counter() - t_start

        # Decode full output from initial prefix + generated tokens
        all_ids = initial_prefix_ids + generated_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        return (output_text, stats) if return_stats else output_text
