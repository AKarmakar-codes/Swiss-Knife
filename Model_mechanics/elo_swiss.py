"""
Swiss Knife — GSI Strategy 4 (Mode A): Elo Tournament Selection with Verifier Fallback
=======================================================================================

This module implements step-level GSI generation using a simulated Elo-rating tournament
to select the best candidate reasoning step produced by the Drafter model.  If the winner
fails a quality threshold, the system falls back to resampling from the stronger (but
slower) Verifier model.

For Mode B (no fallback), see :mod:`elo_swiss_mode_b`.

─────────────────────────────────────────────────────────────────────────────
Pipeline Overview
─────────────────────────────────────────────────────────────────────────────

For each decoding step (repeated until ``max_new_tokens`` or EOS):

  1. DRAFT CANDIDATE GENERATION
     • Sample n candidate reasoning steps from the fast Drafter model.
     • Compute Drafter log-probabilities log π_draft(step_i) for each candidate.
     • If ``use_tilted_elo=True`` or ``sigma_mode='log_ratio_proxy'``, also compute
       Verifier log-probabilities log π_verifier(step_i).
     • Estimate DPO-Blade rewards μ_i (and optionally σ_i) via ``estimate_mu_sigma``:
         - ``sigma_mode='none'``          → σ_i = 0 for all i
         - ``sigma_mode='log_ratio_proxy'`` → σ_i ≈ |r_blade − (1/β)(log π_V − log π_D)|
         - ``sigma_mode='mc_dropout'``    → σ_i = std over K stochastic forward passes
     • If ``use_tilted_elo=True``, compute tilted rewards:
           tilted_i = μ_i + (1/β)(log π_verifier − log π_draft)

  2. ELO TOURNAMENT MATCH MECHANICS
     Match outcome between candidates A and B is decided by one of two methods:

     (a) PROBABILISTIC (Thurstonian Case-V) — enabled by ``--probabilistic`` flag
         or automatically when ``sigma_mode`` produces non-zero σ values:

             P(A beats B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B² + ε))

         where Φ is the standard normal CDF.  This is a stochastic match: the
         lower-scoring candidate can win with probability 1 − P.  Uncertainty σ
         widens the distribution, making upsets more likely when candidates are
         close or one is highly uncertain.

     (b) BRADLEY-TERRY sigmoid — default when ``--probabilistic`` is NOT set and
         ``sigma_mode='none'``:

             P(A beats B) = σ(score_A − score_B)

         Higher score ⟹ P > 0.5 always.  With many rounds, this degenerates to
         a soft ranking / sorting mechanism with limited true stochasticity.

     Elo ratings are updated after each match:
         R_new = R + K · (actual_outcome − expected_outcome)
     where actual_outcome = P (soft) or Bernoulli(P) (hard, via ``--hard-draw``).

  3. CHAMPION SELECTION (softmax, not hard argmax)
     After all rounds, the champion is chosen via a softmax draw over combined logits:

         logit_i = w_tournament · (R_i − 1500) / T  +  w_blade · (μ_i − λ · σ_i)

     where:
       • w_tournament  — weight of the Elo tournament rating  (``--w-tournament``)
       • w_blade       — weight of the blade reward term       (``--w-blade``)
       • λ (uwo_lambda)— uncertainty *selection* penalty        (``--uwo-lambda``)

     NOTE: λ is NOT a rejection threshold.  It shapes the softmax distribution
     at *selection* time: candidates with high σ (uncertain quality) are
     down-weighted relative to confident, high-reward candidates.
     This is the Uncertainty-Weighted Objective (UWO) from Coste et al. 2023.

  4. THRESHOLD VERIFICATION & VERIFIER FALLBACK
     • Compute the tilted reward of the selected champion:
           r_tilted = μ_winner + (1/β)(log π_verifier − log π_draft)
     • Compare against adaptive or fixed threshold u:
           if r_tilted ≥ u  →  ACCEPT the step
           if r_tilted < u  →  REJECT and resample n steps from the Verifier,
                               run a second Elo tournament, accept unconditionally.

Key CLI flags
─────────────
  --probabilistic       Force Thurstonian CDF for all matches (enables stochastic upsets).
  --sigma-mode          'none' | 'log_ratio_proxy' | 'mc_dropout'
  --hard-draw           Use hard Bernoulli flip instead of soft-P Elo update.
  --w-tournament        Weight of tournament rating in champion selection logits.
  --w-blade             Weight of blade UWO term in champion selection logits.
  --uwo-lambda          Uncertainty penalty λ in μ − λσ (selection only, not rejection).
  --no-fallback         Run in Mode B (unconditional acceptance, use elo_swiss_mode_b).
"""


import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from .config import SwissKnifeConfig
from .blades import DPOBlade
from .elo_system import elo_bracket

# Import utilities from evaluation
from evaluation.retokenisation_llama_to_qwen import compute_logprob

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EloSwissStats:
    """Statistics from one Elo-system generation run."""

    total_steps: int = 0
    total_tokens: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    total_candidates_scored: int = 0
    total_time_s: float = 0.0
    step_rewards: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.accepted_steps / self.total_steps

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s < 1e-6:
            return 0.0
        return self.total_tokens / self.total_time_s

    @property
    def avg_step_tokens(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_tokens / self.total_steps

    def to_dict(self) -> dict:
        return {
            "strategy": "elo_swiss",
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_candidates_scored": self.total_candidates_scored,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "avg_step_tokens": round(self.avg_step_tokens, 2),
            "total_time_s": round(self.total_time_s, 3),
            "mean_reward": round(sum(self.step_rewards) / max(len(self.step_rewards), 1), 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class EloSwissGenerator:
    """GSI Strategy 4: Elo tournament → tilted reward check → verifier fallback.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline configuration.
    drafter_model : PreTrainedModel
        The draft model (e.g. Qwen 2.5 3B).
    drafter_tokenizer : PreTrainedTokenizer
        Tokenizer for the draft model.
    verifier_model : PreTrainedModel
        The verifier model (e.g. Qwen 2.5 7B).
    verifier_tokenizer : PreTrainedTokenizer
        Tokenizer for the verifier model.
    blade_model : PeftModel
        Active DPO blade adapter on the verifier model.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model: PreTrainedModel,
        drafter_tokenizer: PreTrainedTokenizer,
        verifier_model: PreTrainedModel,
        verifier_tokenizer: PreTrainedTokenizer,
        blade_model: PeftModel,
    ):
        self.cfg = cfg
        self.drafter_model = drafter_model
        self.drafter_tokenizer = drafter_tokenizer
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer
        self.blade_model = blade_model
        self.blade = DPOBlade(cfg, verifier_model, blade_model, verifier_tokenizer)

        # Set devices
        self.drafter_device = next(drafter_model.parameters()).device
        self.verifier_device = next(verifier_model.parameters()).device

        # Initialize threshold calibrator
        from .sigma_estimator import RunningPercentileThreshold
        self.threshold_calibrator = RunningPercentileThreshold(
            percentile=cfg.threshold_percentile,
            buffer_size=cfg.threshold_buffer_size,
        )

        logger.info(
            "EloSwissGenerator initialized: n=%d, α=%.2f, β=%.3f, "
            "elo_rounds=%d, elo_temp=%.3f, threshold=%.3f, sigma_mode=%s, probabilistic=%s",
            cfg.gsi_n, cfg.alpha, cfg.beta, cfg.elo_rounds,
            cfg.elo_temperature, cfg.gsi_threshold, cfg.sigma_mode,
            getattr(cfg, 'probabilistic', False),
        )

    # ── Step sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample_reasoning_steps(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prefix_ids: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample n reasoning steps from a model."""
        batch_ids = prefix_ids.expand(n, -1).contiguous()
        batch_mask = torch.ones_like(batch_ids)

        outputs = model.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=self.cfg.gsi_max_step_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

        prefix_len = prefix_ids.shape[1]
        delimiter = self.cfg.gsi_step_delimiter

        step_ids_list = []
        step_texts = []

        for i in range(n):
            new_tokens = outputs[i, prefix_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

            delim_pos = decoded.find(delimiter)
            if delim_pos >= 0:
                step_text = decoded[:delim_pos + len(delimiter)]
            else:
                step_text = decoded

            step_tokens = tokenizer.encode(
                step_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0).to(device)

            eos_positions = (step_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                step_tokens = step_tokens[:eos_positions[0]]
                step_text = tokenizer.decode(step_tokens, skip_special_tokens=True)

            step_ids_list.append(step_tokens)
            step_texts.append(step_text)

        return step_ids_list, step_texts

    # ── Main generation loop ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        use_tilted_elo: Optional[bool] = None,
    ):
        """Run Elo tournament selection over reasoning steps.

        Parameters
        ----------
        prompt : str
            Input prompt.
        max_new_tokens : int, optional
            Override cfg.max_new_tokens.
        verbose : bool
            Log per-step details.
        return_stats : bool
            If True, return (text, stats) tuple.

        Returns
        -------
        str | (str, EloSwissStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        threshold = self.cfg.gsi_threshold
        elo_rounds = self.cfg.elo_rounds
        elo_temp = self.cfg.elo_temperature
        active_use_tilted_elo = use_tilted_elo if use_tilted_elo is not None else getattr(self.cfg, "use_tilted_elo", False)

        prefix_text = prompt

        generated_tokens: List[int] = []
        stats = EloSwissStats()
        t_start = time.perf_counter()

        initial_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_prefix_ids = initial_encoded["input_ids"].squeeze(0).tolist()

        while len(generated_tokens) < max_tokens:
            stats.total_steps += 1

            # Prepare tokenized prefix
            encoded = self.verifier_tokenizer(
                prefix_text, return_tensors="pt", padding=False, truncation=True
            )
            prefix_ids_verifier = encoded["input_ids"].squeeze(0).to(self.verifier_device)
            prefix_ids_drafter = prefix_ids_verifier.to(self.drafter_device)

            # ── Step 1: Sample n reasoning steps from Drafter ────────────────
            draft_step_ids_list, step_texts = self._sample_reasoning_steps(
                self.drafter_model, self.drafter_tokenizer, prefix_ids_drafter.unsqueeze(0), n, self.drafter_device
            )
            stats.total_candidates_scored += n

            non_empty = [(ids, txt) for ids, txt in zip(draft_step_ids_list, step_texts) if len(ids) > 0]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            draft_step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]
            n_actual = len(step_texts)

            # Compute Drafter logprobs
            draft_logprobs_list = []
            verifier_step_ids_list = []
            for i in range(n_actual):
                draft_step_ids = draft_step_ids_list[i]
                draft_lp = compute_logprob(self.drafter_model, prefix_ids_drafter, draft_step_ids)
                draft_logprobs_list.append(draft_lp)
                verifier_step_ids_list.append(draft_step_ids.to(self.verifier_device))

            if not draft_logprobs_list:
                logger.info("All candidate steps empty. Stopping.")
                break

            draft_logprobs = torch.tensor(draft_logprobs_list, dtype=torch.float, device=self.verifier_device)

            # Compute verifier logprobs if needed for tilted selection or log_ratio_proxy
            if active_use_tilted_elo or self.cfg.sigma_mode == "log_ratio_proxy":
                verifier_logprobs_list = [
                    compute_logprob(self.verifier_model, prefix_ids_verifier, step_ids)
                    for step_ids in verifier_step_ids_list
                ]
                verifier_logprobs = torch.tensor(verifier_logprobs_list, dtype=torch.float, device=self.verifier_device)
            else:
                verifier_logprobs = None

            # Estimate mu and sigma
            from .sigma_estimator import estimate_mu_sigma
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

            if active_use_tilted_elo:
                tilted_rewards = blade_rewards + (1.0 / beta) * (verifier_logprobs - draft_logprobs)
            else:
                tilted_rewards = None

            # ── Step 2: Elo tournament to select winner ────────────────
            w_tournament = getattr(self.cfg, "w_tournament", 1.0)
            w_blade = getattr(self.cfg, "w_blade", 1.0)
            uwo_lambda = getattr(self.cfg, "uwo_lambda", 0.5)

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
                probabilistic=getattr(self.cfg, 'probabilistic', False),
            )
            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_verifier_step_ids = verifier_step_ids_list[selected_idx]

            # ── Step 3: Compute tilted reward for the winner ────────────────
            if active_use_tilted_elo:
                winner_target_lp = verifier_logprobs[selected_idx].item()
            else:
                winner_target_lp = compute_logprob(self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids)
            selected_tilted_reward = selected_reward + (1.0 / beta) * (winner_target_lp - winner_draft_lp)

            # ── Step 4: Adaptive threshold calibration & Fallback check ───────────────────────
            if getattr(self.cfg, "adaptive_threshold", False):
                current_threshold = self.threshold_calibrator.get_threshold(threshold)
            else:
                current_threshold = threshold

            if selected_tilted_reward >= current_threshold or not getattr(self.cfg, "with_fallback", True):
                if selected_tilted_reward < current_threshold:
                    logger.debug(
                        "Step %d: Below threshold but fallback is disabled. Accepting anyway.",
                        stats.total_steps
                    )
                stats.accepted_steps += 1
                winner_text = step_texts[selected_idx]

                # Update running threshold buffers with accepted step
                kl_term = (1.0 / beta) * (winner_target_lp - winner_draft_lp)
                self.threshold_calibrator.update(selected_reward, kl_term)
            else:
                stats.rejected_steps += 1
                logger.debug(
                    "Step %d: Rejected (tilted_r=%.4f < threshold=%.4f). Resampling from Qwen...",
                    stats.total_steps, selected_tilted_reward, current_threshold,
                )
                resample_ids_list, resample_texts = self._sample_reasoning_steps(
                    self.verifier_model, self.verifier_tokenizer, prefix_ids_verifier.unsqueeze(0), n, self.verifier_device
                )
                stats.total_candidates_scored += n

                resample_ids_list_clean = []
                resample_texts_clean = []
                for ids, txt in zip(resample_ids_list, resample_texts):
                    if len(ids) > 0:
                        resample_ids_list_clean.append(ids)
                        resample_texts_clean.append(txt)

                if not resample_ids_list_clean:
                    logger.info("Resample produced all empty steps. Stopping.")
                    break

                resample_verifier_lps = [
                    compute_logprob(self.verifier_model, prefix_ids_verifier, step_ids)
                    for step_ids in resample_ids_list_clean
                ]
                resample_verifier_logprobs = torch.tensor(
                    resample_verifier_lps, dtype=torch.float, device=self.verifier_device
                )

                # Estimate mu and sigma for fallback candidates
                resample_mu, resample_sigma = estimate_mu_sigma(
                    prefix_ids=prefix_ids_verifier.unsqueeze(0),
                    step_token_ids_list=resample_ids_list_clean,
                    blade=self.blade,
                    sigma_mode=self.cfg.sigma_mode,
                    K=self.cfg.sigma_mc_samples,
                    dropout_p=self.cfg.sigma_dropout_p,
                    draft_logprobs=resample_verifier_logprobs,
                    verifier_logprobs=resample_verifier_logprobs,
                    beta=beta,
                )
                resample_blade = resample_mu

                resample_idx = elo_bracket(
                    resample_verifier_logprobs,
                    resample_blade,
                    alpha,
                    normalize=self.cfg.normalize_scores,
                    temperature=elo_temp,
                    rounds=elo_rounds,
                    beta=beta,
                    tilted_rewards=resample_blade if active_use_tilted_elo else None,
                    sigmas=resample_sigma if self.cfg.sigma_mode != "none" else None,
                    hard_draw=self.cfg.hard_draw,
                    w_tournament=w_tournament,
                    w_blade=w_blade,
                    uwo_lambda=uwo_lambda,
                    probabilistic=getattr(self.cfg, 'probabilistic', False),
                )
                selected_reward = resample_blade[resample_idx].item()
                selected_tilted_reward = selected_reward  # no log ratio term on fallback
                winner_verifier_step_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

                # Update running threshold buffers with accepted fallback step (kl is 0)
                self.threshold_calibrator.update(selected_reward, 0.0)

            stats.step_rewards.append(selected_tilted_reward)

            # ── Step 5: Commit ──────────────────────────────────────────
            winner_tokens = winner_verifier_step_ids.tolist()
            remaining = max_tokens - len(generated_tokens)
            winner_tokens = winner_tokens[:remaining]

            eos_hit = False
            clean_tokens = []
            for tok in winner_tokens:
                if tok == self.verifier_tokenizer.eos_token_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_tokens.extend(clean_tokens)
            stats.total_tokens += len(clean_tokens)

            # Update prefix for next iteration
            prefix_text = prefix_text + winner_text

            if verbose:
                logger.info(
                    "Step %d | tilted_r=%.4f | tokens=%d | text='%s'",
                    stats.total_steps, selected_tilted_reward,
                    len(clean_tokens), winner_text[:80],
                )

            if eos_hit:
                logger.info("EOS encountered. Stopping.")
                break

        # ── Finalize ─────────────────────────────────────────────────────
        stats.total_time_s = time.perf_counter() - t_start

        all_ids = initial_prefix_ids + generated_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "Elo-Swiss complete | %d steps | %d tokens | %.2fs | "
                "acceptance=%.1f%% | %.2f tok/s",
                stats.total_steps, stats.total_tokens, stats.total_time_s,
                100 * stats.acceptance_rate, stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text
