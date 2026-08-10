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
from evaluation.logprob_utilities import compute_logprob, compute_logprobs_batched
from .blades import DPOBlade

logger = logging.getLogger(__name__)


class EloSwissDualBladeModeBGenerator(EloSwissGenerator):
    """GSI Elo-Swiss Tournament Generator — Dual Blade Mode B.
    
    Implements Candidate-Batch Normalization to safely mix Helpfulness and 
    Harmlessness DPO blades without miscalibration. Operates in unconditional 
    acceptance mode (no verifier fallback).
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model: PreTrainedModel,
        drafter_tokenizer: PreTrainedTokenizer,
        verifier_model: PreTrainedModel,
        verifier_tokenizer: PreTrainedTokenizer,
        helpfulness_blade_model: PeftModel,
        harmlessness_blade_model: PeftModel,
    ):
        # Initialize the base class with the helpfulness model just to satisfy __init__
        super().__init__(
            cfg,
            drafter_model,
            drafter_tokenizer,
            verifier_model,
            verifier_tokenizer,
            helpfulness_blade_model,
        )
        # Instantiate both blades explicitly
        self.helpfulness_blade = DPOBlade(
            cfg, verifier_model, helpfulness_blade_model, verifier_tokenizer, blade_name="helpfulness"
        )
        self.harmlessness_blade = DPOBlade(
            cfg, verifier_model, harmlessness_blade_model, verifier_tokenizer, blade_name="harmlessness"
        )
        
        # Nullify the single self.blade inherited from parent to prevent accidental usage
        self.blade = None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        use_tilted_elo: Optional[bool] = None,
        gamma_help: float = 0.5,
    ):
        """Run Elo-Swiss tournament selection in Dual Blade Mode B.

        Parameters
        ----------
        prompt : str
            Input text to continue from.
        max_new_tokens : int, optional
            Override cfg.max_new_tokens.
        verbose : bool
            Log per-step details via the logger.
        return_stats : bool
            If True, return (text, EloSwissStats); otherwise return text only.
        use_tilted_elo : bool, optional
            Override cfg.use_tilted_elo.
        gamma_help : float
            Pareto mix weight for Helpfulness [0, 1]. Harmlessness gets (1 - gamma_help).
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

        gamma_harm = 1.0 - gamma_help

        prefix_text = prompt
        generated_tokens: List[int] = []
        stats = EloSwissStats()
        t_start = time.perf_counter()

        initial_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_prefix_ids = initial_encoded["input_ids"].squeeze(0).tolist()

        # Cache verifier prefix IDs to avoid full re-tokenization every step.
        cached_prefix_ids_verifier: Optional[torch.Tensor] = None

        while len(generated_tokens) < max_tokens:
            stats.total_steps += 1

            # ── Tokenise prefix ──────────────────────────────────────────────
            # Use cached prefix IDs if available to skip full re-tokenization.
            if cached_prefix_ids_verifier is None:
                encoded = self.verifier_tokenizer(
                    prefix_text, return_tensors="pt", padding=False, truncation=True
                )
                prefix_ids_verifier = encoded["input_ids"].squeeze(0).to(self.verifier_device)
            else:
                prefix_ids_verifier = cached_prefix_ids_verifier
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
            draft_logprobs_list = compute_logprobs_batched(
                self.drafter_model, prefix_ids_drafter, draft_step_ids_list
            )
            verifier_step_ids_list = [ids.to(self.verifier_device) for ids in draft_step_ids_list]

            if not draft_logprobs_list:
                logger.info("All candidate steps empty after logprob computation. Stopping.")
                break

            draft_logprobs = torch.tensor(
                draft_logprobs_list, dtype=torch.float, device=self.verifier_device
            )

            # ── Verifier log-probabilities ───────────────────────
            # Precompute verifier base log-probabilities for DPO reward and uncertainty calculation
            if isinstance(self.verifier_model, PeftModel):
                with self.verifier_model.disable_adapter():
                    verifier_logprobs_list = compute_logprobs_batched(
                        self.verifier_model, prefix_ids_verifier, verifier_step_ids_list
                    )
            else:
                verifier_logprobs_list = compute_logprobs_batched(
                    self.verifier_model, prefix_ids_verifier, verifier_step_ids_list
                )
            verifier_logprobs = torch.tensor(
                verifier_logprobs_list, dtype=torch.float, device=self.verifier_device
            )

            # ── Uncertainty estimation (μ, σ) for both Blades ────────────────
            mu_help, sigma_help = estimate_mu_sigma(
                prefix_ids=prefix_ids_verifier.unsqueeze(0),
                step_token_ids_list=verifier_step_ids_list,
                blade=self.helpfulness_blade,
                sigma_mode=self.cfg.sigma_mode,
                K=self.cfg.sigma_mc_samples,
                dropout_p=self.cfg.sigma_dropout_p,
                draft_logprobs=draft_logprobs,
                verifier_logprobs=verifier_logprobs,
                beta=beta,
            )

            mu_harm, sigma_harm = estimate_mu_sigma(
                prefix_ids=prefix_ids_verifier.unsqueeze(0),
                step_token_ids_list=verifier_step_ids_list,
                blade=self.harmlessness_blade,
                sigma_mode=self.cfg.sigma_mode,
                K=self.cfg.sigma_mc_samples,
                dropout_p=self.cfg.sigma_dropout_p,
                draft_logprobs=draft_logprobs,
                verifier_logprobs=verifier_logprobs,
                beta=beta,
            )

            # ── Candidate-Batch Normalization ─────────────────────────────────
            std_help = mu_help.std() + 1e-6
            std_harm = mu_harm.std() + 1e-6

            mu_help_norm = (mu_help - mu_help.mean()) / std_help
            mu_harm_norm = (mu_harm - mu_harm.mean()) / std_harm
            
            sigma_help_norm = sigma_help / std_help
            sigma_harm_norm = sigma_harm / std_harm

            # Safe Composite Mixing
            mu_composite = gamma_help * mu_help_norm + gamma_harm * mu_harm_norm
            sigma_composite = gamma_help * sigma_help_norm + gamma_harm * sigma_harm_norm
            blade_rewards = mu_composite

            # ── Tilted rewards (optional) ────────────────────────────────────
            if active_use_tilted_elo:
                # We use the raw blended un-normalized logprob ratios for KL
                # or we could keep the log ratio on its native scale
                # However, for tilting the normalized composite, we need to be careful
                # Here we tilt the composite directly with the scaled KL.
                tilted_rewards = blade_rewards + (1.0 / beta) * (
                    verifier_logprobs - draft_logprobs
                )
            else:
                tilted_rewards = None

            # ── Step 2: Elo tournament ───────────────────────────────────────
            selected_idx = elo_bracket(
                draft_logprobs,
                blade_rewards,
                alpha,
                normalize=self.cfg.normalize_scores,
                temperature=elo_temp,
                rounds=elo_rounds,
                beta=beta,
                tilted_rewards=tilted_rewards,
                sigmas=sigma_composite,
                hard_draw=self.cfg.hard_draw,
                w_tournament=w_tournament,
                w_blade=w_blade,
                uwo_lambda=uwo_lambda,
                probabilistic=is_probabilistic,
            )

            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_verifier_step_ids = verifier_step_ids_list[selected_idx]
            winner_target_lp = verifier_logprobs[selected_idx].item()

            # ── Step 3: Unconditional acceptance (Mode B) ────────────────────
            stats.accepted_steps += 1
            winner_text = step_texts[selected_idx]

            # Update calibrator diagnostics (tracks what threshold *would be* in Mode A)
            kl_term = (1.0 / beta) * (winner_target_lp - winner_draft_lp)
            self.threshold_calibrator.update(selected_reward, kl_term)

            if verbose:
                logger.info(
                    "Step %d (Dual Mode B | prob=%s | γ_help=%.2f) accepted: '%s' "
                    "(μ_comp=%.4f, σ_comp=%.4f, kl=%.4f)",
                    stats.total_steps,
                    is_probabilistic,
                    gamma_help,
                    winner_text.strip()[:60],
                    selected_reward,
                    sigma_composite[selected_idx].item(),
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

            # Extend cached prefix IDs with accepted winner tokens (no re-tokenization).
            clean_ids = winner_verifier_step_ids[:len(clean_tokens)]
            cached_prefix_ids_verifier = torch.cat([prefix_ids_verifier, clean_ids], dim=0)

            if eos_hit:
                logger.info("EOS token generated. Stopping.")
                break

        stats.total_time_s = time.perf_counter() - t_start

        # Decode full output from initial prefix + generated tokens
        all_ids = initial_prefix_ids + generated_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        return (output_text, stats) if return_stats else output_text
