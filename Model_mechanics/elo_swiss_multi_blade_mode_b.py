import logging
import time
from typing import Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from .blades import DPOBlade
from .config import SwissKnifeConfig
from .elo_swiss import EloSwissGenerator, EloSwissStats
from .elo_system import elo_bracket
from .sigma_estimator import estimate_mu_sigma
from evaluation.logprob_utilities import compute_logprob, compute_logprobs_batched

logger = logging.getLogger(__name__)


class EloSwissMultiBladeModeBGenerator(EloSwissGenerator):
    """GSI Elo-Swiss Tournament Generator — Multi-Blade Mode B.
    
    Implements per-blade Candidate-Batch Normalization (individually normalizing
    both mean reward mu and epistemic uncertainty sigma across candidate batches)
    to safely mix arbitrary K DPO blades. Operates in unconditional acceptance
    mode (Mode B, no verifier threshold fallback).
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model: PreTrainedModel,
        drafter_tokenizer: PreTrainedTokenizer,
        verifier_model: PreTrainedModel,
        verifier_tokenizer: PreTrainedTokenizer,
        blades: Optional[Dict[str, DPOBlade]] = None,
        blade_models: Optional[Dict[str, PeftModel]] = None,
        helpfulness_blade_model: Optional[PeftModel] = None,
        harmlessness_blade_model: Optional[PeftModel] = None,
    ):
        first_blade_model = None
        if blades and len(blades) > 0:
            first_blade_model = next(iter(blades.values())).blade_model
        elif blade_models and len(blade_models) > 0:
            first_blade_model = next(iter(blade_models.values()))
        elif helpfulness_blade_model is not None:
            first_blade_model = helpfulness_blade_model
        else:
            first_blade_model = verifier_model

        super().__init__(
            cfg,
            drafter_model,
            drafter_tokenizer,
            verifier_model,
            verifier_tokenizer,
            first_blade_model,
        )

        self.blades: Dict[str, DPOBlade] = {}
        if blades is not None:
            self.blades = dict(blades)
        elif blade_models is not None:
            for b_name, b_model in blade_models.items():
                self.blades[b_name] = DPOBlade(
                    cfg, verifier_model, b_model, verifier_tokenizer, blade_name=b_name
                )
        elif helpfulness_blade_model is not None and harmlessness_blade_model is not None:
            self.blades["helpfulness"] = DPOBlade(
                cfg, verifier_model, helpfulness_blade_model, verifier_tokenizer, blade_name="helpfulness"
            )
            self.blades["harmlessness"] = DPOBlade(
                cfg, verifier_model, harmlessness_blade_model, verifier_tokenizer, blade_name="harmlessness"
            )

        self.blade = None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        use_tilted_elo: Optional[bool] = None,
        gamma_help: Optional[float] = None,
        blade_coefficients: Optional[Dict[str, float]] = None,
    ):
        """Run Elo-Swiss tournament selection in Multi-Blade Mode B.

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
        gamma_help : float, optional
            Legacy dual blade parameter. If set and blade_coefficients is None,
            sets helpfulness=gamma_help and harmlessness=(1 - gamma_help).
            Preserved in this positional slot for backward compatibility.
        blade_coefficients : Dict[str, float], optional
            Dictionary mapping blade names to their positive weight coefficients.
            Unspecified blades default to 0.0. Takes priority over gamma_help.
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
        sigma_mode = getattr(self.cfg, "sigma_mode", "min_entropy")

        # Determine blade coefficients (defaulting unspecified blades to 0.0)
        coeffs: Dict[str, float] = {}
        if blade_coefficients is not None:
            coeffs = dict(blade_coefficients)
        elif gamma_help is not None:
            coeffs = {"helpfulness": float(gamma_help), "harmlessness": 1.0 - float(gamma_help)}
        elif hasattr(self.cfg, "blade_coefficients") and self.cfg.blade_coefficients:
            coeffs = dict(self.cfg.blade_coefficients)
        else:
            # If no coefficients given, default to equal weights among available blades
            if self.blades:
                equal_w = 1.0 / len(self.blades)
                coeffs = {b_name: equal_w for b_name in self.blades}

        # Filter active blades (weight > 0)
        active_blades = {
            b_name: w for b_name, w in coeffs.items()
            if w > 0.0 and b_name in self.blades
        }

        prefix_text = prompt
        generated_tokens: List[int] = []
        stats = EloSwissStats()
        t_start = time.perf_counter()

        initial_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_prefix_ids = initial_encoded["input_ids"].squeeze(0).tolist()

        cached_prefix_ids_verifier: Optional[torch.Tensor] = None

        while len(generated_tokens) < max_tokens:
            stats.total_steps += 1

            # ── Tokenise prefix ──────────────────────────────────────────────
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

            # ── Verifier log-probabilities (if needed) ───────────────────────
            if active_use_tilted_elo or sigma_mode == "log_ratio_proxy":
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
            else:
                verifier_logprobs = None

            # ── Per-Blade Scoring & Individual Candidate-Batch Normalization ─
            mu_norm_list = []
            sigma_norm_list = []
            active_weights = []

            for b_name, b_weight in active_blades.items():
                blade_obj = self.blades[b_name]
                mu_k, sigma_k = estimate_mu_sigma(
                    prefix_ids=prefix_ids_verifier.unsqueeze(0),
                    step_token_ids_list=verifier_step_ids_list,
                    blade=blade_obj,
                    sigma_mode=sigma_mode,
                    K=self.cfg.sigma_mc_samples,
                    dropout_p=self.cfg.sigma_dropout_p,
                    draft_logprobs=draft_logprobs,
                    verifier_logprobs=verifier_logprobs,
                    beta=beta,
                )

                # Normalize mu and sigma INDIVIDUALLY per blade across candidate batch.
                # mu is zero-mean unit-variance (candidates scored relative to each other).
                # sigma is SCALE-ONLY normalized (no zero-centering): sigma is a non-negative
                # uncertainty quantity where 0 is a meaningful anchor (zero uncertainty).
                # Zero-centering sigma would make low-uncertainty candidates appear negative,
                # destroying the absolute ordering that makes UWO penalization meaningful.
                std_mu = mu_k.std() + 1e-6
                mu_k_norm = (mu_k - mu_k.mean()) / std_mu

                std_sigma = sigma_k.std() + 1e-6
                sigma_k_norm = sigma_k / std_sigma  # scale-only, no mean subtraction

                mu_norm_list.append(mu_k_norm)
                sigma_norm_list.append(sigma_k_norm)
                active_weights.append(b_weight)

            # Convex combination across active blades
            total_weight = sum(active_weights)
            if total_weight > 0 and len(mu_norm_list) > 0:
                norm_weights = [w / total_weight for w in active_weights]
                mu_composite = sum(w * m for w, m in zip(norm_weights, mu_norm_list))
                sigma_composite = sum(w * s for w, s in zip(norm_weights, sigma_norm_list))
            else:
                mu_composite = torch.zeros(n_actual, dtype=torch.float, device=self.verifier_device)
                sigma_composite = torch.zeros(n_actual, dtype=torch.float, device=self.verifier_device)

            blade_rewards = mu_composite

            # ── Tilted rewards (optional) ────────────────────────────────────
            if active_use_tilted_elo and verifier_logprobs is not None:
                tilted_rewards = blade_rewards + (1.0 / beta) * (
                    verifier_logprobs - draft_logprobs
                )
            else:
                tilted_rewards = None

            # ── Step 2: Elo tournament via elo_bracket ───────────────────────
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

            # ── Step 3: Unconditional acceptance (Mode B) ────────────────────
            stats.accepted_steps += 1
            winner_text = step_texts[selected_idx]

            if active_use_tilted_elo and verifier_logprobs is not None:
                winner_target_lp = verifier_logprobs[selected_idx].item()
            else:
                if isinstance(self.verifier_model, PeftModel):
                    with self.verifier_model.disable_adapter():
                        winner_target_lp = compute_logprob(
                            self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids
                        )
                else:
                    winner_target_lp = compute_logprob(
                        self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids
                    )
            kl_term = (1.0 / beta) * (winner_target_lp - winner_draft_lp)
            self.threshold_calibrator.update(selected_reward, kl_term)

            # Populate step_details diagnostic metrics
            if len(blade_rewards) >= 2:
                top2 = blade_rewards.topk(2).values
                delta_mu_val = float((top2[0] - top2[1]).item())
            else:
                delta_mu_val = 0.0

            mean_sigma_val = float(sigma_composite.mean().item()) if len(sigma_composite) > 0 else 0.0
            sigma_spread_val = float(sigma_composite.std().item()) if len(sigma_composite) > 1 else 0.0
            champion_sigma_val = float(sigma_composite[selected_idx].item()) if len(sigma_composite) > 0 else 0.0
            champion_sigma_rank_val = (
                int((sigma_composite.argsort(descending=False) == selected_idx).nonzero(as_tuple=True)[0].item())
                if len(sigma_composite) > 0 else 0
            )

            stats.step_rewards.append(selected_reward)
            stats.step_details.append({
                "step": stats.total_steps,
                "candidate_mus": [round(x, 6) for x in blade_rewards.tolist()],
                "candidate_sigmas": [round(x, 6) for x in sigma_composite.tolist()],
                "mean_mu": round(float(blade_rewards.mean().item()), 6),
                "max_mu": round(float(blade_rewards.max().item()), 6),
                "delta_mu": round(delta_mu_val, 6),
                "mean_sigma": round(mean_sigma_val, 6),
                "sigma_spread": round(sigma_spread_val, 6),
                "selected_idx": int(selected_idx),
                "champion_mu": round(selected_reward, 6),
                "champion_sigma": round(champion_sigma_val, 6),
                "champion_sigma_rank": champion_sigma_rank_val,
                "real_champion_sigma_rank": champion_sigma_rank_val,
                "thurstonian_champion_sigma_rank": champion_sigma_rank_val,
                "real_upset": int(selected_idx) != int(blade_rewards.argmax().item()),
            })

            if verbose:
                logger.info(
                    "Step %d (Multi Mode B | prob=%s | active_blades=%s) accepted: '%s' "
                    "(μ_comp=%.4f, σ_comp=%.4f, kl=%.4f)",
                    stats.total_steps,
                    is_probabilistic,
                    list(active_blades.keys()),
                    winner_text.strip()[:60],
                    selected_reward,
                    sigma_composite[selected_idx].item(),
                    kl_term,
                )

            # ── Commit step tokens ───────────────────────────────────────────
            prefix_text += winner_text
            step_tokens_list = winner_verifier_step_ids.tolist()

            eos_hit = False
            clean_tokens = []
            for tok in step_tokens_list:
                if tok == self.verifier_tokenizer.eos_token_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_tokens.extend(clean_tokens)
            stats.total_tokens += len(clean_tokens)

            clean_ids = winner_verifier_step_ids[:len(clean_tokens)]
            cached_prefix_ids_verifier = torch.cat([prefix_ids_verifier, clean_ids], dim=0)

            if eos_hit:
                logger.info("EOS token generated. Stopping.")
                break

        stats.total_time_s = time.perf_counter() - t_start

        all_ids = initial_prefix_ids + generated_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        return (output_text, stats) if return_stats else output_text

