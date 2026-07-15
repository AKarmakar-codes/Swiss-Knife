"""
Swiss Knife — GSI Strategy 3 (Mode B): Swiss-System Matches → Points Table → Softmax (No Fallback)
=================================================================================================

Implements GSI Swiss-System tournament selection in Mode B (without the verifier fallback loop).
This means candidate steps are generated and a champion is selected via the Swiss tournament,
and the winner is accepted unconditionally without any threshold verification or resampling.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from .config import SwissKnifeConfig
from .swiss import SwissGenerator, SwissStats, swiss_system_points_table
from .sigma_estimator import estimate_mu_sigma
from evaluation.retokenisation_llama_to_qwen import compute_logprob

logger = logging.getLogger(__name__)


class SwissModeBGenerator(SwissGenerator):
    """GSI Swiss-System Tournament Generator running in Mode B (unconditional accept)."""

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        use_tilted_selection: Optional[bool] = None,
    ):
        """Run Swiss-system → points → softmax selection under Mode B (no fallback)."""
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        swiss_rounds = self.cfg.swiss_rounds
        active_use_tilted = use_tilted_selection if use_tilted_selection is not None else getattr(self.cfg, "use_tilted_selection", False)

        prefix_text = prompt

        generated_tokens: List[int] = []
        stats = SwissStats()
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
            if active_use_tilted or self.cfg.sigma_mode == "log_ratio_proxy":
                verifier_logprobs_list = [
                    compute_logprob(self.verifier_model, prefix_ids_verifier, step_ids)
                    for step_ids in verifier_step_ids_list
                ]
                verifier_logprobs = torch.tensor(verifier_logprobs_list, dtype=torch.float, device=self.verifier_device)
            else:
                verifier_logprobs = None

            # Estimate mu and sigma
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

            if active_use_tilted:
                tilted_rewards_all = blade_rewards + (1.0 / beta) * (verifier_logprobs - draft_logprobs)
            else:
                tilted_rewards_all = None

            # ── Step 2: Swiss-system tournament → points table ──────────
            points, n_rounds, n_matches = swiss_system_points_table(
                draft_logprobs, blade_rewards, alpha,
                rounds=swiss_rounds if swiss_rounds > 0 else 0,
                tilted_rewards=tilted_rewards_all,
                sigmas=sigma if self.cfg.sigma_mode != "none" else None,
                hard_draw=self.cfg.hard_draw,
            )
            stats.total_swiss_rounds += n_rounds
            stats.total_matches += n_matches
            stats.points_tables.append(points)

            # ── Step 3: Softmax over combined tournament and UWO score selection ────────────
            pts_tensor = torch.tensor(points, dtype=torch.float, device=self.verifier_device)
            w_tournament = getattr(self.cfg, "w_tournament", 1.0)
            w_blade = getattr(self.cfg, "w_blade", 1.0)
            uwo_lambda = getattr(self.cfg, "uwo_lambda", 0.5)

            logits = w_tournament * pts_tensor + w_blade * (blade_rewards - uwo_lambda * sigma)
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)
            selected_idx = int(torch.multinomial(probs, num_samples=1).item())

            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_verifier_step_ids = verifier_step_ids_list[selected_idx]

            # ── Step 4: Mode B: Unconditional Acceptance ───────────────────
            stats.accepted_steps += 1
            winner_text = step_texts[selected_idx]

            # Update running threshold buffers with accepted step
            if active_use_tilted:
                winner_target_lp = verifier_logprobs[selected_idx].item()
            else:
                winner_target_lp = compute_logprob(self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids)
            kl_term = (1.0 / beta) * (winner_target_lp - winner_draft_lp)
            self.threshold_calibrator.update(selected_reward, kl_term)

            if verbose:
                logger.info(
                    "Step %d (Mode B) accepted: '%s' (reward: %.4f, kl: %.4f)",
                    stats.total_steps, winner_text.strip(), selected_reward, kl_term
                )

            # Append winner step to running output
            prefix_text += winner_text
            step_tokens_list = winner_verifier_step_ids.tolist()
            generated_tokens.extend(step_tokens_list)
            stats.total_tokens += len(step_tokens_list)

            # Termination checks
            if len(step_tokens_list) == 0:
                logger.info("Empty step generated. Stopping.")
                break

            if self.verifier_tokenizer.eos_token_id in step_tokens_list:
                logger.info("EOS token generated. Stopping.")
                break

        stats.total_time_s = time.perf_counter() - t_start
        return (prefix_text, stats) if return_stats else prefix_text
