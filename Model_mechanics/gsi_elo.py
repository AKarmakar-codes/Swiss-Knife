"""
Swiss Knife — GSI Strategy 4: Elo Tournament Selection over Reasoning Steps
=============================================================================

Algorithm:
----------
For each decoding loop iteration (until maximum new tokens or EOS is reached):
1. Draft Candidate Generation:
   Sample n candidate reasoning steps from the drafter model (Qwen 2.5 3B).
2. Tournament Selection:
   Rank and select a champion candidate from the n draft steps using a simulated Elo bracket tournament.
   By default, this runs for 6 rounds using a decaying schedule of K-factors:
     Round 1: K = 40, Round 2: K = 32, Round 3: K = 24, Round 4: K = 16, Round 5: K = 12, Round 6: K = 10.
   Matchups are determined stochastically based on Bradley-Terry pairwise winning probabilities:
     P(A wins over B) = sigmoid((Score_A - Score_B) / T)
   where Score_i = draft_logprobs_i + alpha * r_blade_i, and T is the tournament temperature.
3. Verification (Tilted Reward):
   For the tournament champion step, evaluate its verifier log probability log(pi_verifier) and
   compute its tilted reward:
     r_tilted = r_blade + (1 / beta) * (log(pi_verifier) - log(pi_draft))
4. Rejection Threshold & Fallback:
   Compare the champion's tilted reward with the acceptance threshold u:
     - If r_tilted >= u: Accept the champion step and append it to the generation sequence.
     - If r_tilted < u: Reject the champion step, log it as a fallback event, resample n steps
       directly from the verifier model (Qwen 7B) + blade, run a secondary Elo tournament on these
       resamples, and accept the new tournament winner unconditionally.
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
class GSIEloStats:
    """Statistics from one GSI Elo-system generation run."""

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

    def to_dict(self) -> dict:
        return {
            "strategy": "gsi_elo",
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_candidates_scored": self.total_candidates_scored,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time_s": round(self.total_time_s, 3),
            "mean_reward": round(sum(self.step_rewards) / max(len(self.step_rewards), 1), 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class GSIEloGenerator:
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

        logger.info(
            "GSIEloGenerator initialized: n=%d, α=%.2f, β=%.3f, "
            "elo_rounds=%d, elo_temp=%.3f, threshold=%.3f",
            cfg.gsi_n, cfg.alpha, cfg.beta, cfg.elo_rounds,
            cfg.elo_temperature, cfg.gsi_threshold,
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
    ):
        """Run GSI Strategy 4: Elo tournament selection over reasoning steps.

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
        str | (str, GSIEloStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        threshold = self.cfg.gsi_threshold
        elo_rounds = self.cfg.elo_rounds
        elo_temp = self.cfg.elo_temperature

        prefix_text = prompt

        generated_tokens: List[int] = []
        stats = GSIEloStats()
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

            # Compute blade rewards for all candidates
            blade_rewards = self.blade.score_reasoning_steps(prefix_ids_verifier.unsqueeze(0), verifier_step_ids_list)

            # ── Step 2: Elo tournament to select winner ────────────────
            selected_idx = elo_bracket(
                draft_logprobs,
                blade_rewards,
                alpha,
                normalize=self.cfg.normalize_scores,
                temperature=elo_temp,
                rounds=elo_rounds,
                beta=beta,
            )
            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_verifier_step_ids = verifier_step_ids_list[selected_idx]

            # ── Step 3: Compute tilted reward for the winner ────────────────
            winner_target_lp = compute_logprob(self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids)
            selected_tilted_reward = selected_reward + (1.0 / beta) * (winner_target_lp - winner_draft_lp)

            # ── Step 4: Rejection threshold check ───────────────────────
            if selected_tilted_reward >= threshold:
                stats.accepted_steps += 1
                winner_text = step_texts[selected_idx]
            else:
                stats.rejected_steps += 1
                logger.debug(
                    "Step %d: Rejected (tilted_r=%.4f < threshold=%.4f). Resampling from Qwen...",
                    stats.total_steps, selected_tilted_reward, threshold,
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

                resample_blade = self.blade.score_reasoning_steps(
                    prefix_ids_verifier.unsqueeze(0), resample_ids_list_clean,
                )
                # Fallback steps are sampled from the verifier — use log π_verifier
                # as the fluency signal (first arg to elo_bracket).
                resample_verifier_lps = [
                    compute_logprob(self.verifier_model, prefix_ids_verifier, step_ids)
                    for step_ids in resample_ids_list_clean
                ]
                resample_verifier_logprobs = torch.tensor(
                    resample_verifier_lps, dtype=torch.float, device=self.verifier_device
                )
                resample_idx = elo_bracket(
                    resample_verifier_logprobs,
                    resample_blade,
                    alpha,
                    normalize=self.cfg.normalize_scores,
                    temperature=elo_temp,
                    rounds=elo_rounds,
                    beta=beta,
                )
                selected_reward = resample_blade[resample_idx].item()
                selected_tilted_reward = selected_reward  # no log ratio term on fallback
                winner_verifier_step_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

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
                "GSI-Elo complete | %d steps | %d tokens | %.2fs | "
                "acceptance=%.1f%% | %.2f tok/s",
                stats.total_steps, stats.total_tokens, stats.total_time_s,
                100 * stats.acceptance_rate, stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text
