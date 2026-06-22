"""
Swiss Knife — GSI Strategy 2: Pairwise Bradley-Terry Selection
================================================================

Implements a pairwise comparison-based selection over GSI reasoning step
candidates, using the Bradley-Terry model:

    P(A wins) = 1 / (1 + exp(-MATCH(A, B) / τ))

where MATCH(A, B) uses the Swiss Knife match function:

    MATCH(A, B) = α · [log π_draft(A) - log π_draft(B)]
               + (1-α) · [r_blade(A) - r_blade(B)]

and τ is a temperature parameter controlling selection sharpness.

At each reasoning step:
    1. Sample n candidate reasoning steps from the LLaMA drafter model.
    2. Score each step with blade reward (Qwen) AND draft log-probability (LLaMA).
    3. Run all n(n-1)/2 pairwise comparisons using Bradley-Terry.
    4. Accumulate win probabilities into a score vector.
    5. Select winner proportional to cumulative win probability.
    6. Compute verifier log-probability and tilted reward only for the winner.
    7. Apply rejection threshold u on the winner's tilted reward. If rejected,
       fall back to resampling from Qwen verifier model.
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

# Import retokenisation utilities from evaluation
from evaluation.benchmark_gsi_harmless import compute_logprob, retokenize_step

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GSIPairwiseStats:
    """Statistics from one GSI pairwise generation run."""

    total_steps: int = 0
    total_tokens: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    total_candidates_scored: int = 0
    total_pairwise_comparisons: int = 0
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
            "strategy": "gsi_pairwise",
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_candidates_scored": self.total_candidates_scored,
            "total_pairwise_comparisons": self.total_pairwise_comparisons,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time_s": round(self.total_time_s, 3),
            "mean_reward": round(sum(self.step_rewards) / max(len(self.step_rewards), 1), 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Bradley-Terry selection
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_bradley_terry_select(
    draft_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    tau: float,
) -> Tuple[int, int]:
    """Run all-pairs Bradley-Terry comparison and select a winner.

    For every pair (i, j), compute:
        MATCH(i, j) = α·(draft_i - draft_j) + (1-α)·(blade_i - blade_j)
        P(i beats j) = σ(MATCH(i, j) / τ)

    Each candidate accumulates the sum of its win probabilities across
    all pairwise comparisons. The winner is sampled proportional to these
    cumulative win probabilities.

    Parameters
    ----------
    draft_scores : torch.Tensor
        Shape ``[n]``. log π_draft(step_i | prefix).
    blade_scores : torch.Tensor
        Shape ``[n]``. r_blade(step_i) = β·log(π_blade/π_ref).
    alpha : float
        Mixing coefficient α ∈ [0, 1].
    tau : float
        Temperature τ > 0. Lower τ → sharper selection.

    Returns
    -------
    selected_idx : int
        Index of the selected candidate.
    n_comparisons : int
        Number of pairwise comparisons made (n*(n-1)/2).
    """
    n = draft_scores.shape[0]

    # Z-score normalize to put draft and blade on comparable scales
    def _znorm(t: torch.Tensor) -> torch.Tensor:
        if t.std() < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / (t.std() + 1e-6)

    draft_normed = _znorm(draft_scores.float())
    blade_normed = _znorm(blade_scores.float())

    # Compute cumulative win probabilities
    cum_win_prob = torch.zeros(n, device=draft_scores.device)
    n_comparisons = 0

    for i in range(n):
        for j in range(i + 1, n):
            # MATCH(i, j) = α·Δdraft + (1-α)·Δblade
            delta_draft = draft_normed[i] - draft_normed[j]
            delta_blade = blade_normed[i] - blade_normed[j]
            match_score = alpha * delta_draft + (1.0 - alpha) * delta_blade

            # P(i beats j) = σ(MATCH / τ)
            p_i_wins = torch.sigmoid(match_score / tau)

            cum_win_prob[i] += p_i_wins
            cum_win_prob[j] += (1.0 - p_i_wins)
            n_comparisons += 1

    # Sample proportional to cumulative win probability
    # Add small epsilon to prevent zero-probability candidates
    probs = cum_win_prob + 1e-8
    probs = probs / probs.sum()
    selected = int(torch.multinomial(probs, num_samples=1).item())

    return selected, n_comparisons


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class GSIPairwiseGenerator:
    """GSI Strategy 2: Pairwise Bradley-Terry selection over reasoning steps.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline configuration.
    drafter_model : PreTrainedModel
        The draft model (e.g. LLaMA 3.2 3B).
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
            "GSIPairwiseGenerator initialized: n=%d, α=%.2f, β=%.3f, τ=%.3f, "
            "threshold=%.3f",
            cfg.gsi_n, cfg.alpha, cfg.beta, cfg.gsi_tau, cfg.gsi_threshold,
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
        """Sample n reasoning steps from a model.

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizer
        prefix_ids : torch.Tensor
            Shape ``[1, prefix_len]``.
        n : int
            Number of candidate steps.
        device : torch.device

        Returns
        -------
        step_ids_list : list of torch.Tensor
        step_texts : list of str
        """
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
        """Run GSI Strategy 2: Pairwise Bradley-Terry selection.

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
        str | (str, GSIPairwiseStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        tau = self.cfg.gsi_tau
        threshold = self.cfg.gsi_threshold

        llama_prefix_text = prompt
        qwen_prefix_text = prompt

        generated_qwen_tokens: List[int] = []
        stats = GSIPairwiseStats()
        t_start = time.perf_counter()

        while len(generated_qwen_tokens) < max_tokens:
            stats.total_steps += 1

            # Prepare tokenized prefixes for both tokenizers
            llama_encoded = self.drafter_tokenizer(
                llama_prefix_text, return_tensors="pt", padding=False, truncation=True
            )
            llama_prefix_ids = llama_encoded["input_ids"].squeeze(0).to(self.drafter_device)

            qwen_encoded = self.verifier_tokenizer(
                qwen_prefix_text, return_tensors="pt", padding=False, truncation=True
            )
            qwen_prefix_ids = qwen_encoded["input_ids"].squeeze(0).to(self.verifier_device)

            # ── Step 1: Sample n reasoning steps from LLaMA ────────────────
            llama_step_ids_list, step_texts = self._sample_reasoning_steps(
                self.drafter_model, self.drafter_tokenizer, llama_prefix_ids.unsqueeze(0), n, self.drafter_device
            )
            stats.total_candidates_scored += n

            non_empty = [(ids, txt) for ids, txt in zip(llama_step_ids_list, step_texts) if len(ids) > 0]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            llama_step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]
            n_actual = len(step_texts)

            # Compute LLaMA logprobs & retokenize to Qwen for DPO blade scoring
            draft_logprobs_list = []
            qwen_step_ids_list = []
            for i in range(n_actual):
                step_text = step_texts[i]
                llama_step_ids = llama_step_ids_list[i]

                # Compute LLaMA log probability on exact generated IDs
                llama_lp = compute_logprob(self.drafter_model, llama_prefix_ids, llama_step_ids)
                draft_logprobs_list.append(llama_lp)

                # Qwen retokenization
                qwen_step_ids = retokenize_step(
                    self.verifier_tokenizer, qwen_prefix_text, step_text, qwen_prefix_ids, self.verifier_device
                )
                qwen_step_ids_list.append(qwen_step_ids)

            if not draft_logprobs_list:
                logger.info("All candidate steps empty. Stopping.")
                break

            draft_logprobs = torch.tensor(draft_logprobs_list, dtype=torch.float, device=self.verifier_device)

            # Compute blade rewards for all candidates on Qwen-tokenized IDs
            blade_rewards = self.blade.score_reasoning_steps(qwen_prefix_ids.unsqueeze(0), qwen_step_ids_list)

            # ── Step 2: Pairwise Bradley-Terry selection ────────────────
            selected_idx, n_comps = pairwise_bradley_terry_select(
                draft_logprobs, blade_rewards, alpha, tau,
            )
            stats.total_pairwise_comparisons += n_comps

            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_qwen_step_ids = qwen_step_ids_list[selected_idx]

            # ── Step 3: Compute tilted reward for the winner ────────────────
            winner_target_lp = compute_logprob(self.verifier_model, qwen_prefix_ids, winner_qwen_step_ids)
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
                    self.verifier_model, self.verifier_tokenizer, qwen_prefix_ids.unsqueeze(0), n, self.verifier_device
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
                    qwen_prefix_ids.unsqueeze(0), resample_ids_list_clean,
                )
                resample_draft = self.blade.compute_step_draft_logprobs(
                    qwen_prefix_ids.unsqueeze(0), resample_ids_list_clean,
                )
                resample_idx, n_comps2 = pairwise_bradley_terry_select(
                    resample_draft, resample_blade, alpha, tau,
                )
                stats.total_pairwise_comparisons += n_comps2
                selected_reward = resample_blade[resample_idx].item()
                selected_tilted_reward = selected_reward  # no log ratio term on fallback
                winner_qwen_step_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

            stats.step_rewards.append(selected_tilted_reward)

            # ── Step 5: Commit ──────────────────────────────────────────
            winner_tokens = winner_qwen_step_ids.tolist()
            remaining = max_tokens - len(generated_qwen_tokens)
            winner_tokens = winner_tokens[:remaining]

            eos_hit = False
            clean_tokens = []
            for tok in winner_tokens:
                if tok == self.verifier_tokenizer.eos_token_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_qwen_tokens.extend(clean_tokens)
            stats.total_tokens += len(clean_tokens)

            # Update prefixes for next iteration
            llama_prefix_text = llama_prefix_text + winner_text
            qwen_prefix_text = qwen_prefix_text + winner_text

            if verbose:
                logger.info(
                    "Step %d | tilted_r=%.4f | tokens=%d | pairwise_comps=%d | text='%s'",
                    stats.total_steps, selected_tilted_reward,
                    len(clean_tokens), n_comps,
                    winner_text[:80],
                )

            if eos_hit:
                logger.info("EOS encountered. Stopping.")
                break

        # ── Finalize ─────────────────────────────────────────────────────
        stats.total_time_s = time.perf_counter() - t_start

        all_ids = qwen_encoded["input_ids"].squeeze(0).tolist()[:qwen_prefix_ids.shape[0]] + generated_qwen_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "GSI-Pairwise complete | %d steps | %d tokens | %.2fs | "
                "acceptance=%.1f%% | %d total comparisons | %.2f tok/s",
                stats.total_steps, stats.total_tokens, stats.total_time_s,
                100 * stats.acceptance_rate, stats.total_pairwise_comparisons,
                stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text

    # ── Blade hot-swap support ───────────────────────────────────────────

    def swap_blade(self, blade_name: str, blade_rack) -> "ReconfigurationProfile":
        """Hot-swap the active alignment blade."""
        new_blade, profile = blade_rack.swap(blade_name)
        self.blade_model = new_blade.blade_model
        self.blade = new_blade
        return profile
