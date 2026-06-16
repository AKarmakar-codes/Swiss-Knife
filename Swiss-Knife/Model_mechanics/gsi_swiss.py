"""
Swiss Knife — GSI Strategy 3: Swiss-System Matches → Points Table → Softmax
=============================================================================

Implements the Swiss Knife's novel tournament selection adapted for GSI
step-level inference. This is the strategy that combines the Swiss-system
pairing mechanism (§4.3.1 of swiss_knife_analysis.pdf) with softmax
selection over the final cumulative points.

At each reasoning step:
    1. Sample n candidate reasoning steps from the base model.
    2. Score each step with blade reward AND draft log-probability.
    3. Run R = ceil(log2(n)) rounds of Swiss-system matches:
       - Each round: pair candidates by current cumulative score.
       - Match function: MATCH(A,B) = α·Δdraft + (1-α)·Δblade
       - Winner gets 1 point, loser gets 0. Bye = 0.5 points.
    4. Build the points table: [n] cumulative win scores.
    5. Apply softmax over the points to select the winner:
       winner ~ softmax(β · points)
    6. Apply rejection threshold as in standard GSI.

This combines the noise-robustness of Swiss-system (no early elimination)
with the probabilistic smoothness of softmax selection (no hard argmax),
giving a "best of both worlds" approach that is unique to Swiss Knife.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from .config import SwissKnifeConfig
from .blades import DPOBlade

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GSISwissStats:
    """Statistics from one GSI Swiss-system generation run."""

    total_steps: int = 0
    total_tokens: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    total_candidates_scored: int = 0
    total_swiss_rounds: int = 0
    total_matches: int = 0
    total_time_s: float = 0.0
    step_rewards: List[float] = field(default_factory=list)
    points_tables: List[List[float]] = field(default_factory=list)
    """Points table from each iteration (for analysis)."""

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
            "strategy": "gsi_swiss",
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_candidates_scored": self.total_candidates_scored,
            "total_swiss_rounds": self.total_swiss_rounds,
            "total_matches": self.total_matches,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time_s": round(self.total_time_s, 3),
            "mean_reward": round(sum(self.step_rewards) / max(len(self.step_rewards), 1), 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Swiss-system tournament → points table → softmax selection
# ─────────────────────────────────────────────────────────────────────────────

def swiss_system_points_table(
    draft_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    rounds: int = 0,
) -> Tuple[List[float], int, int]:
    """Run Swiss-system matches and return the cumulative points table.

    This implements §4.3.1 of swiss_knife_analysis.pdf:
    - R rounds of pairwise matches.
    - Each round: pair candidates by current cumulative score (Swiss pairing).
    - Match function: MATCH(A,B) = α·(draft_A - draft_B) + (1-α)·(blade_A - blade_B)
    - Winner gets 1 point, bye = 0.5 points.

    Parameters
    ----------
    draft_scores : torch.Tensor
        Shape ``[n]``. log π_draft(step_i | prefix).
    blade_scores : torch.Tensor
        Shape ``[n]``. r_blade(step_i).
    alpha : float
        Mixing coefficient.
    rounds : int
        Number of Swiss rounds. 0 → auto = ceil(log2(n)).

    Returns
    -------
    points : list of float
        Cumulative points for each candidate.
    total_rounds : int
    total_matches : int
    """
    n = draft_scores.shape[0]

    if rounds == 0:
        rounds = max(1, math.ceil(math.log2(n)))

    # Z-score normalize
    def _znorm(t: torch.Tensor) -> torch.Tensor:
        if t.std() < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / (t.std() + 1e-6)

    draft_normed = _znorm(draft_scores.float())
    blade_normed = _znorm(blade_scores.float())

    # Cumulative points
    cum_points = [0.0] * n
    paired_before = set()
    indices = list(range(n))
    total_matches = 0

    for rnd in range(rounds):
        # ── Build pairings (Swiss-system rule) ─────────────────────────
        # Sort by (cumulative points DESC, original index ASC for tie-break)
        sorted_by_score = sorted(
            indices,
            key=lambda i: (-cum_points[i], i),
        )

        pairs: List[tuple] = []
        unpaired = list(sorted_by_score)

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)

            # Find best partner: prefer no rematch
            best_partner_pos = None
            for pos, b in enumerate(unpaired):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in paired_before:
                    best_partner_pos = pos
                    break

            if best_partner_pos is None:
                # All already paired — allow rematch
                best_partner_pos = 0

            b = unpaired.pop(best_partner_pos)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        # Bye for unpaired candidate (if n is odd)
        if unpaired:
            bye_idx = unpaired[0]
            cum_points[bye_idx] += 0.5
            logger.debug("Swiss Round %d | Bye: c%d", rnd + 1, bye_idx)

        # ── Execute matches ────────────────────────────────────────────
        for a, b in pairs:
            delta_draft = draft_normed[a] - draft_normed[b]
            delta_blade = blade_normed[a] - blade_normed[b]
            match_score = alpha * delta_draft + (1.0 - alpha) * delta_blade

            if match_score > 0:
                winner, loser = a, b
            else:
                winner, loser = b, a

            cum_points[winner] += 1.0
            total_matches += 1

            logger.debug(
                "Swiss Round %d | c%d vs c%d → winner=c%d "
                "(Δdraft=%.4f Δblade=%.4f score=%.4f)",
                rnd + 1, a, b, winner,
                delta_draft.item(), delta_blade.item(), match_score.item(),
            )

    return cum_points, rounds, total_matches


def softmax_over_points(
    points: List[float],
    beta: float,
    device: torch.device,
) -> int:
    """Select a winner by applying softmax over Swiss-system points.

    Parameters
    ----------
    points : list of float
        Cumulative points from Swiss-system tournament.
    beta : float
        Inverse temperature for softmax.
    device : torch.device

    Returns
    -------
    int
        Selected index.
    """
    pts = torch.tensor(points, dtype=torch.float, device=device)
    logits = beta * pts
    logits = logits - logits.max()  # stability
    probs = F.softmax(logits, dim=0)
    selected = int(torch.multinomial(probs, num_samples=1).item())
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class GSISwissGenerator:
    """GSI Strategy 3: Swiss-system → points table → softmax selection.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline configuration.
    tokenizer : PreTrainedTokenizer
        Shared tokenizer.
    base_model : PreTrainedModel
        Base model — serves as BOTH the drafter AND π_ref.
    blade_model : PeftModel
        Active DPO blade adapter (π_blade).
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        base_model: PreTrainedModel,
        blade_model: PeftModel,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.blade_model = blade_model
        self.blade = DPOBlade(cfg, base_model, blade_model, tokenizer)

        logger.info(
            "GSISwissGenerator initialized: n=%d, α=%.2f, β=%.3f, "
            "swiss_rounds=%d, threshold=%.3f",
            cfg.gsi_n, cfg.alpha, cfg.beta, cfg.swiss_rounds,
            cfg.gsi_threshold,
        )

    # ── Step sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample_reasoning_steps(
        self,
        prefix_ids: torch.Tensor,
        n: int,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample n reasoning steps from the base model.

        Each step is generated until the step delimiter is encountered
        or gsi_max_step_tokens is reached.

        Parameters
        ----------
        prefix_ids : torch.Tensor
            Shape ``[1, prefix_len]``.
        n : int
            Number of candidate steps.

        Returns
        -------
        step_ids_list : list of torch.Tensor
        step_texts : list of str
        """
        device = prefix_ids.device
        batch_ids = prefix_ids.expand(n, -1).contiguous()
        batch_mask = torch.ones_like(batch_ids)

        outputs = self.base_model.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=self.cfg.gsi_max_step_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prefix_len = prefix_ids.shape[1]
        delimiter = self.cfg.gsi_step_delimiter

        step_ids_list = []
        step_texts = []

        for i in range(n):
            new_tokens = outputs[i, prefix_len:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            delim_pos = decoded.find(delimiter)
            if delim_pos >= 0:
                step_text = decoded[:delim_pos + len(delimiter)]
            else:
                step_text = decoded

            step_tokens = self.tokenizer.encode(
                step_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0).to(device)

            eos_positions = (step_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                step_tokens = step_tokens[:eos_positions[0]]
                step_text = self.tokenizer.decode(step_tokens, skip_special_tokens=True)

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
        """Run GSI Strategy 3: Swiss-system → points → softmax.

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
        str | (str, GSISwissStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        threshold = self.cfg.gsi_threshold
        swiss_rounds = self.cfg.swiss_rounds

        encoded = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True,
        )
        device = next(self.base_model.parameters()).device
        prompt_ids = encoded["input_ids"].to(device)

        generated_ids: List[int] = []
        stats = GSISwissStats()
        t_start = time.perf_counter()

        while len(generated_ids) < max_tokens:
            stats.total_steps += 1

            if generated_ids:
                gen_tensor = torch.tensor(
                    generated_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
                prefix_ids = torch.cat([prompt_ids, gen_tensor], dim=1)
            else:
                prefix_ids = prompt_ids

            # ── Step 1: Sample n reasoning steps ────────────────────────
            step_ids_list, step_texts = self._sample_reasoning_steps(prefix_ids, n)
            stats.total_candidates_scored += n

            non_empty = [(ids, txt) for ids, txt in zip(step_ids_list, step_texts) if len(ids) > 0]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]
            actual_n = len(step_ids_list)

            # ── Step 2: Score with blade AND draft ──────────────────────
            blade_rewards = self.blade.score_reasoning_steps(prefix_ids, step_ids_list)
            draft_logprobs = self.blade.compute_step_draft_logprobs(prefix_ids, step_ids_list)

            # ── Step 3: Swiss-system tournament → points table ──────────
            points, n_rounds, n_matches = swiss_system_points_table(
                draft_logprobs, blade_rewards, alpha,
                rounds=swiss_rounds if swiss_rounds > 0 else 0,
            )
            stats.total_swiss_rounds += n_rounds
            stats.total_matches += n_matches
            stats.points_tables.append(points)

            if verbose:
                logger.debug(
                    "Step %d points table: %s",
                    stats.total_steps,
                    [f"c{i}:{p:.1f}" for i, p in enumerate(points)],
                )

            # ── Step 4: Softmax over points to select winner ────────────
            selected_idx = softmax_over_points(points, beta, device)
            selected_reward = blade_rewards[selected_idx].item()

            # ── Step 5: Rejection threshold check ───────────────────────
            if selected_reward >= threshold:
                stats.accepted_steps += 1
                winner_ids = step_ids_list[selected_idx]
                winner_text = step_texts[selected_idx]
            else:
                stats.rejected_steps += 1
                logger.debug(
                    "Step %d: Rejected (reward=%.4f < threshold=%.4f). Resampling...",
                    stats.total_steps, selected_reward, threshold,
                )
                resample_ids_list, resample_texts = self._sample_reasoning_steps(
                    prefix_ids, n,
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
                    prefix_ids, resample_ids_list_clean,
                )
                resample_draft = self.blade.compute_step_draft_logprobs(
                    prefix_ids, resample_ids_list_clean,
                )
                resample_points, n_r2, n_m2 = swiss_system_points_table(
                    resample_draft, resample_blade, alpha,
                    rounds=swiss_rounds if swiss_rounds > 0 else 0,
                )
                stats.total_swiss_rounds += n_r2
                stats.total_matches += n_m2

                resample_idx = softmax_over_points(resample_points, beta, device)
                selected_reward = resample_blade[resample_idx].item()
                winner_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

            stats.step_rewards.append(selected_reward)

            # ── Step 6: Commit ──────────────────────────────────────────
            winner_tokens = winner_ids.tolist()
            remaining = max_tokens - len(generated_ids)
            winner_tokens = winner_tokens[:remaining]

            eos_hit = False
            clean_tokens = []
            for tok in winner_tokens:
                if tok == self.tokenizer.eos_token_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_ids.extend(clean_tokens)
            stats.total_tokens += len(clean_tokens)

            if verbose:
                logger.info(
                    "Step %d | reward=%.4f | points=%.1f | tokens=%d | "
                    "rounds=%d matches=%d | text='%s'",
                    stats.total_steps, selected_reward, points[selected_idx],
                    len(clean_tokens), n_rounds, n_matches,
                    winner_text[:80],
                )

            if eos_hit:
                logger.info("EOS encountered. Stopping.")
                break

        # ── Finalize ─────────────────────────────────────────────────────
        stats.total_time_s = time.perf_counter() - t_start

        all_ids = prompt_ids.squeeze(0).tolist() + generated_ids
        output_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "GSI-Swiss complete | %d steps | %d tokens | %.2fs | "
                "acceptance=%.1f%% | %d rounds %d matches | %.2f tok/s",
                stats.total_steps, stats.total_tokens, stats.total_time_s,
                100 * stats.acceptance_rate, stats.total_swiss_rounds,
                stats.total_matches, stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text

    # ── Blade hot-swap support ───────────────────────────────────────────

    def swap_blade(self, blade_name: str, blade_rack) -> "ReconfigurationProfile":
        """Hot-swap the active alignment blade."""
        new_blade, profile = blade_rack.swap(blade_name)
        self.blade_model = new_blade.blade_model
        self.blade = new_blade
        return profile
