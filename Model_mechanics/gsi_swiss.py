"""
Swiss Knife — GSI Strategy 3: Swiss-System Matches → Points Table → Softmax
=============================================================================

Implements the Swiss Knife's novel tournament selection adapted for GSI
step-level inference. This is the strategy that combines the Swiss-system
pairing mechanism (§4.3.1 of swiss_knife_analysis.pdf) with softmax
selection over the final cumulative points.
Uses LLaMA 3.2 3B as the drafter and Qwen 2.5 7B as the verifier/base.
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

# Import retokenisation utilities from evaluation
from evaluation.retokenisation_llama_to_qwen import compute_logprob, retokenize_step

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
            "GSISwissGenerator initialized: n=%d, α=%.2f, β=%.3f, "
            "swiss_rounds=%d, threshold=%.3f",
            cfg.gsi_n, cfg.alpha, cfg.beta, cfg.swiss_rounds,
            cfg.gsi_threshold,
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

        llama_prefix_text = prompt
        qwen_prefix_text = prompt

        generated_qwen_tokens: List[int] = []
        stats = GSISwissStats()
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

            # ── Step 2: Swiss-system tournament → points table ──────────
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

            # ── Step 3: Softmax over points to select winner ────────────
            selected_idx = softmax_over_points(points, beta, self.verifier_device)
            selected_reward = blade_rewards[selected_idx].item()
            winner_draft_lp = draft_logprobs_list[selected_idx]
            winner_qwen_step_ids = qwen_step_ids_list[selected_idx]

            # ── Step 4: Compute tilted reward for the winner ────────────────
            winner_target_lp = compute_logprob(self.verifier_model, qwen_prefix_ids, winner_qwen_step_ids)
            selected_tilted_reward = selected_reward + (1.0 / beta) * (winner_target_lp - winner_draft_lp)

            # ── Step 5: Rejection threshold check ───────────────────────
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
                resample_points, n_r2, n_m2 = swiss_system_points_table(
                    resample_draft, resample_blade, alpha,
                    rounds=swiss_rounds if swiss_rounds > 0 else 0,
                )
                stats.total_swiss_rounds += n_r2
                stats.total_matches += n_m2

                resample_idx = softmax_over_points(resample_points, beta, self.verifier_device)
                selected_reward = resample_blade[resample_idx].item()
                selected_tilted_reward = selected_reward  # no log ratio term on fallback
                winner_qwen_step_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

            stats.step_rewards.append(selected_tilted_reward)

            # ── Step 6: Commit ──────────────────────────────────────────
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
                    "Step %d | tilted_r=%.4f | points=%.1f | tokens=%d | "
                    "rounds=%d matches=%d | text='%s'",
                    stats.total_steps, selected_tilted_reward, points[selected_idx],
                    len(clean_tokens), n_rounds, n_matches,
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
