"""
Swiss Knife — GSI Strategy 3: Swiss-System Matches → Points Table → Softmax
=============================================================================

Implements the Swiss Knife's tournament selection adapted for GSI
step-level inference. This combines the Swiss-system pairing mechanism 
(§4.3.1 of swiss_knife_analysis.pdf) with softmax selection over the final 
cumulative points. 
Uses Qwen 2.5 3B as the drafter and Qwen 2.5 7B as the verifier/base.

Algorithm Pipeline:
-------------------
1. Drafting: Sample `n` candidate reasoning steps from the drafting model (π_S).
2. Swiss Tournament (R rounds):
   - Candidates are paired based on their current cumulative points (similar 
     scores play each other, avoiding rematches when possible). If `n` is odd, 
     the unpaired candidate gets a "bye" (0.5 points).
   - Match evaluation: 
     - If `use_tilted_selection` is True: The winner of A vs B is decided directly by comparing their tilted rewards:
       MATCH(A,B) = tilted_reward(A) - tilted_reward(B)
       where tilted_reward = r_blade + (1/β)*(log π_verifier - log π_draft).
     - If `use_tilted_selection` is False: The winner is decided using the blended match score:
       MATCH(A,B) = α·Δlog_draft + (1-α)·Δblade.
   - Winner gets 1 point; loser gets 0 points.
3. Softmax Selection: Apply softmax (temperature β) over the final points 
   table to stochastically select a winning step.
4. Verification & Thresholding: 
   - If r_tilted >= threshold, accept the step, append to prefix, and repeat.
   - If r_tilted < threshold, reject. Fall back to sampling directly from the 
     verifier (π_B), run the tournament again, and accept the selected step unconditionally.
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

# Import logprob utility (shared tokenizer — no retokenisation needed)
from evaluation.retokenisation_llama_to_qwen import compute_logprob

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwissStats:
    """Statistics from one Swiss-system generation run."""

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

    @property
    def avg_step_tokens(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_tokens / self.total_steps

    def to_dict(self) -> dict:
        return {
            "strategy": "swiss",
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_candidates_scored": self.total_candidates_scored,
            "total_swiss_rounds": self.total_swiss_rounds,
            "total_matches": self.total_matches,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "avg_step_tokens": round(self.avg_step_tokens, 2),
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
    tilted_rewards: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    hard_draw: bool = False,
) -> Tuple[List[float], int, int]:
    """Run Swiss-system matches and return the cumulative points table.

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
    tilted_rewards : torch.Tensor, optional
        Shape ``[n]``. Precomputed tilted rewards for all candidates.
    sigmas : torch.Tensor, optional
        Shape ``[n]``. Standard deviation of the blade rewards.
    hard_draw : bool
        If True, sample outcomes using Bernoulli trial. If False, use continuous win probability.

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
        if t.numel() <= 1:
            return torch.zeros_like(t)
        std = t.std()
        if std < 1e-8:
            return t - t.mean()
        return (t - t.mean()) / (std + 1e-6)

    # Calculate standard deviations before normalization
    std_draft = draft_scores.std().item() if draft_scores.numel() > 1 else 1.0
    std_blade = blade_scores.std().item() if blade_scores.numel() > 1 else 1.0
    std_tilted = tilted_rewards.std().item() if (tilted_rewards is not None and tilted_rewards.numel() > 1) else 1.0

    draft_normed = _znorm(draft_scores.float())
    blade_normed = _znorm(blade_scores.float())
    if tilted_rewards is not None:
        tilted_normed = _znorm(tilted_rewards.float())
    else:
        tilted_normed = None

    if sigmas is not None:
        if tilted_rewards is not None:
            sigmas_normed = sigmas.float() / (std_tilted + 1e-8)
        else:
            sigmas_normed = sigmas.float() / (std_blade + 1e-8)
    else:
        sigmas_normed = None

    # Cumulative points
    cum_points = [0.0] * n
    paired_before = set()
    indices = list(range(n))
    total_matches = 0
    import random

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
            if sigmas_normed is not None:
                # Thurstonian match win probability
                if tilted_normed is not None:
                    var_match = sigmas_normed[a]**2 + sigmas_normed[b]**2
                    diff = tilted_normed[a] - tilted_normed[b]
                else:
                    var_match = ((1.0 - alpha) ** 2) * (sigmas_normed[a]**2 + sigmas_normed[b]**2)
                    diff = alpha * (draft_normed[a] - draft_normed[b]) + (1.0 - alpha) * (blade_normed[a] - blade_normed[b])

                denom = torch.sqrt(var_match + 1e-8)
                P_A_beats_B = 0.5 * (1.0 + torch.erf((diff / denom) / math.sqrt(2.0))).item()
            else:
                # Bradley-Terry win probability
                if tilted_normed is not None:
                    diff = tilted_normed[a] - tilted_normed[b]
                else:
                    diff = alpha * (draft_normed[a] - draft_normed[b]) + (1.0 - alpha) * (blade_normed[a] - blade_normed[b])
                P_A_beats_B = torch.sigmoid(diff).item()

            if hard_draw:
                if random.random() < P_A_beats_B:
                    sa, sb = 1.0, 0.0
                    winner = a
                else:
                    sa, sb = 0.0, 1.0
                    winner = b
            else:
                sa, sb = P_A_beats_B, 1.0 - P_A_beats_B
                winner = a if P_A_beats_B > 0.5 else b

            cum_points[a] += sa
            cum_points[b] += sb
            total_matches += 1

            logger.debug(
                "Swiss Round %d | c%d vs c%d → sa=%.3f, sb=%.3f (prob=%.3f) | winner=c%d",
                rnd + 1, a, b, sa, sb, P_A_beats_B, winner
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

class SwissGenerator:
    """GSI Strategy 3: Swiss-system → points table → softmax selection.

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
            "SwissGenerator initialized: n=%d, α=%.2f, β=%.3f, "
            "swiss_rounds=%d, threshold=%.3f, sigma_mode=%s",
            cfg.gsi_n, cfg.alpha, cfg.beta, cfg.swiss_rounds,
            cfg.gsi_threshold, cfg.sigma_mode,
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
        use_tilted_selection: Optional[bool] = None,
    ):
        """Run Swiss-system → points → softmax selection.

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
        str | (str, SwissStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        threshold = self.cfg.gsi_threshold
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

            # Compute Drafter logprobs (no retokenization needed since tokenizers are identical)
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

            if verbose:
                logger.debug(
                    "Step %d points table: %s",
                    stats.total_steps,
                    [f"c{i}:{p:.1f}" for i, p in enumerate(points)],
                )

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

            # ── Step 4: Compute tilted reward for the winner ────────────────
            if active_use_tilted:
                winner_target_lp = verifier_logprobs[selected_idx].item()
            else:
                winner_target_lp = compute_logprob(self.verifier_model, prefix_ids_verifier, winner_verifier_step_ids)
            selected_tilted_reward = selected_reward + (1.0 / beta) * (winner_target_lp - winner_draft_lp)

            # ── Step 5: Adaptive threshold calibration & Fallback check ───────────────────────
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

                resample_points, n_r2, n_m2 = swiss_system_points_table(
                    resample_verifier_logprobs, resample_blade, alpha,
                    rounds=swiss_rounds if swiss_rounds > 0 else 0,
                    tilted_rewards=resample_blade if active_use_tilted else None,
                    sigmas=resample_sigma if self.cfg.sigma_mode != "none" else None,
                    hard_draw=self.cfg.hard_draw,
                )
                stats.total_swiss_rounds += n_r2
                stats.total_matches += n_m2

                # Selection for fallback
                resample_pts_tensor = torch.tensor(resample_points, dtype=torch.float, device=self.verifier_device)
                resample_logits = w_tournament * resample_pts_tensor + w_blade * (resample_blade - uwo_lambda * resample_sigma)
                resample_logits = resample_logits - resample_logits.max()
                resample_probs = torch.softmax(resample_logits, dim=0)
                resample_idx = int(torch.multinomial(resample_probs, num_samples=1).item())

                selected_reward = resample_blade[resample_idx].item()
                selected_tilted_reward = selected_reward  # no log ratio term on fallback
                winner_verifier_step_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

                # Update running threshold buffers with accepted fallback step (kl is 0)
                self.threshold_calibrator.update(selected_reward, 0.0)

            stats.step_rewards.append(selected_tilted_reward)

            # ── Step 6: Commit ──────────────────────────────────────────
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

        all_ids = initial_prefix_ids + generated_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "Swiss complete | %d steps | %d tokens | %.2fs | "
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
