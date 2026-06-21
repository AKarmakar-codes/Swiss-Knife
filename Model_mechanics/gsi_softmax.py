"""
Swiss Knife — GSI Strategy 1: Soft Best-of-N with Softmax Selection
=====================================================================

Implements Algorithm 1 from the GSI paper (Guided Speculative Inference,
ICLR 2026), adapted for Swiss Knife blades as reward models.

At each reasoning step:
    1. Sample n candidate reasoning steps from the base model.
    2. Score each step: blade reward r_i = β·(log π_blade - log π_ref)
    3. Compute tilted rewards: r̃_i = r_i  (since πS = πB, log-ratio = 0)
    4. Select winner: i* ~ softmax(β · r̃_1, ..., β · r̃_n)
    5. If r̃_{i*} >= threshold u:
         Accept: y ← y ⊕ y_{i*}
       Else:
         Resample n steps from base model, rescore, soft-select, accept.

This is the standard GSI selection — softmax-weighted sampling over
blade rewards. No pairwise comparisons, no tournament structure.
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

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GSISoftmaxStats:
    """Statistics from one GSI softmax generation run."""

    total_steps: int = 0
    """Number of reasoning steps produced."""

    total_tokens: int = 0
    """Total tokens generated across all steps."""

    accepted_steps: int = 0
    """Steps accepted on the first sample (above threshold)."""

    rejected_steps: int = 0
    """Steps that triggered rejection resampling."""

    total_candidates_scored: int = 0
    """Total number of candidate steps scored across all iterations."""

    total_time_s: float = 0.0
    """Wall-clock time in seconds."""

    step_rewards: List[float] = field(default_factory=list)
    """Reward of the selected step at each iteration."""

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
            "strategy": "gsi_softmax",
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

class GSISoftmaxGenerator:
    """GSI Strategy 1: Soft Best-of-N with softmax(β·r̃) selection.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline configuration.
    tokenizer : PreTrainedTokenizer
        Shared tokenizer.
    base_model : PreTrainedModel
        Base model — serves as BOTH the drafter (sample steps) AND π_ref.
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
            "GSISoftmaxGenerator initialized: n=%d, β=%.3f, threshold=%.3f, "
            "max_step_tokens=%d",
            cfg.gsi_n, cfg.beta, cfg.gsi_threshold, cfg.gsi_max_step_tokens,
        )

    # ── Step sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample_reasoning_steps(
        self,
        prefix_ids: torch.Tensor,
        n: int,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample n reasoning steps from the base model.

        Each step is generated until the step delimiter (e.g. '\\n\\n') is
        encountered or max_step_tokens is reached.

        Parameters
        ----------
        prefix_ids : torch.Tensor
            Shape ``[1, prefix_len]`` — current context.
        n : int
            Number of candidate steps to sample.

        Returns
        -------
        step_ids_list : list of torch.Tensor
            n tensors, each of shape ``[step_len_i]`` — token IDs for the step.
        step_texts : list of str
            Decoded text for each step (for logging/diagnostics).
        """
        device = prefix_ids.device

        # Expand prefix for batched generation
        batch_ids = prefix_ids.expand(n, -1).contiguous()
        batch_mask = torch.ones_like(batch_ids)

        # Generate up to max_step_tokens
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

            # Decode to find the step delimiter
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            delim_pos = decoded.find(delimiter)
            if delim_pos >= 0:
                # Truncate at delimiter (include delimiter in the step)
                step_text = decoded[:delim_pos + len(delimiter)]
            else:
                # No delimiter found — use the whole generation
                step_text = decoded

            # Re-tokenize the truncated step to get exact token IDs
            step_tokens = self.tokenizer.encode(
                step_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0).to(device)

            # Handle EOS — truncate at first EOS if present
            eos_positions = (step_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                step_tokens = step_tokens[:eos_positions[0]]
                step_text = self.tokenizer.decode(step_tokens, skip_special_tokens=True)

            step_ids_list.append(step_tokens)
            step_texts.append(step_text)

        return step_ids_list, step_texts

    # ── Softmax selection ────────────────────────────────────────────────

    @staticmethod
    def _soft_select(rewards: torch.Tensor, beta: float) -> int:
        """Sample an index from softmax(β · rewards).

        Parameters
        ----------
        rewards : torch.Tensor
            Shape ``[n]`` — blade rewards for each candidate.
        beta : float
            Inverse temperature. Higher β → sharper selection.

        Returns
        -------
        int
            Selected index.
        """
        logits = beta * rewards.float()
        # Stabilize by subtracting max
        logits = logits - logits.max()
        probs = F.softmax(logits, dim=0)
        selected = int(torch.multinomial(probs, num_samples=1).item())
        return selected

    # ── Main generation loop ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        """Run GSI Strategy 1: Softmax selection over reasoning steps.

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
        str | (str, GSISoftmaxStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        beta = self.cfg.beta
        threshold = self.cfg.gsi_threshold

        # Tokenize prompt
        encoded = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True,
        )
        device = next(self.base_model.parameters()).device
        prompt_ids = encoded["input_ids"].to(device)

        # ── GSI main loop: y ← () ───────────────────────────────────────
        generated_ids: List[int] = []
        stats = GSISoftmaxStats()
        t_start = time.perf_counter()

        while len(generated_ids) < max_tokens:
            stats.total_steps += 1

            # Current prefix = prompt ⊕ generated-so-far
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

            # Filter empty steps
            non_empty = [(ids, txt) for ids, txt in zip(step_ids_list, step_texts) if len(ids) > 0]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]

            # ── Step 2: Score with blade ────────────────────────────────
            blade_rewards = self.blade.score_reasoning_steps(prefix_ids, step_ids_list)

            # Since πS = πB, tilted reward = blade reward (log ratio = 0)
            tilted_rewards = blade_rewards

            # ── Step 3: Soft select via softmax(β · r̃) ─────────────────
            selected_idx = self._soft_select(tilted_rewards, beta)
            selected_reward = tilted_rewards[selected_idx].item()

            # ── Step 4: Rejection threshold check ───────────────────────
            if selected_reward >= threshold:
                # Accept
                stats.accepted_steps += 1
                winner_ids = step_ids_list[selected_idx]
                winner_text = step_texts[selected_idx]
            else:
                # Reject → resample and soft-select again
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

                resample_rewards = self.blade.score_reasoning_steps(
                    prefix_ids, resample_ids_list_clean,
                )
                resample_idx = self._soft_select(resample_rewards, beta)
                selected_reward = resample_rewards[resample_idx].item()
                winner_ids = resample_ids_list_clean[resample_idx]
                winner_text = resample_texts_clean[resample_idx]

            stats.step_rewards.append(selected_reward)

            # ── Step 5: Commit ──────────────────────────────────────────
            winner_tokens = winner_ids.tolist()

            # Truncate to budget
            remaining = max_tokens - len(generated_ids)
            winner_tokens = winner_tokens[:remaining]

            # Check for EOS
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
                    "Step %d | reward=%.4f | tokens=%d | accepted=%s | text='%s'",
                    stats.total_steps, selected_reward, len(clean_tokens),
                    "yes" if stats.total_steps == stats.accepted_steps + stats.rejected_steps
                    and selected_reward >= threshold else "resample",
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
                "GSI-Softmax complete | %d steps | %d tokens | %.2fs | "
                "acceptance=%.1f%% | %.2f tok/s",
                stats.total_steps, stats.total_tokens, stats.total_time_s,
                100 * stats.acceptance_rate, stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text

    # ── Blade hot-swap support ───────────────────────────────────────────

    def swap_blade(self, blade_name: str, blade_rack) -> "ReconfigurationProfile":
        """Hot-swap the active alignment blade."""
        new_blade, profile = blade_rack.swap(blade_name)
        self.blade_model = new_blade.blade_model
        self.blade = new_blade
        return profile

