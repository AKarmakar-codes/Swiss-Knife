"""
Swiss Knife — GSI Strategy 1: Soft Best-of-N with Softmax Selection
=====================================================================

Implements Algorithm 1 from the GSI paper (Guided Speculative Inference,
ICLR 2026), adapted for Swiss Knife blades as reward models.
Uses LLaMA 3.2 3B as the drafter and Qwen 2.5 7B as the verifier/base.
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
from evaluation.retokenisation_llama_to_qwen import compute_logprob, retokenize_step

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
        # Construct internal DPOBlade with verifier model and tokenizer
        self.blade = DPOBlade(cfg, verifier_model, blade_model, verifier_tokenizer)

        # Set devices
        self.drafter_device = next(drafter_model.parameters()).device
        self.verifier_device = next(verifier_model.parameters()).device

        logger.info(
            "GSISoftmaxGenerator initialized: n=%d, β=%.3f, threshold=%.3f, "
            "max_step_tokens=%d",
            cfg.gsi_n, cfg.beta, cfg.gsi_threshold, cfg.gsi_max_step_tokens,
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
            Number of candidate steps to sample.
        device : torch.device

        Returns
        -------
        step_ids_list : list of torch.Tensor
            n tensors, each of shape ``[step_len_i]`` — token IDs for the step.
        step_texts : list of str
            Decoded text for each step.
        """
        # Expand prefix for batched generation
        batch_ids = prefix_ids.expand(n, -1).contiguous()
        batch_mask = torch.ones_like(batch_ids)

        # Generate up to max_step_tokens
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

        llama_prefix_text = prompt
        qwen_prefix_text = prompt

        generated_qwen_tokens: List[int] = []
        stats = GSISoftmaxStats()
        t_start = time.perf_counter()

        # Tokenize first prompt context to establish loop start
        initial_qwen_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_qwen_prefix_ids = initial_qwen_encoded["input_ids"].squeeze(0).tolist()

        while len(generated_qwen_tokens) < max_tokens:
            stats.total_steps += 1

            # Prepare prefix token IDs for both tokenizers
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

            # Filter empty steps
            non_empty = [(ids, txt) for ids, txt in zip(llama_step_ids_list, step_texts) if len(ids) > 0]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            llama_step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]
            n_actual = len(step_texts)

            # ── Step 2: Retokenize and compute logprobs + blade rewards ────
            tilted_rewards = []
            softmax_logits = []
            r_blades = []
            qwen_step_ids_list = []

            for i in range(n_actual):
                step_text = step_texts[i]
                llama_step_ids = llama_step_ids_list[i]

                # Compute LLaMA log probability on exact generated IDs
                llama_lp = compute_logprob(self.drafter_model, llama_prefix_ids, llama_step_ids)

                # Qwen retokenization
                qwen_step_ids = retokenize_step(
                    self.verifier_tokenizer, qwen_prefix_text, step_text, qwen_prefix_ids, self.verifier_device
                )
                qwen_step_ids_list.append(qwen_step_ids)

                # Compute Qwen log probability
                qwen_lp = compute_logprob(self.verifier_model, qwen_prefix_ids, qwen_step_ids)

                # Compute DPO blade reward
                r_blade = self.blade.score_reasoning_steps(
                    qwen_prefix_ids.unsqueeze(0), [qwen_step_ids]
                )[0].item()
                r_blades.append(r_blade)

                # Tilted reward: r_blade + (1 / beta) * (qwen_lp - llama_lp)
                tilted_r = r_blade + (1.0 / beta) * (qwen_lp - llama_lp)
                tilted_rewards.append(tilted_r)

                # Softmax logit: beta * tilted_r = beta * r_blade + qwen_lp - llama_lp
                logit = beta * r_blade + qwen_lp - llama_lp
                softmax_logits.append(logit)

            if not softmax_logits:
                logger.info("All candidate steps yielded empty evaluations. Stopping.")
                break

            # ── Step 3: Soft select winner ──────────────────────────────────
            logits_tensor = torch.tensor(softmax_logits, dtype=torch.float, device=self.verifier_device)
            logits_tensor = logits_tensor - logits_tensor.max()  # stable
            probs = F.softmax(logits_tensor, dim=0)
            selected_idx = int(torch.multinomial(probs, num_samples=1).item())

            selected_tilted_reward = tilted_rewards[selected_idx]
            selected_r_blade = r_blades[selected_idx]

            # ── Step 4: Rejection threshold check ───────────────────────────
            if selected_tilted_reward >= threshold:
                stats.accepted_steps += 1
                winner_text = step_texts[selected_idx]
                winner_qwen_step_ids = qwen_step_ids_list[selected_idx]
            else:
                stats.rejected_steps += 1
                logger.debug(
                    "Step %d: Rejected (tilted_r=%.4f < threshold=%.4f). Resampling from Qwen...",
                    stats.total_steps, selected_tilted_reward, threshold,
                )
                
                # Resample n steps from base Qwen model
                qwen_resample_ids, qwen_resample_texts = self._sample_reasoning_steps(
                    self.verifier_model, self.verifier_tokenizer, qwen_prefix_ids.unsqueeze(0), n, self.verifier_device
                )
                stats.total_candidates_scored += n

                res_non_empty = [
                    (ids, txt) for ids, txt in zip(qwen_resample_ids, qwen_resample_texts) if len(ids) > 0
                ]
                if not res_non_empty:
                    logger.info("Resample produced all empty steps. Stopping.")
                    break
                qwen_resample_ids = [x[0] for x in res_non_empty]
                qwen_resample_texts = [x[1] for x in res_non_empty]
                n_res = len(qwen_resample_texts)

                # Score Qwen resamples with DPO blade
                resample_r_blades = []
                for i in range(n_res):
                    r_b = self.blade.score_reasoning_steps(
                        qwen_prefix_ids.unsqueeze(0), [qwen_resample_ids[i]]
                    )[0].item()
                    resample_r_blades.append(r_b)

                # Soft-select using softmax(beta * r_blade)
                res_logits = torch.tensor(resample_r_blades, dtype=torch.float, device=self.verifier_device)
                res_logits = beta * res_logits
                res_logits = res_logits - res_logits.max()
                res_probs = F.softmax(res_logits, dim=0)
                res_selected_idx = int(torch.multinomial(res_probs, num_samples=1).item())

                winner_text = qwen_resample_texts[res_selected_idx]
                winner_qwen_step_ids = qwen_resample_ids[res_selected_idx]
                selected_r_blade = resample_r_blades[res_selected_idx]
                selected_tilted_reward = selected_r_blade  # no log ratio term on fallback

            stats.step_rewards.append(selected_tilted_reward)

            # ── Step 5: Commit ──────────────────────────────────────────────
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

            # Update text prefixes for next iteration
            llama_prefix_text = llama_prefix_text + winner_text
            qwen_prefix_text = qwen_prefix_text + winner_text

            if verbose:
                logger.info(
                    "Step %d | reward=%.4f | tokens=%d | accepted=%s | text='%s'",
                    stats.total_steps, selected_tilted_reward, len(clean_tokens),
                    "yes" if selected_tilted_reward >= threshold else "resample",
                    winner_text[:80],
                )

            if eos_hit:
                logger.info("EOS encountered. Stopping.")
                break

        # ── Finalize ─────────────────────────────────────────────────────
        stats.total_time_s = time.perf_counter() - t_start

        all_ids = initial_qwen_prefix_ids + generated_qwen_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

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
