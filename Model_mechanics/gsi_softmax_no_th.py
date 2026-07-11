"""
Swiss Knife — GSI Strategy 1: Soft Best-of-N with Softmax Selection
=====================================================================

Implements Algorithm 1 from the GSI paper (Guided Speculative Inference,
ICLR 2026), adapted for Swiss Knife blades as reward models.
Uses Qwen 2.5 3B as the drafter and Qwen 2.5 7B as the verifier/base.
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

# Import logprob utility (shared tokenizer — no retokenisation needed)
from evaluation.retokenisation_llama_to_qwen import compute_logprob

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
    """Steps accepted on the first sample."""

    rejected_steps: int = 0
    """Steps that triggered rejection resampling (always 0 now)."""

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
        blade_model: Optional[PeftModel] = None,
    ):
        self.cfg = cfg
        self.drafter_model = drafter_model
        self.drafter_tokenizer = drafter_tokenizer
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer
        self.blade_model = blade_model
        self._blade_cache = {}
        # Construct internal DPOBlade with verifier model and tokenizer
        if blade_model is not None:
            self.blade = DPOBlade(cfg, verifier_model, blade_model, verifier_tokenizer)
            self._blade_cache["default"] = self.blade
        else:
            self.blade = None

        # Set devices
        self.drafter_device = next(iter(drafter_model.parameters())).device
        self.verifier_device = next(iter(verifier_model.parameters())).device

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

    # ── Softmax selection ────────────────────────────────────────────────

    @staticmethod
    def _soft_select(rewards: torch.Tensor, beta: float) -> int:
        """Sample an index from softmax(β · rewards)."""
        logits = beta * rewards.float()
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
        blade: Optional[str] = None,
    ):
        """Run GSI Strategy 1: Softmax selection over reasoning steps (Without Fallback)."""
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        beta = self.cfg.beta

        # Resolve active blade
        active_blade = self.blade
        if blade is not None:
            if isinstance(blade, DPOBlade):
                active_blade = blade
            elif isinstance(blade, PeftModel):
                active_blade = DPOBlade(self.cfg, self.verifier_model, blade, self.verifier_tokenizer)
            elif isinstance(blade, str):
                if not hasattr(self, "_blade_cache"):
                    self._blade_cache = {}
                if blade not in self._blade_cache:
                    from .models import load_blade_model
                    new_blade_model = load_blade_model(self.cfg, blade)
                    self._blade_cache[blade] = DPOBlade(self.cfg, self.verifier_model, new_blade_model, self.verifier_tokenizer)
                active_blade = self._blade_cache[blade]

        if active_blade is None:
            raise ValueError(
                "No blade model/adapter provided. Please provide a blade model "
                "during initialization or pass it in the generate call."
            )

        prefix_text = prompt

        generated_qwen_tokens: List[int] = []
        stats = GSISoftmaxStats()
        t_start = time.perf_counter()

        initial_qwen_encoded = self.verifier_tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        initial_qwen_prefix_ids = initial_qwen_encoded["input_ids"].squeeze(0).tolist()

        while len(generated_qwen_tokens) < max_tokens:
            stats.total_steps += 1

            encoded = self.verifier_tokenizer(
                prefix_text, return_tensors="pt", padding=False, truncation=True
            )
            prefix_ids_verifier = encoded["input_ids"].squeeze(0).to(self.verifier_device)
            prefix_ids_drafter = prefix_ids_verifier.to(self.drafter_device)

            # ── Step 1: Sample n reasoning steps from Drafter ───────────────
            draft_step_ids_list, step_texts = self._sample_reasoning_steps(
                self.drafter_model, self.drafter_tokenizer, prefix_ids_drafter.unsqueeze(0), n, self.drafter_device
            )
            stats.total_candidates_scored += n

            # Filter empty steps
            non_empty = [(ids, txt) for ids, txt in zip(draft_step_ids_list, step_texts) if len(ids) > 0]
            if not non_empty:
                logger.info("All candidate steps empty (EOS). Stopping.")
                break
            draft_step_ids_list = [x[0] for x in non_empty]
            step_texts = [x[1] for x in non_empty]
            n_actual = len(step_texts)

            # ── Step 2: Compute logprobs + blade rewards ────────────────────
            tilted_rewards = []
            softmax_logits = []
            verifier_step_ids_list = []

            for i in range(n_actual):
                draft_step_ids = draft_step_ids_list[i]

                draft_lp = compute_logprob(self.drafter_model, prefix_ids_drafter, draft_step_ids)
                verifier_step_ids = draft_step_ids.to(self.verifier_device)
                verifier_step_ids_list.append(verifier_step_ids)

                verifier_lp = compute_logprob(self.verifier_model, prefix_ids_verifier, verifier_step_ids)

                r_blade = active_blade.score_reasoning_steps(
                    prefix_ids_verifier.unsqueeze(0), [verifier_step_ids]
                )[0].item()

                tilted_r = r_blade + (1.0 / beta) * (verifier_lp - draft_lp)
                tilted_rewards.append(tilted_r)

                logit = beta * r_blade + verifier_lp - draft_lp
                softmax_logits.append(logit)

            if not softmax_logits:
                logger.info("All candidate steps yielded empty evaluations. Stopping.")
                break

            # ── Step 3: Soft select winner ──────────────────────────────────
            logits_tensor = torch.tensor(softmax_logits, dtype=torch.float, device=self.verifier_device)
            logits_tensor = logits_tensor - logits_tensor.max()  # stable
            probs = F.softmax(logits_tensor, dim=0)
            selected_idx = int(torch.multinomial(probs, num_samples=1).item())

            # Map selection straight to winning assignments
            winner_text = step_texts[selected_idx]
            winner_qwen_step_ids = verifier_step_ids_list[selected_idx]
            selected_tilted_reward = tilted_rewards[selected_idx]

            stats.accepted_steps += 1
            stats.step_rewards.append(selected_tilted_reward)

            # ── Step 4: Commit ──────────────────────────────────────────────
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

            prefix_text = prefix_text + winner_text

            if verbose:
                logger.info(
                    "Step %d | reward=%.4f | tokens=%d | text='%s'",
                    stats.total_steps, selected_tilted_reward, len(clean_tokens),
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

    @torch.no_grad()
    def _compute_logprob_batched(
        self,
        model: PreTrainedModel,
        prefixes_list: List[torch.Tensor],
        steps_list: List[torch.Tensor],
        pad_token_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute log-probability of multiple steps given their respective prefixes in batch."""
        M = len(steps_list)
        if M == 0:
            return torch.tensor([], device=device)

        full_sequences = []
        prefix_lens = []
        step_lens = []
        max_len = 0

        for i in range(M):
            p_tensor = prefixes_list[i].to(device)
            s_tensor = steps_list[i].to(device)
            full = torch.cat([p_tensor, s_tensor])
            full_sequences.append(full)
            prefix_lens.append(len(p_tensor))
            step_lens.append(len(s_tensor))
            max_len = max(max_len, len(full))

        if max_len == 0:
            return torch.zeros(M, device=device)

        padded_ids = torch.full((M, max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(M, max_len, dtype=torch.long, device=device)

        for i, seq in enumerate(full_sequences):
            offset = max_len - len(seq)
            padded_ids[i, offset:] = seq
            attention_mask[i, offset:] = 1

        outputs = model(input_ids=padded_ids, attention_mask=attention_mask)
        logits = outputs.logits

        results = torch.zeros(M, device=device)
        for i in range(M):
            step_len = step_lens[i]
            if step_len == 0:
                continue
            p_len = prefix_lens[i]
            seq_len = p_len + step_len
            offset = max_len - seq_len

            pred_positions = torch.arange(offset + p_len - 1, offset + seq_len - 1, device=device)
            label_positions = torch.arange(offset + p_len, offset + seq_len, device=device)
            label_tokens = padded_ids[i, label_positions]

            step_logits = logits[i, pred_positions, :]
            step_logprobs = torch.log_softmax(step_logits.float(), dim=-1)
            
            lp = step_logprobs[torch.arange(step_len, device=device), label_tokens].sum()
            results[i] = lp

        return results

    @torch.no_grad()
    def _sample_reasoning_steps_batched(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prefixes_list: List[torch.Tensor],
        n: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample n reasoning steps in batch for a list of prefixes."""
        M = len(prefixes_list)
        if M == 0:
            return [], []

        max_prefix_len = max(len(p) for p in prefixes_list)

        batch_ids = []
        batch_mask = []

        for p in prefixes_list:
            p_list = p.tolist()
            pad_len = max_prefix_len - len(p_list)
            padded = [tokenizer.pad_token_id] * pad_len + p_list
            mask = [0] * pad_len + [1] * len(p_list)

            for _ in range(n):
                batch_ids.append(padded)
                batch_mask.append(mask)

        batch_ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
        batch_mask_tensor = torch.tensor(batch_mask, dtype=torch.long, device=device)

        outputs = model.generate(
            input_ids=batch_ids_tensor,
            attention_mask=batch_mask_tensor,
            max_new_tokens=self.cfg.gsi_max_step_tokens,
            do_sample=True,
            temperature=self.cfg.drafter_temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

        delimiter = self.cfg.gsi_step_delimiter
        step_ids_list = []
        step_texts = []

        for i in range(M * n):
            new_tokens = outputs[i, max_prefix_len:]
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

    @torch.no_grad()
    def generate_batched(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        blade: Optional[str] = None,
    ) -> List[str]:
        """Run GSI Strategy 1: Softmax selection in parallel (Without Fallback)."""
        batch_size = len(prompts)
        if batch_size == 0:
            return []

        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        n = self.cfg.gsi_n
        beta = self.cfg.beta

        # Resolve active blade
        active_blade = self.blade
        if blade is not None:
            if isinstance(blade, DPOBlade):
                active_blade = blade
            elif isinstance(blade, PeftModel):
                active_blade = DPOBlade(self.cfg, self.verifier_model, blade, self.verifier_tokenizer)
            elif isinstance(blade, str):
                if not hasattr(self, "_blade_cache"):
                    self._blade_cache = {}
                if blade not in self._blade_cache:
                    from .models import load_blade_model
                    new_blade_model = load_blade_model(self.cfg, blade)
                    self._blade_cache[blade] = DPOBlade(self.cfg, self.verifier_model, new_blade_model, self.verifier_tokenizer)
                active_blade = self._blade_cache[blade]

        if active_blade is None:
            raise ValueError("No blade model/adapter provided.")

        prefixes = []
        for p in prompts:
            encoded = self.verifier_tokenizer(p, return_tensors="pt")
            prefixes.append(encoded["input_ids"].squeeze(0).to(self.verifier_device))

        generated_tokens = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        eos_flags = [False] * batch_size

        while len(active_indices) > 0:
            active_prefixes = [prefixes[idx] for idx in active_indices]
            step_ids_list, step_texts = self._sample_reasoning_steps_batched(
                self.drafter_model, self.verifier_tokenizer, active_prefixes, n, self.drafter_device
            )

            flat_prefixes = []
            flat_steps = []
            for i, idx in enumerate(active_indices):
                for j in range(n):
                    flat_prefixes.append(prefixes[idx])
                    flat_steps.append(step_ids_list[i * n + j])

            draft_lps = self._compute_logprob_batched(
                self.drafter_model, flat_prefixes, flat_steps, self.verifier_tokenizer.pad_token_id, self.drafter_device
            )

            verifier_lps = self._compute_logprob_batched(
                self.verifier_model, flat_prefixes, flat_steps, self.verifier_tokenizer.pad_token_id, self.verifier_device
            )

            ref_logprobs = self._compute_logprob_batched(
                active_blade.base_model, flat_prefixes, flat_steps, self.verifier_tokenizer.pad_token_id, self.verifier_device
            )
            blade_logprobs = self._compute_logprob_batched(
                active_blade.blade_model, flat_prefixes, flat_steps, self.verifier_tokenizer.pad_token_id, self.verifier_device
            )
            r_blades = active_blade.beta * (blade_logprobs - ref_logprobs)

            winner_steps = [None] * len(active_indices)
            winner_texts = [None] * len(active_indices)

            for i, idx in enumerate(active_indices):
                start_offset = i * n
                logits = []
                for j in range(n):
                    idx_flat = start_offset + j
                    r_b = r_blades[idx_flat].item()
                    q_lp = verifier_lps[idx_flat].item()
                    l_lp = draft_lps[idx_flat].item()

                    logit = beta * r_b + q_lp - l_lp
                    logits.append(logit)

                logits_tensor = torch.tensor(logits, dtype=torch.float, device=self.verifier_device)
                logits_tensor = logits_tensor - logits_tensor.max()
                probs = F.softmax(logits_tensor, dim=0)
                selected_j = int(torch.multinomial(probs, num_samples=1).item())

                winner_steps[i] = step_ids_list[start_offset + selected_j]
                winner_texts[i] = step_texts[start_offset + selected_j]

            # ── Commit and Update State ──
            new_active_indices = []
            for i, idx in enumerate(active_indices):
                w_step = winner_steps[i]
                w_text = winner_texts[i]

                rem_tokens = max_tokens - len(generated_tokens[idx])
                w_tokens = w_step.tolist()[:rem_tokens]

                clean_tokens = []
                eos_hit = False
                for tok in w_tokens:
                    if tok == self.verifier_tokenizer.eos_token_id:
                        eos_hit = True
                        break
                    clean_tokens.append(tok)

                generated_tokens[idx].extend(clean_tokens)

                new_prefix = torch.cat([prefixes[idx], torch.tensor(clean_tokens, dtype=torch.long, device=self.verifier_device)])
                prefixes[idx] = new_prefix

                if eos_hit or len(generated_tokens[idx]) >= max_tokens:
                    eos_flags[idx] = True
                else:
                    new_active_indices.append(idx)

            active_indices = new_active_indices

        outputs = []
        for i in range(batch_size):
            decoded = self.verifier_tokenizer.decode(prefixes[i], skip_special_tokens=True)
            outputs.append(decoded)

        return outputs