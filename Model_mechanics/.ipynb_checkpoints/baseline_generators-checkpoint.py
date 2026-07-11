"""
Swiss Knife — Baselines for the GSI Strategies Harmlessness Benchmark
======================================================================

This module adds two baselines meant to sit alongside the five GSI
strategies (gsi_softmax, gsi_pairwise, gsi_swiss, gsi_elo, gsi_gumbel)
in `evaluation/benchmark_gsi_strategies_harmlessness.py`:

    1. PlainGenerator
       -----------------
       No tournament, no speculative decoding, no blade steering at
       generation time. Just `model.generate()` sampling from a single
       model. Used as the "no alignment intervention at all" reference
       point. Point this at either:
         - the 3B drafter (Qwen/Qwen2.5-3B-Instruct)   -> cheap baseline
         - the 7B verifier (divyajot5005/ndna SFT ckpt) -> matched-scale
           baseline
       The blade is NEVER consulted during generation for this class;
       the benchmark script's shared `compute_response_blade_reward()`
       still scores the finished text post-hoc for comparison, exactly
       like it does for every GSI strategy.

    2. NormalSpeculativeGenerator
       ---------------------------
       Standard (vanilla / Leviathan-Chen-style) speculative decoding:
       drafter proposes gamma tokens autoregressively, verifier does ONE
       forward pass over the drafted continuation, and tokens are
       accepted/rejected via the classic speculative-sampling ratio
         min(1, p_verifier(x) / p_drafter(x))
       with a fresh resample from the residual (p_verifier - p_drafter)_+
       distribution at the first rejected position. This is the textbook
       drafter+verifier speculative decoding baseline with NO blade /
       DPO steering anywhere in the accept/reject decision -- i.e. it is
       what you'd get if the "Blade" LoRA never existed. The harmlessness
       blade (Qwen Blade) is, again, only used afterward by the benchmark
       script to score the finished output for comparison against the
       GSI-steered strategies.

Both classes expose the exact same `generate(...)` contract used
everywhere else in this codebase:

    output_text, stats = generator.generate(
        prompt,
        max_new_tokens=...,
        verbose=...,
        return_stats=True,
    )

`output_text` includes the prompt prefix (the benchmark script slices
it off with `output[len(prompt):]`, matching every GSI generator).
`stats` exposes `.to_dict()`. For PlainGenerator this dict carries no
"acceptance_rate" / "rejected_steps" keys, so the benchmark's
`extract_override_rate()` correctly reports `None` (no tournament /
override concept applies to an unsteered single-model baseline). For
NormalSpeculativeGenerator, `.to_dict()` DOES carry an
`acceptance_rate` (fraction of drafted tokens accepted by the verifier)
so the benchmark's existing override-rate plumbing works for free --
here "override" means "the vanilla speculative sampler rejected the
draft token", NOT "the blade overrode the draft", which is an
important distinction to keep in mind when reading the comparison
table.

Wire-up in the benchmark script (for your own edits)
-----------------------------------------------------
    from Model_mechanics.baseline_generators import (
        PlainGenerator,
        NormalSpeculativeGenerator,
    )

    # Qwen 3B, unsteered, no speculative decoding at all:
    "baseline_qwen3b": lambda cfg: PlainGenerator(
        cfg, drafter_model, drafter_tokenizer,
    ),

    # Qwen 7B, unsteered, no speculative decoding at all:
    "baseline_qwen7b": lambda cfg: PlainGenerator(
        cfg, base_model, tokenizer,
    ),

    # Normal speculative decoding: Qwen 3B drafts, Qwen 7B verifies,
    # blade is NOT part of the accept/reject rule (only scored after
    # the fact by the benchmark harness, same as every other strategy):
    "normal_spec_3b_7b": lambda cfg: NormalSpeculativeGenerator(
        cfg, drafter_model, drafter_tokenizer, base_model, tokenizer,
    ),

`drafter_model`/`drafter_tokenizer` and `base_model`/`tokenizer` are
already loaded in the benchmark's `main()` via
`load_drafter_model`/`load_drafter_tokenizer` and
`load_base_model`/`load_tokenizer` respectively -- no new loader is
required for the 3B-drafts-into-7B configuration. If you also want a
plain "Qwen 7B generator + Blade verifier speculative decoding" run
(i.e. treat the blade adapter itself as the speculative verifier
instead of the frozen 7B base model), swap `base_model`/`tokenizer` in
the calls above for `blade_model`/`tokenizer` -- `NormalSpeculativeGenerator`
only requires the verifier expose the standard `PreTrainedModel`
forward-pass interface, which `PeftModel` satisfies.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SwissKnifeConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Plain single-model generation (no tournament, no speculative
# decoding, no blade steering of any kind)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlainStats:
    """Minimal statistics for a plain (single-model) generation run.

    Deliberately carries NEITHER `acceptance_rate` NOR
    `rejected_steps`/`total_steps` so that the benchmark script's
    `extract_override_rate()` returns `None` for this baseline -- there
    is no tournament/verifier to "override" the draft, since there is
    no draft-vs-verifier distinction at all in this baseline.
    """

    total_tokens_generated: int = 0
    total_time_s: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s < 1e-6:
            return 0.0
        return self.total_tokens_generated / self.total_time_s

    def to_dict(self) -> dict:
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_time_s": round(self.total_time_s, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class PlainGenerator:
    """Single-model baseline: `model.generate()`, nothing else.

    Use this to benchmark an unsteered Qwen 3B or Qwen 7B directly
    against the GSI-steered strategies and against
    `NormalSpeculativeGenerator` below. No blade adapter is touched
    during generation; the benchmark script applies the harmlessness
    blade purely as a post-hoc scorer on the finished text, identically
    for every strategy in the comparison table.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline config. Only `max_new_tokens`, `temperature`,
        `top_k`, `top_p`, and `seed` are actually used here.
    model : PreTrainedModel
        The single model to sample from (e.g. the 3B drafter or the 7B
        verifier/base model -- pass whichever you're benchmarking).
    tokenizer : PreTrainedTokenizer
        Tokenizer matching `model`.
    do_sample : bool
        If True (default), sample with temperature/top_k/top_p exactly
        like the GSI step-sampling calls elsewhere in this codebase. If
        False, generate greedily -- useful as a low-variance reference.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        do_sample: bool = True,
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.do_sample = do_sample
        self.device = next(model.parameters()).device

        logger.info(
            "PlainGenerator initialized: do_sample=%s, temperature=%.2f, "
            "top_k=%d, top_p=%.2f",
            do_sample, cfg.temperature, cfg.top_k, cfg.top_p,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        """Sample directly from `self.model` with no verifier/blade in the loop.

        Returns
        -------
        str | (str, PlainStats)
            Generated text INCLUDING the prompt prefix (matching every
            other generator's contract in this codebase), optionally
            paired with statistics.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        stats = PlainStats()
        t_start = time.perf_counter()

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if self.do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
            )
        else:
            gen_kwargs.update(do_sample=False)

        output_ids = self.model.generate(**gen_kwargs)

        stats.total_tokens_generated = int(output_ids.shape[1] - prompt_len)
        stats.total_time_s = time.perf_counter() - t_start

        output_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True,
        )

        if verbose:
            logger.info(
                "PlainGenerator: %d tokens in %.2fs (%.2f tok/s)",
                stats.total_tokens_generated, stats.total_time_s,
                stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Normal (vanilla) speculative decoding, drafter + verifier,
# NO blade / DPO steering in the accept/reject rule.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalSpeculativeStats:
    """Statistics from one vanilla speculative-decoding generation run.

    Mirrors `SpeculativeStats` in `speculative_generator.py` closely
    enough that the benchmark script's existing
    `extract_override_rate()` picks up `acceptance_rate` here for free
    -- but note the semantics: this is the fraction of *drafted tokens*
    accepted under the classic speculative-sampling ratio
    min(1, p_verifier/p_drafter), NOT a tournament/blade override rate.
    There is no blade anywhere in this loop.
    """

    total_rounds: int = 0
    """Number of draft-then-verify cycles."""

    total_tokens_accepted: int = 0
    """Tokens committed to the output (drafted + accepted, plus any
    resampled/bonus tokens), excluding EOS."""

    total_drafted_tokens: int = 0
    """Total tokens proposed by the drafter across all rounds."""

    total_tokens_accepted_from_draft: int = 0
    """Of the drafted tokens, how many passed the accept test (i.e.
    were NOT rejected and resampled)."""

    verifier_forward_passes: int = 0
    """Number of verifier forward passes (one per round, evaluating the
    whole drafted chunk in parallel)."""

    drafter_forward_passes: int = 0
    """Number of drafter forward passes (one per drafted token)."""

    total_time_s: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens accepted by the verifier's
        speculative-sampling test (standard speculative decoding
        metric -- higher means the drafter and verifier agree more,
        i.e. more speedup)."""
        if self.total_drafted_tokens == 0:
            return 0.0
        return self.total_tokens_accepted_from_draft / self.total_drafted_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s < 1e-6:
            return 0.0
        return self.total_tokens_accepted / self.total_time_s

    def to_dict(self) -> dict:
        return {
            "total_rounds": self.total_rounds,
            "total_tokens_accepted": self.total_tokens_accepted,
            "total_drafted_tokens": self.total_drafted_tokens,
            "total_tokens_accepted_from_draft": self.total_tokens_accepted_from_draft,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "verifier_forward_passes": self.verifier_forward_passes,
            "drafter_forward_passes": self.drafter_forward_passes,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time_s": round(self.total_time_s, 3),
        }


class NormalSpeculativeGenerator:
    """Textbook drafter+verifier speculative decoding (Leviathan et al. /
    Chen et al. style), with NO blade or DPO steering in the loop.

    Algorithm (per round):
        1. Drafter autoregressively proposes `gamma` tokens, recording
           its own per-token sampling distribution p_drafter(x_i).
        2. Verifier does ONE forward pass over the drafted chunk
           (context + gamma proposed tokens), producing
           p_verifier(x_i) for each drafted position in parallel.
        3. For i = 1..gamma, accept x_i with probability
               min(1, p_verifier(x_i) / p_drafter(x_i))
           At the first rejection, resample a token from the residual
           distribution  norm(max(0, p_verifier - p_drafter))  at that
           position and stop the round there (standard speculative
           decoding: discard the drafted tail).
        4. If all gamma tokens are accepted, additionally sample one
           bonus token directly from the verifier's next-token
           distribution at the final position (this is the standard
           "extra token for free" step in speculative decoding, since
           the verifier's forward pass already computed it).

    Nothing here consults a blade/DPO adapter -- this is the reference
    "normal speculative decoding" point of comparison against the
    GSI-steered strategies and against `PlainGenerator`. The harmlessness
    blade is only applied afterward, by the benchmark script, purely as
    a scorer of the finished text.

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Full pipeline config (`gamma`, `max_new_tokens`, `temperature`,
        `top_k`, `top_p`, `seed`).
    drafter_model : PreTrainedModel
        The cheap draft model (e.g. Qwen 2.5 3B Instruct).
    drafter_tokenizer : PreTrainedTokenizer
        Tokenizer for the drafter. Must share vocabulary with the
        verifier tokenizer for speculative decoding to be valid; when
        drafter and verifier are both Qwen2.5-family models (as in this
        repo) this holds.
    verifier_model : PreTrainedModel
        The larger/target model used to validate draft proposals (e.g.
        the 7B SFT-merged checkpoint, `base_model` in the benchmark
        script) -- or, if you want a "blade-as-verifier" variant, the
        `PeftModel` blade itself (same forward-pass interface).
    verifier_tokenizer : PreTrainedTokenizer
        Tokenizer for the verifier (in this repo, the same tokenizer
        object as the drafter's, since both are Qwen2.5 family).
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model: PreTrainedModel,
        drafter_tokenizer: PreTrainedTokenizer,
        verifier_model: PreTrainedModel,
        verifier_tokenizer: PreTrainedTokenizer,
    ):
        self.cfg = cfg
        self.drafter_model = drafter_model
        self.drafter_tokenizer = drafter_tokenizer
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer

        self.drafter_device = next(drafter_model.parameters()).device
        self.verifier_device = next(verifier_model.parameters()).device

        self._seed = cfg.seed
        self._generator = torch.Generator(device=self.drafter_device)
        if self._seed is not None:
            self._generator.manual_seed(self._seed)

        logger.info(
            "NormalSpeculativeGenerator initialized: gamma=%d, temperature=%.2f, "
            "top_k=%d, top_p=%.2f (no blade in accept/reject loop)",
            cfg.gamma, cfg.temperature, cfg.top_k, cfg.top_p,
        )

    # ── Sampling helpers ─────────────────────────────────────────────────

    def _filtered_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured temperature/top-k/top-p filtering and
        return a normalized probability distribution over the vocab.

        Shared by both drafter and verifier so that p_drafter(x) and
        p_verifier(x) are computed under the exact same sampling rule
        -- required for the speculative-sampling accept/reject ratio to
        be well-defined and unbiased.
        """
        temperature = max(self.cfg.temperature, 1e-5)
        logits = logits / temperature

        top_k = self.cfg.top_k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            kth_val = torch.topk(logits, top_k)[0][..., -1, None]
            logits = torch.where(
                logits < kth_val, torch.full_like(logits, float("-inf")), logits,
            )

        top_p = self.cfg.top_p
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits,
            )

        probs = F.softmax(logits, dim=-1)
        return probs

    def _sample_from(self, probs: torch.Tensor) -> int:
        return int(torch.multinomial(probs, num_samples=1, generator=self._generator).item())

    # ── Main generation loop ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        """Run standard drafter+verifier speculative decoding (no blade).

        Returns
        -------
        str | (str, NormalSpeculativeStats)
            Generated text INCLUDING the prompt prefix, optionally
            paired with statistics. Matches the shared contract used by
            every GSI generator and by `PlainGenerator` above.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        gamma = self.cfg.gamma

        encoded = self.drafter_tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        prompt_ids = encoded["input_ids"].to(self.drafter_device)

        generated_ids: List[int] = []
        stats = NormalSpeculativeStats()
        t_start = time.perf_counter()
        eos_id = self.drafter_tokenizer.eos_token_id

        while len(generated_ids) < max_tokens:
            stats.total_rounds += 1

            if generated_ids:
                gen_tensor = torch.tensor(
                    generated_ids, dtype=torch.long, device=self.drafter_device,
                ).unsqueeze(0)
                context_ids = torch.cat([prompt_ids, gen_tensor], dim=1)
            else:
                context_ids = prompt_ids

            steps_remaining = max_tokens - len(generated_ids)
            gamma_this_round = min(gamma, steps_remaining)
            if gamma_this_round <= 0:
                break

            # ── Step 1: drafter autoregressively proposes gamma tokens ──
            draft_tokens: List[int] = []
            draft_probs_list: List[torch.Tensor] = []
            running_ids = context_ids.clone()

            for _ in range(gamma_this_round):
                out = self.drafter_model(
                    input_ids=running_ids,
                    attention_mask=torch.ones_like(running_ids),
                )
                logits_t = out.logits[0, -1, :]
                probs_t = self._filtered_probs(logits_t)
                token_t = self._sample_from(probs_t)

                draft_tokens.append(token_t)
                draft_probs_list.append(probs_t)
                stats.drafter_forward_passes += 1

                running_ids = torch.cat(
                    [running_ids,
                     torch.tensor([[token_t]], device=self.drafter_device)],
                    dim=1,
                )

            # ── Step 2: verifier scores the drafted chunk in ONE pass ───
            draft_tensor = torch.tensor(
                draft_tokens, dtype=torch.long, device=self.verifier_device,
            ).unsqueeze(0)
            verifier_context = context_ids.to(self.verifier_device)
            full_seq = torch.cat([verifier_context, draft_tensor], dim=1)

            verifier_out = self.verifier_model(
                input_ids=full_seq,
                attention_mask=torch.ones_like(full_seq),
            )
            stats.verifier_forward_passes += 1

            context_len = verifier_context.shape[1]
            # Logits predicting position (context_len - 1 + i) for i in [0, gamma)
            # i.e. the verifier's distribution over the token AT each drafted
            # position, conditioned on everything before it.
            verifier_logits = verifier_out.logits[
                0, context_len - 1: context_len - 1 + gamma_this_round, :
            ]

            # ── Step 3: accept/reject each drafted token ────────────────
            accepted: List[int] = []
            rejection_occurred = False

            for i in range(gamma_this_round):
                stats.total_drafted_tokens += 1

                v_logits_i = verifier_logits[i].to(self.drafter_device)
                v_probs_i = self._filtered_probs(v_logits_i)
                d_probs_i = draft_probs_list[i]
                x_i = draft_tokens[i]

                p_v = v_probs_i[x_i].clamp(min=1e-12)
                p_d = d_probs_i[x_i].clamp(min=1e-12)
                accept_prob = torch.clamp(p_v / p_d, max=1.0)

                coin = torch.rand((), generator=self._generator, device=self.drafter_device).item()
                if coin < accept_prob.item():
                    accepted.append(x_i)
                    stats.total_tokens_accepted_from_draft += 1
                    if verbose:
                        logger.info(
                            "Round %d pos %d: ACCEPT token=%d (p_v=%.4f p_d=%.4f)",
                            stats.total_rounds, i, x_i, p_v.item(), p_d.item(),
                        )
                else:
                    # Reject: resample from the residual distribution
                    # norm(max(0, p_verifier - p_drafter)) at this position.
                    residual = torch.clamp(v_probs_i - d_probs_i, min=0.0)
                    residual_sum = residual.sum()
                    if residual_sum > 1e-8:
                        residual = residual / residual_sum
                    else:
                        residual = v_probs_i
                    resampled_token = self._sample_from(residual)
                    accepted.append(resampled_token)
                    rejection_occurred = True
                    if verbose:
                        logger.info(
                            "Round %d pos %d: REJECT draft=%d -> resample=%d "
                            "(p_v=%.4f p_d=%.4f)",
                            stats.total_rounds, i, x_i, resampled_token,
                            p_v.item(), p_d.item(),
                        )
                    break  # discard the rest of the drafted tail

            if not rejection_occurred:
                # All gamma tokens accepted -> bonus token for free from the
                # verifier's distribution at the final position (standard
                # speculative decoding "free extra token").
                bonus_pos = context_len - 1 + gamma_this_round
                last_logits = (
                    verifier_out.logits[0, bonus_pos, :]
                    if bonus_pos < verifier_out.logits.shape[1]
                    else None
                )
                if last_logits is not None:
                    bonus_probs = self._filtered_probs(last_logits.to(self.drafter_device))
                    bonus_token = self._sample_from(bonus_probs)
                    accepted.append(bonus_token)

            # ── Commit accepted tokens (respecting the token budget) ────
            remaining = max_tokens - len(generated_ids)
            accepted = accepted[:remaining]

            eos_hit = False
            clean_tokens = []
            for tok in accepted:
                if tok == eos_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_ids.extend(clean_tokens)
            stats.total_tokens_accepted += len(clean_tokens)

            if eos_hit or len(generated_ids) >= max_tokens:
                break

        stats.total_time_s = time.perf_counter() - t_start

        all_ids = prompt_ids.squeeze(0).tolist() + generated_ids
        output_text = self.drafter_tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "NormalSpeculativeGenerator: %d tokens in %.2fs | "
                "%d rounds | acceptance_rate=%.4f | %.2f tok/s",
                stats.total_tokens_accepted, stats.total_time_s,
                stats.total_rounds, stats.acceptance_rate,
                stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text