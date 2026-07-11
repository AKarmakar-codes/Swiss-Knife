"""
Swiss Knife — GSI Strategy 5: Speculative Gumbel-Top-k with GSI Fallback
=====================================================================

Implements a token-level speculative Gumbel-Top-k generation loop with
DPO blade rewards and Guided Speculative Inference (GSI) fallback.
Uses Qwen 2.5 3B as the drafter and Qwen 2.5 7B as the verifier/base.
"""

import logging
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
class GSIGumbelStats:
    """Collected statistics from one GSI Gumbel-Top-k generation run."""

    total_rounds: int = 0
    """Total speculative draft-verify cycles."""

    total_tokens_accepted: int = 0
    """Tokens committed to y (excluding EOS)."""

    full_accept_rounds: int = 0
    """Rounds where all γ draft tokens were accepted (no rejection)."""

    partial_accept_rounds: int = 0
    """Rounds where rejection occurred mid-sequence."""

    fallback_rounds: int = 0
    """Rounds that triggered expensive fallback verification."""

    acceptance_positions: List[int] = field(default_factory=list)
    """List of accept-prefix lengths before first rejection in partial rounds."""

    target_forward_passes: int = 0
    """Total target model forward passes."""

    blade_forward_passes: int = 0
    """Total blade model forward passes."""

    drafter_forward_passes: int = 0
    """Total drafter model forward passes."""

    total_time_s: float = 0.0
    """Wall-clock generation time in seconds."""

    @property
    def acceptance_rate(self) -> float:
        """Fraction of speculative rounds that accepted all γ tokens."""
        if self.total_rounds == 0:
            return 0.0
        return self.full_accept_rounds / self.total_rounds

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s < 1e-6:
            return 0.0
        return self.total_tokens_accepted / self.total_time_s

    def to_dict(self) -> dict:
        return {
            "strategy": "gsi_gumbel",
            "total_rounds": self.total_rounds,
            "total_tokens_accepted": self.total_tokens_accepted,
            "full_accept_rounds": self.full_accept_rounds,
            "partial_accept_rounds": self.partial_accept_rounds,
            "fallback_rounds": self.fallback_rounds,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "target_forward_passes": self.target_forward_passes,
            "blade_forward_passes": self.blade_forward_passes,
            "drafter_forward_passes": self.drafter_forward_passes,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time_s": round(self.total_time_s, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class GSIGumbelGenerator:
    """GSI Strategy 5: Speculative Gumbel-Top-k with GSI Fallback.

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
            "GSIGumbelGenerator initialized: γ=%d, K=%d, α=%.3f, β=%.3f, threshold=%.3f",
            cfg.gamma, cfg.K, cfg.alpha, cfg.beta, cfg.gsi_threshold,
        )

    # ── Draft proposal: [γ, K] candidate tensor ──────────────────────────

    @torch.no_grad()
    def _draft_propose(
        self,
        context_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the draft model autoregressively for γ steps, capturing top-K
        token IDs at each step.

        Returns candidate_ids and draft_topk_logits on verifier_device so the
        rest of the pipeline (which lives on verifier_device) can use them
        without extra .to() calls.
        """
        gamma = self.cfg.gamma
        K = self.cfg.K
        # context_ids lives on verifier_device; run_ids must stay on verifier_device
        # for the cat, but forward pass must go to drafter_device.
        device = context_ids.device  # verifier_device

        candidate_ids = torch.zeros(gamma, K, dtype=torch.long, device=device)
        draft_topk_logits = torch.zeros(gamma, K, device=device)

        running_ids = context_ids.clone()  # [1, T] on verifier_device

        for t in range(gamma):
            # Move to drafter for the forward pass only
            outputs = self.drafter_model(
                input_ids=running_ids.to(self.drafter_device),
                attention_mask=torch.ones(
                    running_ids.shape, dtype=torch.long, device=self.drafter_device
                ),
            )
            # Return logits to verifier_device immediately
            logits_t = outputs.logits[0, -1, :].to(device)  # [vocab_size]

            topk_logits, topk_ids = torch.topk(logits_t, K)  # both on device
            candidate_ids[t] = topk_ids
            draft_topk_logits[t] = topk_logits

            # Append greedy token back on verifier_device
            greedy_token = topk_ids[0:1].unsqueeze(0)  # [1, 1] on device
            running_ids = torch.cat([running_ids, greedy_token], dim=1)

        return candidate_ids, draft_topk_logits  # [γ, K], [γ, K] on device

    # ── Parallel Drafter Logprobs ───────────────────────────────────────

    @torch.no_grad()
    def _drafter_logprob_parallel(
        self,
        context_ids: torch.Tensor,
        candidate_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drafter log-probabilities for all [gamma, K] candidates.

        Mirrors the same greedy-path trick used in DPOBlade.score_parallel:
        feed [context + greedy_draft_tokens] through the drafter in one pass,
        then gather log-probs at each position for all K candidates.

        Returns result on the same device as context_ids (verifier_device).
        """
        gamma, K = candidate_matrix.shape
        context_len = context_ids.shape[1]
        device = context_ids.device  # verifier_device

        # Build greedy path on drafter_device
        greedy_tokens = candidate_matrix[:, 0].to(self.drafter_device)  # [gamma]
        full_ids = torch.cat([
            context_ids.squeeze(0).to(self.drafter_device),
            greedy_tokens,
        ], dim=0).unsqueeze(0)  # [1, context_len + gamma] on drafter_device
        full_mask = torch.ones_like(full_ids)  # on drafter_device

        logits = self.drafter_model(
            input_ids=full_ids,
            attention_mask=full_mask,
        ).logits.squeeze(0)  # [context_len + gamma, vocab_size] on drafter_device

        log_probs = F.log_softmax(logits.float(), dim=-1)  # [T, V]

        # Position i reads from slot (context_len - 1 + i): the logit at position t
        # predicts token t+1, so to predict position (context_len + i) we read t = context_len - 1 + i.
        position_indices = torch.arange(
            context_len - 1, context_len - 1 + gamma, device=self.drafter_device
        )  # [gamma]

        # candidate_matrix must be on drafter_device for advanced indexing
        gathered = log_probs[
            position_indices.unsqueeze(1),           # [gamma, 1]
            candidate_matrix.to(self.drafter_device),  # [gamma, K]
        ]  # [gamma, K] on drafter_device

        return gathered.to(device)  # move back to verifier_device

    # ── Gumbel-Top-k Selection ──────────────────────────────────────────

    def _gumbel_topk_verifier(
        self,
        target_logprobs: torch.Tensor,
        blade_scores: torch.Tensor,
        alpha: float,
        tau: float,
    ) -> torch.Tensor:
        """Select winners stochastically via exact Gumbel-Top-k sample."""
        # Normalize if configured
        if self.cfg.normalize_scores:
            target_logprobs = (target_logprobs - target_logprobs.mean(dim=1, keepdim=True)) / (target_logprobs.std(dim=1, keepdim=True) + 1e-6)
            blade_scores = (blade_scores - blade_scores.mean(dim=1, keepdim=True)) / (blade_scores.std(dim=1, keepdim=True) + 1e-6)

        # Fuse target fluency and blade alignment
        s = alpha * target_logprobs + (1.0 - alpha) * blade_scores  # [gamma, K]

        # Draw independent Gumbel(0,1) noise
        u = torch.rand_like(s).clamp(min=1e-20)
        g = -torch.log(-torch.log(u))  # [gamma, K]

        # Perturb and argmax
        s_perturbed = s / tau + g
        winners = s_perturbed.argmax(dim=1)  # [gamma]
        return winners

    # ── Main Generation Loop ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
        blade: Optional[str] = None,
    ):
        """Run GSI Strategy 5: Speculative Gumbel-Top-k with GSI Fallback.

        Parameters
        ----------
        prompt : str
            Input prompt text.
        max_new_tokens : int, optional
            Override cfg.max_new_tokens.
        verbose : bool
            Log per-round details.
        return_stats : bool
            If True, return (text, GSIGumbelStats) tuple instead of just text.
        blade : str | PeftModel | DPOBlade, optional
            Dynamic blade selection for inference.
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        gamma = self.cfg.gamma
        K = self.cfg.K
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        tau = self.cfg.gsi_tau
        threshold = self.cfg.gsi_threshold

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

        # Tokenize prompt
        encoded = self.verifier_tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        prompt_ids = encoded["input_ids"].to(self.verifier_device)   # [1, P]

        generated_ids: List[int] = []
        stats = GSIGumbelStats()
        t_start = time.perf_counter()

        while len(generated_ids) < max_tokens:
            stats.total_rounds += 1

            # Current context = prompt ⊕ y
            if generated_ids:
                gen_tensor = torch.tensor(
                    generated_ids, dtype=torch.long, device=self.verifier_device,
                ).unsqueeze(0)
                context_ids = torch.cat([prompt_ids, gen_tensor], dim=1)
            else:
                context_ids = prompt_ids  # [1, P]

            # ── Step 1: D ← draft.propose(...) → [γ, K] ────────────────
            candidate_matrix, draft_topk_logits = self._draft_propose(context_ids)
            stats.drafter_forward_passes += gamma

            # ── Step 2: Parallel scoring: verifier, blade, and drafter ──
            # Target logprobs (verifier)
            target_logprobs = active_blade.target_logprob_parallel(
                context_ids, candidate_matrix, self.verifier_model,
            )  # [γ, K]
            stats.target_forward_passes += 1

            # Blade scores
            blade_scores = active_blade.score_parallel(
                context_ids, candidate_matrix,
            )  # [γ, K]
            stats.blade_forward_passes += 1

            # Drafter logprobs (for tilted reward correction)
            drafter_logprobs = self._drafter_logprob_parallel(
                context_ids, candidate_matrix,
            )  # [γ, K]
            stats.drafter_forward_passes += 1

            # ── Step 3: Speculative Gumbel selection ───────────────────────
            winners = self._gumbel_topk_verifier(
                target_logprobs=target_logprobs,
                blade_scores=blade_scores,
                alpha=alpha,
                tau=tau,
            )  # [γ]

            accepted_prefix: List[int] = []
            rejection_occurred = False

            # ── Step 4: Left-to-right GSI Threshold Check & Fallback ──────
            for i in range(gamma):
                if len(generated_ids) + len(accepted_prefix) >= max_tokens:
                    break

                wi = int(winners[i].item())
                winner_token = int(candidate_matrix[i, wi].item())

                # Tilted reward: r_blade + (1 / beta) * (target_lp - draft_lp)
                # All three tensors are on verifier_device at this point.
                r_blade_val = blade_scores[i, wi].item()
                target_lp_val = target_logprobs[i, wi].item()
                draft_lp_val = drafter_logprobs[i, wi].item()
                tilted_reward = r_blade_val + (1.0 / beta) * (target_lp_val - draft_lp_val)

                if tilted_reward >= threshold:
                    # Cheap path: Gumbel winner passes the tilted-reward gate.
                    if wi != 0:
                        # Winner diverges from greedy — commit greedy prefix up to
                        # position i then the Gumbel winner token, then restart draft.
                        greedy_prefix = candidate_matrix[:i, 0].tolist()
                        accepted_prefix.extend(greedy_prefix)
                        accepted_prefix.append(winner_token)
                        rejection_occurred = True
                        stats.partial_accept_rounds += 1
                        stats.acceptance_positions.append(i)

                        if verbose:
                            logger.info(
                                "Round %d | Gumbel winner at pos %d (wi=%d) | Token: '%s'",
                                stats.total_rounds, i, wi,
                                self.verifier_tokenizer.decode(
                                    [winner_token], skip_special_tokens=True
                                )
                            )
                        break
                    # wi == 0: Gumbel agrees with greedy, continue to next position.
                else:
                    # Expensive fallback path: tilted reward < threshold.
                    # Resample K fresh tokens directly from verifier at position i,
                    # conditioned on the greedy prefix D[0:i, 0].
                    stats.fallback_rounds += 1
                    rejection_occurred = True
                    stats.partial_accept_rounds += 1
                    stats.acceptance_positions.append(i)

                    if verbose:
                        logger.info(
                            "Round %d | Rejected pos %d (tilted_r=%.4f < u=%.4f). "
                            "Falling back to verifier...",
                            stats.total_rounds, i, tilted_reward, threshold
                        )

                    # Build extended context: context + greedy tokens before position i
                    greedy_prefix_tokens = candidate_matrix[:i, 0].tolist()
                    if greedy_prefix_tokens:
                        extended_context = torch.cat([
                            context_ids.squeeze(0),
                            torch.tensor(
                                greedy_prefix_tokens, dtype=torch.long,
                                device=self.verifier_device
                            ),
                        ])  # [context_len + i]
                    else:
                        extended_context = context_ids.squeeze(0)  # i == 0

                    # 1 target forward pass to get top-K logits
                    outputs_fallback = self.verifier_model(
                        input_ids=extended_context.unsqueeze(0),
                        attention_mask=torch.ones(
                            (1, extended_context.shape[0]),
                            dtype=torch.long,
                            device=self.verifier_device,
                        ),
                    )
                    stats.target_forward_passes += 1
                    logits_i = outputs_fallback.logits[0, -1, :]  # [vocab_size]
                    _topk_logits, topk_ids = torch.topk(logits_i, K)  # [K]

                    # Score the K fresh tokens with the blade (no log-ratio correction
                    # needed — they come from the on-policy verifier distribution).
                    # score_parallel expects [gamma, K]; pass [1, K] and squeeze.
                    fresh_blade_scores = active_blade.score_parallel(
                        extended_context.unsqueeze(0),  # [1, context_len+i]
                        topk_ids.unsqueeze(0),          # [1, K]
                    ).squeeze(0)  # [K]
                    stats.blade_forward_passes += 1

                    # Optionally z-score normalise the fresh scores
                    if self.cfg.normalize_scores and fresh_blade_scores.std() > 1e-6:
                        fresh_blade_scores_norm = (
                            fresh_blade_scores - fresh_blade_scores.mean()
                        ) / fresh_blade_scores.std()
                    else:
                        fresh_blade_scores_norm = fresh_blade_scores

                    # Gumbel-Max sample on β * blade_score (no log-ratio on fallback)
                    u_fresh = torch.rand_like(fresh_blade_scores_norm).clamp(min=1e-20)
                    g_fresh = -torch.log(-torch.log(u_fresh))  # Gumbel(0,1)
                    perturbed_fresh = (beta * fresh_blade_scores_norm) / tau + g_fresh
                    j_star = int(perturbed_fresh.argmax().item())

                    fallback_winner_token = int(topk_ids[j_star].item())

                    # Commit: greedy prefix up to position i + fallback winner token
                    accepted_prefix.extend(greedy_prefix_tokens)
                    accepted_prefix.append(fallback_winner_token)

                    if verbose:
                        logger.info(
                            "Round %d | Fallback winner at pos %d | Token: '%s'",
                            stats.total_rounds, i,
                            self.verifier_tokenizer.decode(
                                [fallback_winner_token], skip_special_tokens=True
                            ),
                        )
                    break

            if not rejection_occurred:
                # Accept entire greedy draft D[:, 0]
                all_greedy = candidate_matrix[:, 0].tolist()
                accepted_prefix.extend(all_greedy)
                stats.full_accept_rounds += 1
                if verbose:
                    logger.info(
                        "Round %d | Full accept (%d tokens)",
                        stats.total_rounds, gamma
                    )

            # ── Commit accepted tokens ───────────────────────────────────
            remaining = max_tokens - len(generated_ids)
            accepted_prefix = accepted_prefix[:remaining]

            eos_id = self.verifier_tokenizer.eos_token_id
            eos_hit = False
            clean_tokens = []
            for tok in accepted_prefix:
                if tok == eos_id:
                    eos_hit = True
                    break
                clean_tokens.append(tok)

            generated_ids.extend(clean_tokens)
            stats.total_tokens_accepted += len(clean_tokens)

            if eos_hit:
                logger.info("EOS encountered. Stopping.")
                break

        stats.total_time_s = time.perf_counter() - t_start

        all_ids = prompt_ids.squeeze(0).tolist() + generated_ids
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "Generation complete | %d tokens | %.2fs | "
                "%d rounds (%.1f%% full-accept) | %.2f tok/s",
                stats.total_tokens_accepted,
                stats.total_time_s,
                stats.total_rounds,
                100 * stats.acceptance_rate,
                stats.tokens_per_second,
            )

        return (output_text, stats) if return_stats else output_text
