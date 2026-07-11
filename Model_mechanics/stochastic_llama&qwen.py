"""
Swiss Knife — Stochastic LLaMA & Qwen Generator
===============================================

Implements Token-Level Speculative Tournament Decoding (Algorithm 2 / Option B
from swiss_knife_analysis.pdf), with per-match stochastic perturbations from
StochasticAuditor and GSI tilted-reward thresholding.

Architecture:
  - Drafter:  LLaMA 3.2 3B  (πS)
  - Verifier: Qwen 2.5 7B   (πB / πtarget)
  - Blade:    Qwen DPO adapter (r_blade = β * log πblade/πref)

Algorithm per outer iteration (until max_tokens or EOS):
  1. LLaMA drafts γ tokens. We extract the full score tensor at each of the
     γ positions, giving us the top-K candidate token IDs and their log-probs.

  2. Build candidate_matrix [γ, K] — top-K LLaMA token IDs at each step.
     Row 0 of each position holds the greedy draft token.

  3. Retokenise: decode every candidate token ID to a text string using the
     LLaMA tokeniser, then re-encode with the Qwen tokeniser to produce Qwen
     token ID sequences.  Because the vocabularies differ, 1 LLaMA token may
     map to 0, 1, or 2+ Qwen tokens.

  4. PARALLEL scoring (2 Qwen forward passes total per outer iteration):
       - blade.score_parallel(context, candidate_matrix_qwen)  → [γ, K] blade rewards
         (using the greedy Qwen-token path to build the shared prefix for each row)
       - blade.target_logprob_parallel(context, candidate_matrix_qwen, verifier_model)
         → [γ, K] verifier log-probs

  5. Process positions i = 0 … γ-1 sequentially (must be sequential because
     accepting/rejecting at position i determines whether we continue to i+1):
       a. At position i, run a Knockout Bracket over the K candidates.
          - For EACH match (a, b) inside the bracket: draw a fresh stochastic
            functional → re-score ONLY candidates a and b under that functional
            → compute Match(a, b); clear functional.
          - Match formula (§4.4):
              Match(A, B) = α·[log πverifier(A) − log πverifier(B)]
                          + (1−α)·[r_blade(A)   − r_blade(B)]
          - A wins iff Match(A, B) > 0.

       b. Compute tilted reward for the bracket winner w:
              r̃(w) = r_blade(w) + (1/β)·[log πverifier(w) − log πdraft(w)]
          where r_blade(w) is the CLEAN (unperturbed) blade reward for w,
          log πverifier(w) is the Qwen verifier log-prob, and
          log πdraft(w) is the LLaMA draft log-prob for that candidate.

       c. If r̃(w) ≥ u (threshold):
            Accept w, append its Qwen tokens, update context.
            Continue to position i+1.
          Else:
            Reject. Fallback: generate 1 token from Qwen autoregressively.
            Append fallback token. Discard remainder of γ draft. Restart outer loop.

Parallelism note
----------------
The blade / verifier SCORING of all γ×K candidates is fully parallel (done in
steps 3–4 above, before the sequential acceptance loop in step 5).  The per-match
stochastic perturbation in the knockout bracket must score ONLY the two candidates
in that match (so the hook applies only to their evaluation), but we can cache the
pre-computed clean scores from step 4 for the Match formula and use the stochastic
hook only for re-scoring in each match.

This gives us:
  - 2 "bulk" forward passes (blade + verifier) per outer iteration for all γK candidates.
  - Exactly K−1 stochastic matches per position. Since each match scores 2 candidates 
    (A and B), we perform K−1 batched forward passes (batch size 2) through the blade.
    This means 2×(K−1) candidate sequences are scored stochastically per position.
    This is unavoidable because each match needs a DIFFERENT random functional.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.blades import DPOBlade
from Model_mechanics.stochastic_auditor import StochasticAuditor, StochasticAuditorConfig
from evaluation.retokenisation_llama_to_qwen import compute_logprob, retokenize_step

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StochasticLlamaQwenStats:
    """Tracks statistics over a single prompt generation."""
    total_outer_iterations: int = 0   # Number of γ-draft restarts
    total_tokens: int = 0             # Total Qwen tokens generated
    accepted_positions: int = 0       # Positions where winner passed threshold
    rejected_positions: int = 0       # Positions where threshold failed (fallback)
    total_candidates_scored: int = 0  # Total per-match stochastic scores
    total_pairwise_matches: int = 0   # Total match evaluations in brackets
    total_time_s: float = 0.0
    step_tilted_rewards: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_positions + self.rejected_positions
        if total == 0:
            return 0.0
        return self.accepted_positions / total

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s < 1e-6:
            return 0.0
        return self.total_tokens / self.total_time_s

    def to_dict(self) -> dict:
        return {
            "total_outer_iterations": self.total_outer_iterations,
            "total_tokens": self.total_tokens,
            "accepted_positions": self.accepted_positions,
            "rejected_positions": self.rejected_positions,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_candidates_scored": self.total_candidates_scored,
            "total_pairwise_matches": self.total_pairwise_matches,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time_s": round(self.total_time_s, 3),
            "mean_tilted_reward": round(
                sum(self.step_tilted_rewards) / max(len(self.step_tilted_rewards), 1), 6
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class StochasticLlamaQwenGenerator:
    """Token-Level Stochastic Tournament Decoding (LLaMA drafter + Qwen verifier).

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Pipeline config (gamma, K=gsi_n, alpha, beta, gsi_threshold, etc.)
    drafter_model : LLaMA 3.2 3B (πS)
    drafter_tokenizer : LLaMA tokenizer
    verifier_model : Qwen 2.5 7B (πB / πtarget)
    verifier_tokenizer : Qwen tokenizer
    blade_model : Qwen DPO LoRA adapter
    auditor_cfg : Which stochastic mode to use (mc_dropout | random_proj | head_subsample)
                  Set mode=None for the deterministic baseline.
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        drafter_model: PreTrainedModel,
        drafter_tokenizer: PreTrainedTokenizer,
        verifier_model: PreTrainedModel,
        verifier_tokenizer: PreTrainedTokenizer,
        blade_model: PeftModel,
        auditor_cfg: StochasticAuditorConfig,
    ):
        self.cfg = cfg
        self.drafter_model = drafter_model
        self.drafter_tokenizer = drafter_tokenizer
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer
        self.blade_model = blade_model

        self.blade = DPOBlade(cfg, verifier_model, blade_model, verifier_tokenizer)
        self.auditor = StochasticAuditor(self.blade, cfg, auditor_cfg)

        self.drafter_device = next(drafter_model.parameters()).device
        self.verifier_device = next(verifier_model.parameters()).device
        self.use_stochastic = (auditor_cfg.mode is not None)

        logger.info(
            "StochasticLlamaQwenGenerator | γ=%d K=%d α=%.2f β=%.3f u=%.3f mode=%s",
            cfg.gamma, cfg.gsi_n, cfg.alpha, cfg.beta, cfg.gsi_threshold,
            auditor_cfg.mode or "NONE (baseline)",
        )

    # ── Fallback ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _fallback_generate_one(
        self, qwen_prefix_ids: torch.Tensor
    ) -> Tuple[List[int], str]:
        """Fallback to Qwen+Blade: generate the greedy token from the blade model.
        This is exactly equivalent to strategy 1 (baseline_argmax_harmlessness) 
        for a single token step, ignoring the LLaMA drafter entirely.
        """
        # 1. Get next-token logits directly from the Blade model (Qwen 7B + DPO adapter)
        blade_outputs = self.blade_model(
            input_ids=qwen_prefix_ids,
            attention_mask=torch.ones_like(qwen_prefix_ids)
        )
        blade_logits = blade_outputs.logits[0, -1, :]  # [vocab_size]
        
        # 2. Select the token with the highest probability
        best_idx = torch.argmax(blade_logits).item()
        
        text = self.verifier_tokenizer.decode([best_idx], skip_special_tokens=True)
        return [best_idx], text

    # ── Retokenisation ────────────────────────────────────────────────────────

    def _retokenize_candidates(
        self,
        llama_token_ids: List[int],   # K LLaMA token IDs for one position
        qwen_prefix_text: str,
        qwen_prefix_ids: torch.Tensor,  # [1, qwen_prefix_len]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Map K LLaMA-vocab token IDs → K lists of Qwen-vocab token IDs.

        Steps for each candidate k:
          1. Decode the single LLaMA token ID to a string fragment.
          2. Concatenate with qwen_prefix_text.
          3. Re-encode with the Qwen tokenizer; take only the NEW token IDs
             beyond the existing qwen_prefix_ids using retokenize_step().

        Returns
        -------
        qwen_seq_ids_list : list of K 1D tensors (Qwen token IDs for the fragment)
        step_texts : list of K decoded strings
        """
        qwen_seq_ids_list = []
        step_texts = []

        for token_id in llama_token_ids:
            text_str = self.drafter_tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            step_texts.append(text_str)

            qwen_ids = retokenize_step(
                self.verifier_tokenizer,
                qwen_prefix_text,
                text_str,
                qwen_prefix_ids.squeeze(0),
                self.verifier_device,
            )
            qwen_seq_ids_list.append(qwen_ids)

        return qwen_seq_ids_list, step_texts

    # ── Knockout Bracket (per-position) ──────────────────────────────────────

    def _run_knockout_bracket(
        self,
        K: int,
        clean_verifier_logprobs: torch.Tensor,   # [K]
        clean_blade_rewards: torch.Tensor,        # [K]  (pre-computed, unperturbed)
        qwen_prefix_ids: torch.Tensor,            # [1, prefix_len]
        candidate_matrix_qwen: torch.Tensor,      # [gamma, K]
        step_idx: int,
        ref_logprobs_matrix: torch.Tensor,        # [gamma, K]
        alpha: float,
        stats: StochasticLlamaQwenStats,
    ) -> int:
        """Single-elimination knockout bracket with per-match stochastic perturbations.

        For each match (a, b):
          - If use_stochastic: draw a fresh functional, re-score candidates
            with the perturbed blade, clear functional.
          - If not use_stochastic: use the pre-computed clean blade rewards.
          - Compute Match(a, b) = α·Δverifier + (1-α)·Δblade.
          - A wins iff score > 0.

        Returns the index of the overall bracket winner.
        """
        active = list(range(K))

        while len(active) > 1:
            next_round: List[int] = []

            for i in range(0, len(active) - 1, 2):
                a, b = active[i], active[i + 1]

                if self.use_stochastic:
                    # Per-match perturbation: score under a fresh stochastic functional.
                    self.auditor.draw_fresh_functional()
                    stoch_rewards = self.auditor.score_candidates_for_match(
                        qwen_prefix_ids,
                        candidate_matrix_qwen,
                        ref_logprobs_matrix,
                    )  # [gamma, K]
                    self.auditor.clear_functional()
                    blade_a = stoch_rewards[step_idx, a]
                    blade_b = stoch_rewards[step_idx, b]
                    stats.total_candidates_scored += 2
                else:
                    blade_a = clean_blade_rewards[a]
                    blade_b = clean_blade_rewards[b]

                delta_verifier = clean_verifier_logprobs[a] - clean_verifier_logprobs[b]
                delta_blade = blade_a - blade_b
                score = alpha * delta_verifier + (1.0 - alpha) * delta_blade

                winner = a if score > 0 else b
                next_round.append(winner)
                stats.total_pairwise_matches += 1

            # Bye: odd candidate advances automatically
            if len(active) % 2 == 1:
                next_round.append(active[-1])

            active = next_round

        return active[0]

    # ── Main generate loop ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        """Generate a response for `prompt` using stochastic tournament decoding.

        Parameters
        ----------
        prompt : str
        max_new_tokens : int, optional — overrides cfg.max_new_tokens
        verbose : bool — log per-step detail
        return_stats : bool — if True, return (text, StochasticLlamaQwenStats)

        Returns
        -------
        str | (str, StochasticLlamaQwenStats)
        """
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        K       = self.cfg.gsi_n      # candidates per position
        gamma   = self.cfg.gamma      # lookahead depth
        alpha   = self.cfg.alpha
        beta    = self.cfg.beta
        threshold = self.cfg.gsi_threshold  # tilted-reward acceptance threshold u

        llama_prefix_text = prompt
        qwen_prefix_text  = prompt

        generated_qwen_tokens: List[int] = []
        stats  = StochasticLlamaQwenStats()
        t_start = time.perf_counter()

        # Keep Qwen prefix as tensor so we can efficiently extend it
        qwen_prefix_ids = self.verifier_tokenizer(
            prompt, return_tensors="pt"
        )["input_ids"].to(self.verifier_device)   # [1, prompt_len]

        eos_id = self.verifier_tokenizer.eos_token_id

        # ─────────────────────────────────────────────────────────────────────
        # Outer loop: draft γ tokens, verify, accept/reject position-by-position
        # ─────────────────────────────────────────────────────────────────────
        while len(generated_qwen_tokens) < max_tokens:
            stats.total_outer_iterations += 1

            # ── Step 1: LLaMA drafts γ tokens ────────────────────────────────
            llama_encoded = self.drafter_tokenizer(
                llama_prefix_text, return_tensors="pt"
            )
            llama_prefix_ids = llama_encoded["input_ids"].to(self.drafter_device)

            draft_out = self.drafter_model.generate(
                input_ids=llama_prefix_ids,
                attention_mask=torch.ones_like(llama_prefix_ids),
                max_new_tokens=gamma,
                do_sample=False,        # greedy draft; top-K extracted from scores
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=(
                    self.drafter_tokenizer.pad_token_id
                    or self.drafter_tokenizer.eos_token_id
                ),
            )
            # draft_out.scores: tuple of γ tensors, each [1, llama_vocab_size]
            draft_scores_tuple = draft_out.scores
            actual_gamma = len(draft_scores_tuple)   # may be < gamma at end-of-sequence

            if actual_gamma == 0:
                logger.info("LLaMA produced 0 draft tokens; stopping.")
                break

            # ── Step 2: Build [γ, K] candidate matrices (LLaMA vocab) ─────────
            # For each draft position: extract top-K token IDs and their log-probs.
            llama_topk_ids   = []    # list of γ tensors [K] — LLaMA token IDs
            llama_topk_lps   = []    # list of γ tensors [K] — LLaMA log-probs

            for step_scores in draft_scores_tuple:
                # step_scores: [1, llama_vocab_size]  (raw logits from generate)
                logprobs = F.log_softmax(step_scores[0], dim=-1)
                topk_lp, topk_ids = torch.topk(logprobs, K)
                llama_topk_ids.append(topk_ids)       # [K]
                llama_topk_lps.append(topk_lp)        # [K]

            # ── Step 3: Retokenise all candidates to Qwen vocab ───────────────
            # We process each of the γ positions.  The Qwen prefix text is
            # the CURRENT accepted prefix (not updated until a token is accepted).
            # Within a single γ-draft all positions share the SAME current prefix.
            all_qwen_seq_ids: List[List[torch.Tensor]] = []
            all_step_texts:   List[List[str]] = []

            for step_idx in range(actual_gamma):
                llama_ids_k = llama_topk_ids[step_idx].tolist()  # K ints
                qwen_seqs, texts = self._retokenize_candidates(
                    llama_ids_k, qwen_prefix_text, qwen_prefix_ids
                )
                all_qwen_seq_ids.append(qwen_seqs)   # [K] Qwen tensors
                all_step_texts.append(texts)

            # ── Step 4: PARALLEL scoring — 2 Qwen forward passes ─────────────
            # We score ALL γ×K candidates in one blade pass and one verifier pass.
            # Both score_parallel and target_logprob_parallel take a [γ, K]
            # candidate_matrix of token IDs over the GREEDY Qwen prefix path.
            # Strategy: use the Qwen greedy-top-1 token at each step as the
            # "greedy path" token (candidate_matrix[:, 0]) so position-specific
            # logits are correctly conditioned.
            # Build candidate_matrix [γ, K] from the Qwen retokenized top-1 IDs.
            # For positions where a LLaMA token maps to multiple Qwen tokens, we
            # use the FIRST Qwen token of that mapping as the representative.
            candidate_matrix_qwen = torch.zeros(
                actual_gamma, K, dtype=torch.long, device=self.verifier_device
            )
            for step_idx in range(actual_gamma):
                for k in range(K):
                    seq = all_qwen_seq_ids[step_idx][k]
                    if seq.numel() > 0:
                        candidate_matrix_qwen[step_idx, k] = seq[0]
                    else:
                        # Empty mapping: use EOS as placeholder (will score low)
                        candidate_matrix_qwen[step_idx, k] = eos_id

            # Clean (unperturbed) blade rewards and verifier log-probs [γ, K]
            clean_blade_rewards_matrix = self.blade.score_parallel(
                qwen_prefix_ids, candidate_matrix_qwen
            )  # [γ, K]

            clean_verifier_lp_matrix = self.blade.target_logprob_parallel(
                qwen_prefix_ids, candidate_matrix_qwen, self.verifier_model
            )  # [γ, K]

            # ── Step 5: Sequential position acceptance ────────────────────────
            restart_draft = False

            if self.use_stochastic:
                self.auditor.precompute_hidden_states(
                    qwen_prefix_ids, candidate_matrix_qwen
                )

            for step_idx in range(actual_gamma):
                qwen_seqs = all_qwen_seq_ids[step_idx]   # K Qwen token-id tensors
                texts     = all_step_texts[step_idx]      # K text strings

                verifier_lp_k = clean_verifier_lp_matrix[step_idx]  # [K]
                blade_reward_k = clean_blade_rewards_matrix[step_idx]  # [K]

                # LLaMA log-probs for this position [K]
                draft_lp_k = llama_topk_lps[step_idx].to(self.verifier_device)

                # ── Knockout Bracket ─────────────────────────────────────────
                winner_k = self._run_knockout_bracket(
                    K=K,
                    clean_verifier_logprobs=verifier_lp_k,
                    clean_blade_rewards=blade_reward_k,
                    qwen_prefix_ids=qwen_prefix_ids,
                    candidate_matrix_qwen=candidate_matrix_qwen,
                    step_idx=step_idx,
                    ref_logprobs_matrix=clean_verifier_lp_matrix,
                    alpha=alpha,
                    stats=stats,
                )

                # ── Tilted Reward (GSI §4, eq. r̃) ───────────────────────────
                # r̃(w) = r_blade(w) + (1/β) * [log πverifier(w) − log πdraft(w)]
                # r_blade uses the CLEAN (unperturbed) blade reward from step 4.
                winner_blade_r   = blade_reward_k[winner_k].item()
                winner_verifier_lp = verifier_lp_k[winner_k].item()
                winner_draft_lp    = draft_lp_k[winner_k].item()

                tilted_reward = winner_blade_r + (1.0 / beta) * (
                    winner_verifier_lp - winner_draft_lp
                )

                if verbose:
                    logger.info(
                        "  pos=%d winner=%d r_blade=%.4f lp_v=%.4f lp_d=%.4f "
                        "r̃=%.4f thresh=%.4f %s",
                        step_idx, winner_k, winner_blade_r, winner_verifier_lp,
                        winner_draft_lp, tilted_reward, threshold,
                        "ACCEPT" if tilted_reward >= threshold else "REJECT",
                    )

                # ── Accept / Reject ──────────────────────────────────────────
                if tilted_reward >= threshold:
                    # Accept: append the winner's Qwen tokens
                    winner_qwen_ids = qwen_seqs[winner_k].tolist()
                    winner_text     = texts[winner_k]

                    generated_qwen_tokens.extend(winner_qwen_ids)
                    stats.total_tokens += len(winner_qwen_ids)
                    stats.accepted_positions += 1
                    stats.step_tilted_rewards.append(tilted_reward)

                    # Update both text prefixes and the Qwen tensor prefix
                    llama_prefix_text += winner_text
                    qwen_prefix_text  += winner_text
                    winner_ids_t = torch.tensor(
                        winner_qwen_ids, dtype=torch.long, device=self.verifier_device
                    ).unsqueeze(0)  # [1, n_new_qwen_tokens]
                    qwen_prefix_ids = torch.cat(
                        [qwen_prefix_ids, winner_ids_t], dim=1
                    )  # [1, extended_len]

                    # Stop if EOS was generated
                    if eos_id in winner_qwen_ids:
                        restart_draft = True
                        break

                    # Stop if we've hit max_tokens
                    if len(generated_qwen_tokens) >= max_tokens:
                        restart_draft = True
                        break

                else:
                    # Reject: fallback to Qwen for 1 token, restart draft
                    stats.rejected_positions += 1

                    fb_ids, fb_text = self._fallback_generate_one(qwen_prefix_ids)
                    generated_qwen_tokens.extend(fb_ids)
                    stats.total_tokens += len(fb_ids)

                    llama_prefix_text += fb_text
                    qwen_prefix_text  += fb_text
                    fb_ids_t = torch.tensor(
                        fb_ids, dtype=torch.long, device=self.verifier_device
                    ).unsqueeze(0)
                    qwen_prefix_ids = torch.cat(
                        [qwen_prefix_ids, fb_ids_t], dim=1
                    )

                    if eos_id in fb_ids:
                        restart_draft = True
                    else:
                        restart_draft = True  # Always restart after rejection

                    break  # Exit γ-loop; outer while will check max_tokens

            if self.use_stochastic:
                self.auditor.clear_precomputed()

            # Check termination after processing all positions (or early exit)
            if eos_id in generated_qwen_tokens:
                break
            if len(generated_qwen_tokens) >= max_tokens:
                break

        stats.total_time_s = time.perf_counter() - t_start

        # Decode final output
        initial_prompt_ids = self.verifier_tokenizer(
            prompt, return_tensors="pt"
        )["input_ids"].squeeze(0).tolist()
        all_ids = initial_prompt_ids + generated_qwen_tokens
        output_text = self.verifier_tokenizer.decode(all_ids, skip_special_tokens=True)

        if verbose:
            logger.info(
                "Generation complete | %d tokens | %.2fs | acceptance=%.1f%%",
                stats.total_tokens, stats.total_time_s, 100 * stats.acceptance_rate,
            )

        return (output_text, stats) if return_stats else output_text
