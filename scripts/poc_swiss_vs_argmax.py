"""Real-model POC: Swiss-system tournament vs blade-reward argmax.

This script runs the actual speculative generation loop (Option B) under
different verifier configurations using the truthfulness DPO blade.

It compares:
  1. Swiss-system tournament verifier
  2. Argmax-based verifier
  3. Shifted Swiss-system tournament verifier (+1000 blade reward shift)

It prints generated responses, detailed efficiency/acceptance statistics,
and prints the Chess-style round-by-round tournament bracket (Table 4.3.1).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This POC requires torch because it calls the real Swiss Knife models. "
        "Install requirements.txt or run it inside your Vast.ai ML environment."
    ) from exc

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model
from Model_mechanics.speculative_generator import SwissKnifeSpeculativeGenerator

logger = logging.getLogger(__name__)


def run_swiss_system_with_history(
    target_scores: torch.Tensor,
    blade_scores: torch.Tensor,
    alpha: float,
    rounds: int = 0,
    normalize: bool = True,
):
    """Run a Swiss-system tournament and return the history for printing."""
    K = target_scores.shape[0]
    if rounds == 0:
        rounds = max(1, math.ceil(math.log2(K)))

    # Z-score normalization
    if normalize:
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            return (t - t.mean()) / (t.std() + 1e-6)
        ts_norm = _znorm(target_scores)
        bs_norm = _znorm(blade_scores)
    else:
        ts_norm = target_scores
        bs_norm = blade_scores

    cum_wins = [0.0] * K
    paired_before = set()
    indices = list(range(K))

    # Detailed history for each round: list of dicts
    # Each round dict maps candidate_idx -> {"opponent": str, "result": str, "cum_wins": float}
    history = []

    for round_idx in range(rounds):
        # Sort by (cumulative wins DESC, original index ASC)
        sorted_by_score = sorted(
            indices,
            key=lambda i: (-cum_wins[i], i),
        )

        pairs = []
        unpaired = list(sorted_by_score)

        while len(unpaired) >= 2:
            a = unpaired[0]
            unpaired.pop(0)

            best_partner_pos = None
            for pos, b in enumerate(unpaired):
                pair_key = (min(a, b), max(a, b))
                if pair_key not in paired_before:
                    best_partner_pos = pos
                    break

            if best_partner_pos is None:
                best_partner_pos = 0

            b = unpaired.pop(best_partner_pos)
            pairs.append((a, b))
            paired_before.add((min(a, b), max(a, b)))

        round_history = {}
        # Handle bye
        if unpaired:
            bye_idx = unpaired[0]
            cum_wins[bye_idx] += 0.5
            round_history[bye_idx] = {
                "opponent": "BYE",
                "result": f"Bye ({cum_wins[bye_idx]})",
                "cum_wins": cum_wins[bye_idx],
            }

        # Execute matches
        for a, b in pairs:
            delta_target = ts_norm[a] - ts_norm[b]
            delta_blade  = bs_norm[a]  - bs_norm[b]
            score = alpha * delta_target + (1.0 - alpha) * delta_blade

            if score > 0:
                winner, loser = a, b
            else:
                winner, loser = b, a

            cum_wins[winner] += 1.0
            
            winner_score_str = f"{int(cum_wins[winner]) if cum_wins[winner].is_integer() else cum_wins[winner]}"
            loser_score_str = f"{int(cum_wins[loser]) if cum_wins[loser].is_integer() else cum_wins[loser]}"

            round_history[winner] = {
                "opponent": f"c{loser + 1}",
                "result": f"W ({winner_score_str})",
                "cum_wins": cum_wins[winner],
            }
            round_history[loser] = {
                "opponent": f"c{winner + 1}",
                "result": f"L ({loser_score_str})",
                "cum_wins": cum_wins[loser],
            }

        history.append(round_history)

    champion = max(
        indices,
        key=lambda i: (cum_wins[i], ts_norm[i].item()),
    )
    return champion, cum_wins, history


class TrackingSwissSpeculativeGenerator(SwissKnifeSpeculativeGenerator):
    """Subclass of speculative generator that intercepts the very first step
    to capture the tournament matches and scores for history tracking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_bracket = None
        self.current_candidates = None

    def _draft_propose(self, context_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_ids, draft_topk_logits = super()._draft_propose(context_ids)
        # Store the candidate IDs for the very first step/position
        if self.captured_bracket is None:
            self.current_candidates = candidate_ids[0].clone()
        return candidate_ids, draft_topk_logits

    def _swiss_at_position(self, target_scores_i: torch.Tensor, blade_scores_i: torch.Tensor) -> int:
        if self.captured_bracket is None and self.current_candidates is not None:
            champion, cum_wins, history = run_swiss_system_with_history(
                target_scores_i,
                blade_scores_i,
                alpha=self.cfg.alpha,
                rounds=self.cfg.swiss_rounds,
                normalize=self.cfg.normalize_scores,
            )
            # Decode the top-K candidate tokens for printing
            tokens = []
            for tid in self.current_candidates:
                tok_str = self.tokenizer.decode([tid.item()])
                # Remove extra spaces/quotes to keep table formatting clean
                tok_str = repr(tok_str).strip("'\"")
                tokens.append(tok_str)

            self.captured_bracket = {
                "champion": champion,
                "cum_wins": cum_wins,
                "history": history,
                "target_scores": target_scores_i.clone(),
                "blade_rewards": blade_scores_i.clone(),
                "tokens": tokens,
            }
            return champion
        return super()._swiss_at_position(target_scores_i, blade_scores_i)


class ArgmaxSpeculativeGenerator(SwissKnifeSpeculativeGenerator):
    """Speculative generator using the Argmax verifier selection baseline.

    Instead of running a pairwise tournament, the winner at position i is
    simply the candidate that maximizes:
        score_k = α * target_logprob_k + (1 - α) * blade_score_k
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_tournament = self._argmax_at_position

    def _argmax_at_position(
        self,
        target_scores_i: torch.Tensor,
        blade_scores_i: torch.Tensor,
    ) -> int:
        score = self.cfg.alpha * target_scores_i + (1.0 - self.cfg.alpha) * blade_scores_i
        return int(score.argmax().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-model POC comparing Swiss-system speculative generation with argmax selection.",
    )
    parser.add_argument("--blade", default="truthfulness", choices=["helpfulness", "harmlessness", "truthfulness"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", dest="normalize_scores", action="store_false", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--prompt", default="Q: What happens to you if you eat watermelon seeds?\nA:")
    parser.add_argument("--output", default="runs/poc_swiss_vs_argmax/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # We use a default lookahead gamma=4 and topK=8 for Option B
    cfg_swiss = SwissKnifeConfig(
        K=8,
        gamma=4,
        beta=args.beta,
        tournament_mode="swiss",
        swiss_rounds=args.rounds,
        normalize_scores=args.normalize_scores,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )

    print("=" * 96)
    print("Swiss-system Speculative Decoding POC vs Argmax Selection")
    print("=" * 96)
    print(f"Prompt: {args.prompt}")
    print(f"Blade:  {args.blade} | alpha={args.alpha} | beta={args.beta} | dtype={args.dtype}")
    print()

    logger.info("Loading tokenizer + base model...")
    tokenizer = load_tokenizer(cfg_swiss)
    base_model = load_base_model(cfg_swiss)
    logger.info("Loading blade: %s", args.blade)
    blade_model = load_blade_model(cfg_swiss, args.blade)

    # 1. Run Swiss-system Speculative Generation
    logger.info("Running Swiss-system Speculative Generation...")
    swiss_gen = TrackingSwissSpeculativeGenerator(
        cfg=cfg_swiss,
        tokenizer=tokenizer,
        base_model=base_model,
        blade_model=blade_model,
    )
    swiss_output, swiss_stats = swiss_gen.generate(
        prompt=args.prompt,
        verbose=True,
        return_stats=True,
    )

    # 2. Run Argmax-based Speculative Generation
    logger.info("Running Argmax Speculative Generation...")
    cfg_argmax = SwissKnifeConfig(
        K=8,
        gamma=4,
        beta=args.beta,
        tournament_mode="swiss",  # Ignored by ArgmaxSpeculativeGenerator
        swiss_rounds=args.rounds,
        normalize_scores=args.normalize_scores,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )
    argmax_gen = ArgmaxSpeculativeGenerator(
        cfg=cfg_argmax,
        tokenizer=tokenizer,
        base_model=base_model,
        blade_model=blade_model,
    )
    argmax_output, argmax_stats = argmax_gen.generate(
        prompt=args.prompt,
        verbose=True,
        return_stats=True,
    )

    # 3. Run Shifted Swiss-system Speculative Generation (+1000 blade bias)
    logger.info("Running Shifted Swiss-system Speculative Generation...")
    cfg_shifted = SwissKnifeConfig(
        K=8,
        gamma=4,
        beta=args.beta,
        tournament_mode="swiss",
        swiss_rounds=args.rounds,
        normalize_scores=args.normalize_scores,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        blade_bias=1000.0,
    )
    shifted_gen = SwissKnifeSpeculativeGenerator(
        cfg=cfg_shifted,
        tokenizer=tokenizer,
        base_model=base_model,
        blade_model=blade_model,
    )
    shifted_output, shifted_stats = shifted_gen.generate(
        prompt=args.prompt,
        verbose=True,
        return_stats=True,
    )

    # Print tournament schedule (Table 4.3.1 format) from step 1
    if swiss_gen.captured_bracket is not None:
        bracket = swiss_gen.captured_bracket
        R = len(bracket["history"])
        print("\n" + "=" * (20 + R * 20 + 6))
        print("Swiss-System Tournament Bracket at Step 1, Position 1 (Like Section 4.3.1 in PDF)")
        print("=" * (20 + R * 20 + 6))
        
        header_line1 = f"{'#':<4} {'Candidate':<15}" + "".join(f"{f'Round {r}':<20}" for r in range(1, R + 1)) + "Final"
        header_line2 = f"{'':<4} {'Token':<15}" + "".join(f"{'Opponent':<10}{'Result':<10}" for _ in range(R)) + "score"
        print(header_line1)
        print(header_line2)
        print("-" * (20 + R * 20 + 6))
        
        for idx in range(len(bracket["tokens"])):
            token_str = bracket["tokens"][idx]
            # Limit display length of candidate tokens
            if len(token_str) > 10:
                token_str = token_str[:8] + ".."
            row_str = f"{idx + 1:<4} c{idx + 1} ({token_str:<10})"
            for r in range(R):
                round_data = bracket["history"][r][idx]
                opp = round_data["opponent"]
                res = round_data["result"]
                row_str += f"{opp:<10}{res:<10}"
            
            final_score = bracket["cum_wins"][idx]
            final_score_str = f"{int(final_score) if final_score.is_integer() else final_score}"
            row_str += f" {final_score_str}"
            print(row_str)
            
        print("=" * (20 + R * 20 + 6))
        print()

    # Print comparative outputs
    print("\n" + "=" * 96)
    print("GENERATION RESULTS COMPARISON")
    print("=" * 96)
    print(f"1. Swiss System Speculative Output:\n{swiss_output}\n")
    print(f"2. Argmax Speculative Output:\n{argmax_output}\n")
    print(f"3. Shifted Swiss System Speculative Output (+1000 bias):\n{shifted_output}\n")
    print("=" * 96)

    # Print comparative statistics
    print("\n" + "=" * 96)
    print("SPECULATIVE DECODING STATISTICS COMPARISON")
    print("=" * 96)
    print(
        f"{'Metric':<30} | "
        f"{'Swiss System':>15} | "
        f"{'Argmax':>15} | "
        f"{'Shifted Swiss':>15}"
    )
    print("-" * 96)
    
    s_stats = swiss_stats.to_dict()
    a_stats = argmax_stats.to_dict()
    sh_stats = shifted_stats.to_dict()

    metrics = [
        ("total_rounds", "Total Rounds (cycles)", "{:.0f}"),
        ("total_tokens_accepted", "Total Tokens Accepted", "{:.0f}"),
        ("full_accept_rounds", "Full Accept Rounds", "{:.0f}"),
        ("partial_accept_rounds", "Partial Accept Rounds", "{:.0f}"),
        ("acceptance_rate", "Acceptance Rate", "{:.2%}"),
        ("target_forward_passes", "Target Forward Passes", "{:.0f}"),
        ("blade_forward_passes", "Blade Forward Passes", "{:.0f}"),
        ("tournament_calls", "Tournament Calls", "{:.0f}"),
        ("tokens_per_second", "Tokens / Second", "{:.2f}"),
        ("total_time_s", "Total Time (s)", "{:.3f}s"),
    ]

    for key, label, fmt in metrics:
        v_s = fmt.format(s_stats[key])
        v_a = fmt.format(a_stats[key])
        v_sh = fmt.format(sh_stats[key])
        print(f"{label:<30} | {v_s:>15} | {v_a:>15} | {v_sh:>15}")

    print("=" * 96)
    print(f"Shift Invariance Check (Standard Swiss vs. Shifted Swiss):")
    is_invariant = (swiss_output == shifted_output)
    print(f"  * Outputs match exactly: {is_invariant}")
    print(f"  * Status: {'PASSED (Invariant to constant shift)' if is_invariant else 'FAILED'}")
    print("=" * 96)

    # Save results to JSON
    output_data = {
        "prompt": args.prompt,
        "blade": args.blade,
        "alpha": args.alpha,
        "beta": args.beta,
        "dtype": args.dtype,
        "device": args.device,
        "rounds": args.rounds,
        "normalize_scores": args.normalize_scores,
        "swiss": {
            "output": swiss_output,
            "stats": s_stats,
            "captured_bracket": {
                "cum_wins": bracket["cum_wins"],
                "tokens": bracket["tokens"],
                "target_scores": bracket["target_scores"].tolist(),
                "blade_rewards": bracket["blade_rewards"].tolist(),
                "history": bracket["history"]
            } if swiss_gen.captured_bracket is not None else None
        },
        "argmax": {
            "output": argmax_output,
            "stats": a_stats,
        },
        "shifted_swiss": {
            "output": shifted_output,
            "stats": sh_stats,
        },
        "shift_invariant": is_invariant,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\nSaved results JSON to: {out_path}")


if __name__ == "__main__":
    main()
