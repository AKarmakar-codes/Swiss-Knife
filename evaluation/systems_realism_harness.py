"""
Swiss Knife — Phase 4: Systems Realism Harness
===============================================

Measures wall-clock performance of Option A vs. Option B generation loops.

Metrics tracked:
  • tokens/sec           — throughput
  • acceptance_rate      — fraction of speculative rounds with full γ-accept
  • auditor_calls/token  — blade forward passes per generated token
  • target_passes/token  — target model forward passes per generated token
  • tournament_calls     — total per-position tournaments run
  • total_time_s         — end-to-end wall clock time

This harness code-only; no runs are executed by default (dry_run=True).
Set dry_run=False to actually generate text and measure.
"""

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class SystemsProfile:
    """Performance profile for one generation configuration."""
    mode: str                  # "option_a" or "option_b"
    blade: str
    prompt: str
    total_tokens: int = 0
    total_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    acceptance_rate: float = 0.0
    auditor_calls_per_token: float = 0.0
    target_passes_per_token: float = 0.0
    tournament_calls: int = 0
    gamma: int = 0
    K: int = 0
    tournament_mode: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "blade": self.blade,
            "prompt": self.prompt[:60],
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time_s, 3),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "auditor_calls_per_token": round(self.auditor_calls_per_token, 4),
            "target_passes_per_token": round(self.target_passes_per_token, 4),
            "tournament_calls": self.tournament_calls,
            "gamma": self.gamma,
            "K": self.K,
            "tournament_mode": self.tournament_mode,
        }


class SystemsRealismHarness:
    """Evaluation harness for Axis 3: Systems Realism.

    Compares Option A vs. Option B on throughput and auditor efficiency
    metrics. All measurement is done by capturing the SpeculativeStats
    object returned by SwissKnifeSpeculativeGenerator.generate(..., return_stats=True).

    Parameters
    ----------
    option_a_factory : callable
        (blade_name: str) → SwissKnifeGenerator (Option A)
    option_b_factory : callable
        (blade_name: str) → SwissKnifeSpeculativeGenerator (Option B)
    output_dir : str
    """

    SYSTEMS_PROMPTS = [
        "Explain the theory of relativity in simple terms.",
        "What are the pros and cons of renewable energy?",
        "Write a short paragraph about the history of the internet.",
        "Describe the water cycle.",
        "What is machine learning and how does it work?",
    ]

    def __init__(
        self,
        option_a_factory,
        option_b_factory,
        output_dir: str = "evaluation_results/systems",
    ):
        self.option_a_factory = option_a_factory
        self.option_b_factory = option_b_factory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        blade_names: List[str],
        max_new_tokens: int = 100,
        dry_run: bool = True,
    ) -> List[SystemsProfile]:
        """Run the systems realism evaluation.

        Parameters
        ----------
        blade_names : list of str
        max_new_tokens : int
        dry_run : bool
            If True (default), skip actual generation and return stub profiles.
            Set False to generate real text and measure.

        Returns
        -------
        list of SystemsProfile
        """
        all_profiles: List[SystemsProfile] = []

        for blade_name in blade_names:
            for prompt in self.SYSTEMS_PROMPTS:
                # ── Option A ───────────────────────────────────────────
                if dry_run:
                    profile_a = self._stub_profile("option_a", blade_name, prompt)
                else:
                    gen_a = self.option_a_factory(blade_name)
                    t0 = time.perf_counter()
                    text_a = gen_a.generate(prompt, max_new_tokens=max_new_tokens)
                    elapsed_a = time.perf_counter() - t0
                    n_tokens_a = len(text_a.split())  # rough estimate
                    profile_a = SystemsProfile(
                        mode="option_a",
                        blade=blade_name,
                        prompt=prompt,
                        total_tokens=n_tokens_a,
                        total_time_s=elapsed_a,
                        tokens_per_sec=n_tokens_a / max(elapsed_a, 1e-6),
                        acceptance_rate=1.0,  # Option A always "accepts" its winner
                        auditor_calls_per_token=1.0,  # one blade pass per span
                        target_passes_per_token=0.0,  # no separate target in Option A
                        tournament_calls=n_tokens_a,
                    )
                all_profiles.append(profile_a)

                # ── Option B ───────────────────────────────────────────
                if dry_run:
                    profile_b = self._stub_profile("option_b", blade_name, prompt)
                else:
                    gen_b = self.option_b_factory(blade_name)
                    text_b, stats_b = gen_b.generate(
                        prompt, max_new_tokens=max_new_tokens, return_stats=True
                    )
                    profile_b = SystemsProfile(
                        mode="option_b",
                        blade=blade_name,
                        prompt=prompt,
                        total_tokens=stats_b.total_tokens_accepted,
                        total_time_s=stats_b.total_time_s,
                        tokens_per_sec=stats_b.tokens_per_second,
                        acceptance_rate=stats_b.acceptance_rate,
                        auditor_calls_per_token=stats_b.auditor_calls_per_token,
                        target_passes_per_token=(
                            stats_b.target_forward_passes
                            / max(stats_b.total_tokens_accepted, 1)
                        ),
                        tournament_calls=stats_b.tournament_calls,
                        gamma=gen_b.cfg.gamma,
                        K=gen_b.cfg.K,
                        tournament_mode=gen_b.cfg.tournament_mode,
                    )
                all_profiles.append(profile_b)

        self._save_results(all_profiles)
        self._print_summary(all_profiles)
        return all_profiles

    def _stub_profile(self, mode: str, blade: str, prompt: str) -> SystemsProfile:
        """Return a stub (dry-run) profile with placeholder values."""
        return SystemsProfile(
            mode=mode, blade=blade, prompt=prompt,
            total_tokens=0, total_time_s=0.0,
            tokens_per_sec=float("nan"),
            acceptance_rate=float("nan"),
            auditor_calls_per_token=float("nan"),
            target_passes_per_token=float("nan"),
            tournament_calls=0,
        )

    def _save_results(self, profiles: List[SystemsProfile]):
        path = os.path.join(self.output_dir, "systems_results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(SystemsProfile.__dataclass_fields__.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in profiles:
                writer.writerow({
                    k: getattr(p, k) for k in fieldnames
                })
        logger.info("Systems CSV saved to: %s", path)

    def _print_summary(self, profiles: List[SystemsProfile]):
        print("\n" + "═" * 80)
        print("  Systems Realism Harness — Summary")
        print("═" * 80)
        print(
            f"  {'Mode':10s} | {'Blade':15s} | {'tok/s':>8} | "
            f"{'acc_rate':>9} | {'aud/tok':>7} | {'tgt/tok':>7}"
        )
        print("─" * 80)
        for p in profiles:
            print(
                f"  {p.mode:10s} | {p.blade:15s} | "
                f"{p.tokens_per_sec:>8.2f} | "
                f"{p.acceptance_rate:>9.3f} | "
                f"{p.auditor_calls_per_token:>7.4f} | "
                f"{p.target_passes_per_token:>7.4f}"
            )
        print("═" * 80 + "\n")
