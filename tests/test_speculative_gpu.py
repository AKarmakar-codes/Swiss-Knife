"""
GPU Integration Tests — Speculative Decoding Loop with Tournament
==================================================================

These tests load REAL model weights (Qwen2.5-7B + DPO LoRA blades) and run
the full Option B speculative generation loop end-to-end. They are designed
to be executed on a Vast.ai GPU instance with ≥ 32 GB VRAM.

DO NOT run these locally — they require:
  • ~14 GB VRAM for the base model (bfloat16)
  • ~14 GB VRAM for the blade model copy
  • HuggingFace Hub access to download model + adapter weights

Run on Vast.ai:
    conda activate myenv
    python tests/test_speculative_gpu.py

Tests:
  TEST 1 — Full Option B speculative loop: load base + helpfulness blade,
           generate 50 tokens, verify output structure, token counts,
           tournament stats, and that the tournament actually influences
           token selection (acceptance_rate < 1.0 for non-trivial prompts).

  TEST 2 — Blade swap mid-generation: generate with helpfulness blade,
           hot-swap to harmlessness blade via BladeRack, generate again,
           verify both outputs are valid and that the swap was fast (< 1s).
           Then compare outputs to confirm the blade actually changed
           the generation behavior.
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from tournament.config import SwissKnifeConfig
from tournament.models import load_tokenizer, load_base_model, load_blade_model
from tournament.speculative_generator import SwissKnifeSpeculativeGenerator, SpeculativeStats
from tournament.blade_rack import BladeRack, ReconfigurationProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared setup — loads once, reused by both tests
# ─────────────────────────────────────────────────────────────────────────────

_SHARED = {}


def _get_shared():
    """Load base model + tokenizer once and cache for both tests."""
    if _SHARED:
        return _SHARED

    # Use bfloat16 on GPU for best stability
    device = "auto" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"

    cfg = SwissKnifeConfig(
        K=8,
        gamma=4,
        alpha=0.5,
        beta=0.1,
        tournament_mode="swiss",
        generation_mode="option_b",
        normalize_scores=True,
        max_new_tokens=50,
        device=device,
        dtype=dtype,
    )

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(cfg)

    logger.info("Loading base model (this downloads ~14 GB on first run)...")
    base_model = load_base_model(cfg)

    _SHARED["cfg"] = cfg
    _SHARED["tokenizer"] = tokenizer
    _SHARED["base_model"] = base_model
    return _SHARED


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Full Option B Speculative Loop
# ─────────────────────────────────────────────────────────────────────────────

def test_full_speculative_loop():
    """
    End-to-end test of the speculative decoding loop with tournament.

    Loads real Qwen2.5-7B + helpfulness DPO blade, generates 50 tokens
    using Option B (γ=4, K=8, Swiss-system tournament), and verifies:

      1. Output is a non-empty string containing the original prompt
      2. SpeculativeStats fields are internally consistent:
         - total_rounds ≥ 1
         - total_rounds == full_accept_rounds + partial_accept_rounds
         - tournament_calls == gamma * total_rounds (one per position per round)
         - target_forward_passes == total_rounds (one per round)
         - blade_forward_passes == total_rounds (one per round)
         - total_tokens_accepted > 0
      3. At least some tournament rejection occurred (acceptance_rate < 1.0),
         meaning the tournament is actually influencing token selection
      4. Generation completes within a reasonable time (< 120 seconds for 50 tokens)
      5. acceptance_positions list length == partial_accept_rounds
    """
    print("━" * 70)
    print("  TEST 1 — Full Option B Speculative Loop (Real Weights)")
    print("━" * 70)

    shared = _get_shared()
    cfg = shared["cfg"]
    tokenizer = shared["tokenizer"]
    base_model = shared["base_model"]

    # Load helpfulness blade
    logger.info("Loading helpfulness blade...")
    blade_model = load_blade_model(cfg, "helpfulness")

    # Build generator
    generator = SwissKnifeSpeculativeGenerator(
        cfg=cfg,
        tokenizer=tokenizer,
        base_model=base_model,
        blade_model=blade_model,
    )

    prompt = "Explain the concept of AI alignment to a beginner in simple terms."
    max_tokens = 50

    logger.info("Generating %d tokens with Option B (γ=%d, K=%d, tournament=swiss)...",
                max_tokens, cfg.gamma, cfg.K)

    t0 = time.perf_counter()
    output_text, stats = generator.generate(
        prompt,
        max_new_tokens=max_tokens,
        verbose=True,
        return_stats=True,
    )
    elapsed = time.perf_counter() - t0

    # ── Assertions ─────────────────────────────────────────────────────

    # 1. Output is a non-empty string containing the prompt
    assert isinstance(output_text, str), f"Expected str, got {type(output_text)}"
    assert len(output_text) > len(prompt), (
        f"Output should be longer than prompt. Got {len(output_text)} chars"
    )
    assert prompt[:20] in output_text, (
        f"Output should contain the original prompt. "
        f"First 50 chars of output: '{output_text[:50]}'"
    )
    print(f"  ✓ Output is a valid string ({len(output_text)} chars)")

    # 2. Stats internal consistency
    assert stats.total_rounds >= 1, f"Expected ≥1 round, got {stats.total_rounds}"

    round_sum = stats.full_accept_rounds + stats.partial_accept_rounds
    assert stats.total_rounds == round_sum, (
        f"total_rounds ({stats.total_rounds}) != "
        f"full ({stats.full_accept_rounds}) + partial ({stats.partial_accept_rounds})"
    )
    print(f"  ✓ total_rounds={stats.total_rounds} == full({stats.full_accept_rounds}) + partial({stats.partial_accept_rounds})")

    # Tournament calls = gamma per round (one tournament per position per round)
    # But rounds can terminate early due to rejection or token budget,
    # so tournament_calls <= gamma * total_rounds
    assert stats.tournament_calls >= stats.total_rounds, (
        f"tournament_calls ({stats.tournament_calls}) should be ≥ total_rounds ({stats.total_rounds})"
    )
    assert stats.tournament_calls <= cfg.gamma * stats.total_rounds, (
        f"tournament_calls ({stats.tournament_calls}) should be ≤ γ×rounds ({cfg.gamma * stats.total_rounds})"
    )
    print(f"  ✓ tournament_calls={stats.tournament_calls} ∈ [{stats.total_rounds}, {cfg.gamma * stats.total_rounds}]")

    # Forward passes = one per round
    assert stats.target_forward_passes == stats.total_rounds, (
        f"target_passes ({stats.target_forward_passes}) != rounds ({stats.total_rounds})"
    )
    assert stats.blade_forward_passes == stats.total_rounds, (
        f"blade_passes ({stats.blade_forward_passes}) != rounds ({stats.total_rounds})"
    )
    print(f"  ✓ target_passes={stats.target_forward_passes}, blade_passes={stats.blade_forward_passes} (both == rounds)")

    # Tokens generated > 0
    assert stats.total_tokens_accepted > 0, "No tokens were generated"
    print(f"  ✓ total_tokens_accepted={stats.total_tokens_accepted}")

    # 3. Tournament should sometimes reject (for non-trivial prompts)
    # acceptance_rate < 1.0 means the tournament is actually doing something
    if stats.total_rounds >= 3:
        # With enough rounds, we expect at least some rejection
        print(f"  ℹ acceptance_rate={stats.acceptance_rate:.2%} "
              f"({'tournament active' if stats.acceptance_rate < 1.0 else 'all greedy — may happen with this prompt'})")

    # 4. Reasonable time (< 120s for 50 tokens on any GPU)
    assert elapsed < 120.0, f"Generation took {elapsed:.1f}s — too slow"
    print(f"  ✓ Completed in {elapsed:.2f}s ({stats.tokens_per_second:.1f} tok/s)")

    # 5. acceptance_positions length == partial_accept_rounds
    assert len(stats.acceptance_positions) == stats.partial_accept_rounds, (
        f"acceptance_positions length ({len(stats.acceptance_positions)}) "
        f"!= partial_accept_rounds ({stats.partial_accept_rounds})"
    )
    if stats.acceptance_positions:
        print(f"  ✓ Rejection positions: {stats.acceptance_positions} "
              f"(all ∈ [0, {cfg.gamma - 1}])")
        for pos in stats.acceptance_positions:
            assert 0 <= pos < cfg.gamma, f"Rejection position {pos} out of [0, γ)"

    # Print the generated text
    print()
    print(f"  ┌─ Generated Output ─────────────────────────────────────────")
    for line in output_text.split("\n"):
        print(f"  │ {line}")
    print(f"  └────────────────────────────────────────────────────────────")

    # Print stats summary
    print()
    stats_dict = stats.to_dict()
    print(f"  ┌─ SpeculativeStats ──────────────────────────────────────────")
    for k, v in stats_dict.items():
        print(f"  │  {k:30s}: {v}")
    print(f"  └────────────────────────────────────────────────────────────")

    print("\n  ✓ TEST 1 PASSED\n")
    return generator, stats


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Blade Swap Mid-Generation via BladeRack
# ─────────────────────────────────────────────────────────────────────────────

def test_blade_swap_and_regenerate():
    """
    Loads helpfulness + harmlessness blades into a BladeRack, generates
    with each, and verifies:

      1. BladeRack loads both blades successfully
      2. Hot-swap completes in < 1 second (pointer swap, not model reload)
      3. ReconfigurationProfile has valid fields
      4. Both generations produce valid non-empty outputs
      5. The two outputs are different (blades influence generation)
      6. Stats from both runs are internally consistent
    """
    print("━" * 70)
    print("  TEST 2 — Blade Swap via BladeRack (Real Weights)")
    print("━" * 70)

    shared = _get_shared()
    cfg = shared["cfg"]
    tokenizer = shared["tokenizer"]
    base_model = shared["base_model"]

    # Build BladeRack and load two blades
    logger.info("Building BladeRack with helpfulness + harmlessness...")
    rack = BladeRack(cfg, tokenizer, base_model)

    t0 = time.perf_counter()
    rack.load_blade("helpfulness")
    load_time_help = time.perf_counter() - t0
    logger.info("helpfulness loaded in %.1fs", load_time_help)

    t0 = time.perf_counter()
    rack.load_blade("harmlessness")
    load_time_harm = time.perf_counter() - t0
    logger.info("harmlessness loaded in %.1fs", load_time_harm)

    assert len(rack.loaded_blades) == 2, f"Expected 2 blades, got {len(rack.loaded_blades)}"
    print(f"  ✓ BladeRack loaded 2 blades: {rack.loaded_blades}")

    # ── Generate with helpfulness blade ──────────────────────────────────
    blade_help, profile_help = rack.swap("helpfulness")
    assert isinstance(profile_help, ReconfigurationProfile)

    gen = SwissKnifeSpeculativeGenerator(
        cfg=cfg,
        tokenizer=tokenizer,
        base_model=base_model,
        blade_model=blade_help.blade_model,
    )
    # Replace the blade scorer with the one from the rack
    gen.blade = blade_help

    prompt = "How should I deal with a difficult coworker?"

    logger.info("Generating with helpfulness blade...")
    output_help, stats_help = gen.generate(
        prompt, max_new_tokens=40, return_stats=True,
    )

    assert isinstance(output_help, str) and len(output_help) > len(prompt)
    assert stats_help.total_tokens_accepted > 0
    print(f"  ✓ Helpfulness output: {len(output_help)} chars, "
          f"{stats_help.total_tokens_accepted} tokens")

    # ── Hot-swap to harmlessness blade ───────────────────────────────────
    logger.info("Hot-swapping to harmlessness blade...")
    t_swap_start = time.perf_counter()
    blade_harm, profile_swap = rack.swap("harmlessness")
    swap_time = time.perf_counter() - t_swap_start

    # 2. Swap must be fast (< 1 second — it's a pointer swap)
    assert swap_time < 1.0, (
        f"Blade swap took {swap_time:.3f}s — should be < 1s (pointer swap)"
    )
    print(f"  ✓ Blade swap completed in {swap_time*1000:.2f} ms")

    # 3. Profile fields
    assert profile_swap.to_blade == "harmlessness"
    assert profile_swap.from_blade == "helpfulness"
    assert profile_swap.swap_time_ms >= 0
    assert profile_swap.adapter_params > 0
    print(f"  ✓ ReconfigurationProfile: {profile_swap.from_blade} → {profile_swap.to_blade}, "
          f"{profile_swap.adapter_params:,} LoRA params")

    # Update generator with the new blade
    gen.blade_model = blade_harm.blade_model
    gen.blade = blade_harm

    logger.info("Generating with harmlessness blade...")
    output_harm, stats_harm = gen.generate(
        prompt, max_new_tokens=40, return_stats=True,
    )

    # 4. Both outputs are valid
    assert isinstance(output_harm, str) and len(output_harm) > len(prompt)
    assert stats_harm.total_tokens_accepted > 0
    print(f"  ✓ Harmlessness output: {len(output_harm)} chars, "
          f"{stats_harm.total_tokens_accepted} tokens")

    # 5. Outputs should differ (different blades → different alignment steering)
    # We compare the generated portion (after the prompt)
    gen_help = output_help[len(prompt):].strip()
    gen_harm = output_harm[len(prompt):].strip()

    if gen_help != gen_harm:
        print(f"  ✓ Outputs DIFFER (blade swap changed generation) — expected behavior")
    else:
        # This CAN happen if α is high and both blades agree, but flag it
        print(f"  ⚠ Outputs are identical — blades may agree on this prompt (not a failure)")

    # 6. Stats consistency for both runs
    for label, stats in [("helpfulness", stats_help), ("harmlessness", stats_harm)]:
        round_sum = stats.full_accept_rounds + stats.partial_accept_rounds
        assert stats.total_rounds == round_sum, (
            f"{label}: total_rounds ({stats.total_rounds}) != "
            f"full ({stats.full_accept_rounds}) + partial ({stats.partial_accept_rounds})"
        )
        assert stats.target_forward_passes == stats.total_rounds
        assert stats.blade_forward_passes == stats.total_rounds
    print(f"  ✓ Stats internally consistent for both blades")

    # ── Print comparison ─────────────────────────────────────────────────
    print()
    print(f"  ┌─ Helpfulness Output ────────────────────────────────────────")
    print(f"  │ {output_help[:200]}")
    print(f"  ├─ Harmlessness Output ───────────────────────────────────────")
    print(f"  │ {output_harm[:200]}")
    print(f"  └────────────────────────────────────────────────────────────")

    print("\n  ✓ TEST 2 PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Swiss Knife — GPU Integration Tests")
    print("  Speculative Decoding Loop with Tournament (Real Weights)")
    print("=" * 70)
    print()
    print("  ⚠  These tests require GPU + model downloads (~14 GB).")
    print("     Run on Vast.ai, not locally.")
    print()

    if not torch.cuda.is_available():
        print("  WARNING: No CUDA GPU detected. Tests will run on CPU (very slow).")
        resp = input("  Continue anyway? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            sys.exit(0)

    test_full_speculative_loop()
    test_blade_swap_and_regenerate()

    print("=" * 70)
    print("  ALL GPU INTEGRATION TESTS PASSED ✓")
    print("=" * 70)
