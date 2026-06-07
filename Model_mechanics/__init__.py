"""
Swiss Knife — Decode-Time Alignment via Tournament Sampling

Option A (Non-Speculative Best-of-K Tournament):
    Sample K independent spans → tournament selects best → commit → repeat.
    See: Model_mechanics/generation.py, Model_mechanics/tournament.py

Option B (Speculative-Decoding-Integrated Tournament Verifier):
    Draft proposes γ tokens → top-K per position → [γ, K] candidate tensor.
    Target + Blade: ONE forward pass each → [γ, K] scores.
    Per-position tournament → acceptance propagation (discard tail on rejection).
    See: Model_mechanics/speculative_generator.py, Model_mechanics/swiss_system.py

Architecture:
    Base/Draft Model   : Qwen2.5 SFT-merged (frozen)
    Alignment Blades   : DPO LoRA adapters (helpfulness, harmlessness, truthfulness)
    Tournament Formats : Knockout bracket or Swiss-system schedule
    Hot-swap           : BladeRack pointer swap, O(1), no retraining

Reference:
    Swiss Knife Analysis — Pragya Lab, BITS Pilani Goa (2026)
    Section 5 (Option A / Algorithm 1) and Section 6 (Option B / Algorithm 2)
"""

__version__ = "0.2.0"
