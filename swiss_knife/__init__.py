"""
Swiss Knife — Decode-Time Alignment via Tournament Sampling

Option A (Non-Speculative Best-of-K Tournament) POC Pipeline.

Architecture:
    Base/Draft Model : Llama-3.2-1B (frozen)
    Alignment Blades : DPO LoRA adapters (hh_helpfulness, truthfulness)
    Sampling         : Span-level tournament sampling
    Tournament       : Knockout bracket, K=8 candidates, span length L=5

Reference:
    Swiss Knife Analysis — Pragya Lab, BITS Pilani Goa (2026)
    Section 5, Algorithm 1 / Listing 1
"""

__version__ = "0.1.0"
