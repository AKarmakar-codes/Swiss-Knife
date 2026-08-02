"""
Benchmarking Strategies Package
===============================

This package contains the implementations of the Tier 1 baseline alignment
strategies used for evaluating and comparing against the Swiss Knife methodology.
The strategies implemented here follow the requirements defined in `benchmarking_plan.md`
and are designed to interface seamlessly with the generation evaluation pipeline.

Available strategies:
- Best-of-N (BoN): Generates multiple candidates and selects the one with the highest reward.
- ARGS: Token-level steering by shifting logits based on blade rewards.
- MOD (Multi-Objective Decoding): Token-level linear combination of log-probabilities from multiple blades.
- DeAL: Decoding-time ALignment via top-k candidate scoring.
"""
