"""
Swiss-Knife — Decode-Time Alignment via Elo Tournament Selection

Step-level guided inference using a probabilistic Elo tournament to select
the best candidate reasoning step from a fast Drafter model.

Core strategy (Mode B — unconditional acceptance):
    Sample n reasoning steps → Blade reward scoring + uncertainty estimation
    → Thurstonian Elo tournament → softmax champion selection → commit unconditionally.

For Mode A (with Verifier acceptance gate), see: elo_swiss.py
For tournament mechanics, see: elo_system.py
For uncertainty estimation, see: sigma_estimator.py

Architecture:
    Base/Draft Model   : Qwen2.5 SFT-merged (frozen)
    Alignment Blades   : DPO LoRA adapters (helpfulness, harmlessness, truthfulness)
    Blade Registry     : BladeRack pointer swap, O(1), no retraining
    Tournament Format  : Elo-rating system (Thurstonian Case-V or Bradley-Terry)

Reference:
    Swiss-Knife: Elo Tournament Selection for Decoding-Time Alignment
    AAAI 2027 (Anonymous Submission)
"""

__version__ = "0.3.0"

__all__ = []
