"""
Swiss Knife — Configuration

All hyperparameters and model identifiers in one place.

NOTE ON BASE MODEL:
    The DPO adapters at MGPGRAD/Swiss-Knife were trained on a
    Qwen2.5-based SFT-merged checkpoint, NOT on Llama-3.2-1B.
    The adapter_config.json specifies:
        base_model_name_or_path: ./ndna_data/SFT/Qwen_SFT_merged
    which is a Qwen2ForCausalLM (hidden=3584, 28 layers, vocab=152064).
    This SFT-merged model is hosted ungated at:
        divyajot5005/ndna  →  SFT/Qwen_SFT_merged/
    We load it as a HuggingFace dataset-hosted model (no gating).
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SwissKnifeConfig:
    """Central configuration for the Swiss Knife Option A pipeline."""

    # ── Model identifiers ───────────────────────────────────────────────
    base_model_id: str = "divyajot5005/ndna"
    """HuggingFace *dataset* repo hosting the SFT-merged base model.
    The actual model files are under the subfolder SFT/Qwen_SFT_merged/."""

    base_model_subfolder: str = "SFT/Qwen_SFT_merged"
    """Subfolder within base_model_id containing the full model weights."""

    blade_repo_id: str = "MGPGRAD/Swiss-Knife"
    """HuggingFace repo containing the DPO adapter checkpoints."""

    blade_subfolder_map: Dict[str, str] = field(default_factory=lambda: {
        "helpfulness": "dpo_out/hh_helpfulness/final_adapter",
        "truthfulness": "dpo_out/truthfulness/final_adapter",
    })
    """Maps human-readable blade names → subfolder paths within *blade_repo_id*."""

    # ── Tournament hyperparameters ──────────────────────────────────────
    K: int = 8
    """Number of candidate spans per tournament round."""

    L: int = 5
    """Span length (number of tokens per candidate)."""

    alpha: float = 0.5
    """Mixing coefficient  α ∈ [0, 1].
       α = 1.0 → pure draft likelihood (no alignment).
       α = 0.0 → pure blade reward (ignores fluency).
       α ≈ 0.5 → balanced (default operating point)."""

    beta: float = 0.1
    """DPO implicit reward scaling:  r_blade = β · log(π_blade / π_ref)."""

    # ── Generation parameters ───────────────────────────────────────────
    max_new_tokens: int = 200
    """Maximum total tokens to generate."""

    temperature: float = 1.0
    """Sampling temperature for candidate span generation."""

    top_k: int = 50
    """Top-k filtering for candidate span generation."""

    top_p: float = 0.95
    """Nucleus (top-p) filtering for candidate span generation."""

    # ── System ──────────────────────────────────────────────────────────
    device: str = "auto"
    """Device for model placement.  'auto' uses accelerate device_map.
    On CPU-only machines, 'cpu' is set automatically when no CUDA is found."""

    dtype: str = "float32"
    """Compute dtype: 'float16', 'bfloat16', or 'float32'.
    float32 is the safe default for CPU.  Use float16/bfloat16 on GPU only.
    Memory budget (Qwen2.5-3B):
        float32  → ~13 GB  (2× copies needed: draft + blade = ~26 GB)
        float16  → ~6.5 GB (needs GPU; 2× = ~13 GB VRAM)
        bfloat16 → ~6.5 GB (safer than float16 on CPU, but still large)"""

    seed: int = 42
    """Random seed for reproducibility."""

    def __post_init__(self):
        assert 0.0 <= self.alpha <= 1.0, f"α must be in [0,1], got {self.alpha}"
        assert self.K >= 2 and (self.K & (self.K - 1) == 0), \
            f"K must be a power of 2 for knockout bracket, got {self.K}"
        assert self.L >= 1, f"Span length L must be ≥ 1, got {self.L}"
        assert self.beta > 0, f"β must be positive, got {self.beta}"
