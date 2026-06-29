"""
Swiss Knife — Configuration

All hyperparameters and model identifiers in one place.
Covers both Option A (non-speculative Best-of-K tournament) and
Option B (speculative-decoding-integrated tournament verifier).

NOTE ON BASE MODEL:
    The DPO adapters at MGPGRAD/Swiss-Knife and divyajot5005/ndna were
    trained on a Qwen2.5-based SFT-merged checkpoint (Qwen2ForCausalLM,
    hidden=3584, 28 layers, vocab=152064). The SFT-merged model is hosted
    ungated as a HuggingFace *dataset* at:
        divyajot5005/ndna  →  SFT/Qwen_SFT_merged/
    We load it via snapshot_download (no gating).

NOTE ON BLADES:
    Blades are heterogeneous in where they live on the HF Hub:
      • helpfulness  — model repo  MGPGRAD/Swiss-Knife
      • harmlessness — dataset repo divyajot5005/ndna
    So each blade entry carries (repo_id, repo_type, subfolder). The
    loader downloads to a local path before calling PeftModel.
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

    blade_sources: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "helpfulness": {
            "repo_id":    "MGPGRAD/Swiss-Knife",
            "repo_type":  "model",
            "subfolder":  "dpo_out/hh_helpfulness/final_adapter",
        },
        "harmlessness": {
            "repo_id":    "divyajot5005/ndna",
            "repo_type":  "dataset",
            "subfolder":  "SFT/qwen25_dpo_output/final_dpo_adapter",
        },
        "truthfulness": {
            "repo_id":    "MGPGRAD/Swiss-Knife",
            "repo_type":  "model",
            "subfolder":  "dpo_out/truthfulness/final_adapter",
        },
    })
    """Per-blade source descriptor: (repo_id, repo_type, subfolder).
    repo_type ∈ {"model", "dataset"} — controls which Hub API is used."""

    # ── Tournament hyperparameters ──────────────────────────────────────
    K: int = 8
    """Number of candidate tokens per position in the tournament.
    Option A: K independent spans are sampled.
    Option B: top-K token IDs are retained at each of the γ draft positions."""

    L: int = 5
    """Span length (number of tokens per candidate).
    Used only in Option A (generation.py). Ignored by Option B."""

    gamma: int = 4
    """Speculative lookahead depth γ (Option B only).
    The draft proposes γ future tokens. The target + blade each do
    ONE forward pass over all γ positions. Typical values: 4–8."""

    tournament_mode: str = "swiss"
    """Which tournament format to use: 'knockout' or 'swiss'.
    'knockout'  — single-elimination bracket (log2 K rounds, K−1 matches).
    'swiss'     — Swiss-system schedule (swiss_rounds rounds, K/2·R matches).
    Swiss-system is the default: more robust to auditor noise because
    candidates play all rounds (no early elimination)."""

    swiss_rounds: int = 3
    """Number of rounds in the Swiss-system tournament (used only when
    tournament_mode='swiss'). Typical value: ceil(log2(K)).
    With K=8, ceil(log2(8))=3 rounds, giving 12 total matches."""

    generation_mode: str = "option_b"
    """Which generation loop to run:
    'option_a' — non-speculative Best-of-K tournament (generation.py).
    'option_b' — speculative-decoding-integrated verifier (speculative_generator.py).
    'gsi_softmax'  — GSI with softmax selection over reasoning steps.
    'gsi_pairwise' — GSI with pairwise Bradley-Terry selection.
    'gsi_swiss'    — GSI with Swiss-system matches → points → softmax."""

    # ── GSI hyperparameters ─────────────────────────────────────────────
    gsi_n: int = 8
    """Number of candidate reasoning steps sampled per iteration (GSI's n).
    Higher n → better coverage of the reward landscape but slower per step."""

    gsi_threshold: float = 0.0
    """Rejection threshold u for GSI. If the tilted reward of the selected
    step falls below u, rejection triggers a resample.
    Set to -inf to disable rejection (accept everything)."""

    gsi_step_delimiter: str = "\n\n"
    """Delimiter marking the end of a reasoning step. GSI generates steps
    until this delimiter is produced, then scores the complete step."""

    gsi_max_step_tokens: int = 512
    """Maximum tokens per reasoning step. If the model hasn't produced the
    step delimiter after this many tokens, the step is force-terminated."""

    gsi_tau: float = 1.0
    """Temperature τ for pairwise Bradley-Terry selection (Strategy 2).
    P(A wins) = 1 / (1 + exp(-MATCH(A,B) / τ)).
    Lower τ → sharper (more deterministic) selection."""

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

    # ── Stochastic Auditor hyperparameters ──────────────────────────────
    use_stochastic_auditor: bool = False
    """If True, use stochastic functional mapping (e.g. MC dropout, random projection,
    head subsampling) instead of deterministic blade scores."""

    stochastic_mode: str = "mc_dropout"
    """Stochastic functional family to use: 'mc_dropout', 'random_proj', 'head_subsample'."""

    stochastic_dropout_p: float = 0.1
    """Dropout probability for mc_dropout mode."""

    stochastic_proj_epsilon: float = 0.1
    """Perturbation weight epsilon for random_proj mode."""

    stochastic_head_frac: float = 0.5
    """Fraction of attention heads to keep active in head_subsample mode."""

    stochastic_num_layers_to_mask: int = 2
    """Number of final transformer layers to mask attention heads in head_subsample mode."""


    dtype: str = "float32"
    """Compute dtype: 'float16', 'bfloat16', or 'float32'.
    float32 is the safe default for CPU.  Use float16/bfloat16 on GPU only.
    Memory budget (Qwen2.5-3B):
        float32  → ~13 GB  (2× copies needed: draft + blade = ~26 GB)
        float16  → ~6.5 GB (needs GPU; 2× = ~13 GB VRAM)
        bfloat16 → ~6.5 GB (safer than float16 on CPU, but still large)"""

    seed: int = 42
    """Random seed for reproducibility."""

    blade_bias: float = 0.0
    """Additive constant added to every blade score before the tournament.
    Used to empirically verify calibration invariance: the winner should
    be unchanged regardless of this value (pairwise differences cancel)."""

    normalize_scores: bool = True
    """Per-round z-score normalisation of both score tensors before the
    bracket. Fixes the scale mismatch between draft span log-likelihoods
    (O(10–100)) and DPO blade rewards (O(0.001–0.1)) that otherwise lets
    the draft term swamp the blade term at every α > 0. Set to False to
    reproduce the pristine paper-equation behaviour (useful for the
    kernel-level calibration-invariance test in Demo 6)."""

    scores_log: str = ""
    """Optional path to a JSONL file. When non-empty, every tournament
    round appends one JSON line with raw + post-normalisation draft and
    blade score vectors and the winner index. Used by make_plots.py to
    visualise the score-scale mismatch."""

    def __post_init__(self):
        assert 0.0 <= self.alpha <= 1.0, f"α must be in [0,1], got {self.alpha}"
        assert self.K >= 2, f"K must be ≥ 2, got {self.K}"
        if self.tournament_mode == "knockout":
            assert (self.K & (self.K - 1) == 0), \
                f"K must be a power of 2 for knockout bracket, got {self.K}. "\
                f"Use tournament_mode='swiss' for arbitrary K."
        assert self.L >= 1, f"Span length L must be ≥ 1, got {self.L}"
        assert self.beta > 0, f"β must be positive, got {self.beta}"
        assert self.gamma >= 1, f"γ (lookahead) must be ≥ 1, got {self.gamma}"
        assert self.tournament_mode in ("knockout", "swiss"), \
            f"tournament_mode must be 'knockout' or 'swiss', got '{self.tournament_mode}'"
        _valid_gen_modes = (
            "option_a", "option_b",
            "gsi_softmax", "gsi_pairwise", "gsi_swiss",
        )
        assert self.generation_mode in _valid_gen_modes, \
            f"generation_mode must be one of {_valid_gen_modes}, got '{self.generation_mode}'"

        # GSI-specific validation
        if self.generation_mode.startswith("gsi_"):
            assert self.gsi_n >= 2, f"gsi_n must be ≥ 2, got {self.gsi_n}"
            assert self.gsi_max_step_tokens >= 1, \
                f"gsi_max_step_tokens must be ≥ 1, got {self.gsi_max_step_tokens}"
            assert self.gsi_tau > 0, f"gsi_tau must be positive, got {self.gsi_tau}"

        # Stochastic Auditor validation
        if self.use_stochastic_auditor:
            assert self.generation_mode == "option_b", \
                "Stochastic auditor is currently only supported in speculative decoding Option B."
            _valid_stochastic_modes = ("mc_dropout", "random_proj", "head_subsample")
            assert self.stochastic_mode in _valid_stochastic_modes, \
                f"stochastic_mode must be one of {_valid_stochastic_modes}, got '{self.stochastic_mode}'"
            assert 0.0 <= self.stochastic_dropout_p <= 1.0, f"stochastic_dropout_p must be in [0,1], got {self.stochastic_dropout_p}"
            assert self.stochastic_proj_epsilon >= 0.0, f"stochastic_proj_epsilon must be non-negative, got {self.stochastic_proj_epsilon}"
            assert 0.0 < self.stochastic_head_frac <= 1.0, f"stochastic_head_frac must be in (0,1], got {self.stochastic_head_frac}"
            assert self.stochastic_num_layers_to_mask >= 0, f"stochastic_num_layers_to_mask must be non-negative, got {self.stochastic_num_layers_to_mask}"



