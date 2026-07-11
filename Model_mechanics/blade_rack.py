"""
Swiss Knife — Phase 3: Plug-and-Play Blade Rack
================================================

Manages pre-loaded alignment blades and provides hot-swap with profiling.

Design:
  • BladeRack loads all configured blades once and caches them.
  • swap(name) returns the target DPOBlade in ~O(1) — it's a pointer swap,
    not a model reload. Memory footprint is: base_model × 1 + blade_adapters × N.
  • ReconfigurationProfile captures: swap_time_ms, memory_delta_mb, adapter_params.
  • MoDStyleRetrainEstimate analytically computes the cost of retraining an MoD
    router to accommodate a new objective — used as the comparison baseline.

Why this matters:
  Adding a new alignment objective to Swiss Knife = upload a LoRA adapter,
  add one entry to config.blade_sources, call BladeRack.load_blade(name).
  Adding a new objective to MoD = retrain the one-hot router + joint pathways
  over the full training corpus = O(GPU-days).
"""

import gc
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SwissKnifeConfig
from .blades import DPOBlade
from .models import load_blade_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Profiling data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReconfigurationProfile:
    """Profiling data for a single blade swap operation."""

    from_blade: str
    """The previously active blade name."""

    to_blade: str
    """The newly active blade name."""

    swap_time_ms: float
    """Wall-clock time for the pointer swap in milliseconds."""

    memory_before_mb: float
    """GPU/CPU memory allocated before the swap (MB)."""

    memory_after_mb: float
    """GPU/CPU memory allocated after the swap (MB)."""

    memory_delta_mb: float
    """Net memory change = after - before. Typically near 0 (LoRA adapters
    already loaded; only the active pointer changes)."""

    adapter_params: int
    """Number of LoRA parameters in the newly active blade."""

    def to_dict(self) -> dict:
        return {
            "from_blade": self.from_blade,
            "to_blade": self.to_blade,
            "swap_time_ms": round(self.swap_time_ms, 3),
            "memory_before_mb": round(self.memory_before_mb, 2),
            "memory_after_mb": round(self.memory_after_mb, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 4),
            "adapter_params": self.adapter_params,
        }

    def __str__(self) -> str:
        return (
            f"Swap '{self.from_blade}' → '{self.to_blade}' | "
            f"{self.swap_time_ms:.2f} ms | "
            f"ΔMem: {self.memory_delta_mb:+.2f} MB | "
            f"LoRA params: {self.adapter_params:,}"
        )


@dataclass
class MoDStyleRetrainEstimate:
    """Analytical estimate of the cost to retrain an MoD router for a new objective.

    This is a documentation/comparison stub — no actual MoD model is run.
    The estimates are computed from standard scaling laws and are used to
    justify the Swiss Knife modular swap approach.

    Estimates assume a Qwen2.5-3B scale MoD system.
    """

    model_params: int = 3_000_000_000
    """Number of parameters in the base model (Qwen2.5-3B)."""

    router_params: int = 0
    """MoD one-hot router parameter count (estimated as hidden_size × num_objectives)."""

    joint_pathway_params: int = 0
    """Additional joint pathway parameters shared across objectives."""

    gpu_hours_estimate: float = 0.0
    """Estimated GPU-hours to retrain the router + joint pathways."""

    training_tokens: int = 0
    """Token budget for retraining."""

    def __post_init__(self):
        # Qwen2.5-3B: hidden=3584, typical MoD router = hidden × num_objectives
        num_objectives = 3  # helpfulness, harmlessness, truthfulness
        self.router_params = 3584 * num_objectives  # ~10k params
        # Joint pathway: assume 2 transformer layers worth ~2 × 3584² × 4
        self.joint_pathway_params = 2 * 3584 * 3584 * 4  # ~100M
        # Rough GPU-hour estimate: 1 A100 × 24h minimum for router convergence
        self.gpu_hours_estimate = 24.0  # conservative lower bound
        self.training_tokens = 500_000_000  # 500M tokens (standard DPO dataset)

    def summary(self) -> dict:
        return {
            "model_params": f"{self.model_params / 1e9:.1f}B",
            "router_params": f"{self.router_params:,}",
            "joint_pathway_params": f"{self.joint_pathway_params / 1e6:.1f}M",
            "gpu_hours_estimate": f"{self.gpu_hours_estimate:.0f}h",
            "training_tokens": f"{self.training_tokens / 1e6:.0f}M",
            "note": (
                "MoD requires full retraining when a new objective is added — "
                "the one-hot router and joint pathways must be re-optimized "
                "over the full training corpus. Swiss Knife adds a new blade "
                "by uploading one LoRA adapter (~0 GPU-hours)."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BladeRack
# ─────────────────────────────────────────────────────────────────────────────

class BladeRack:
    """Pre-loads and caches all alignment blades for zero-cost hot-swapping.

    Usage
    -----
    ::

        rack = BladeRack(cfg, tokenizer, base_model)
        rack.load_all()  # downloads and caches all blades from cfg.blade_sources

        # Hot-swap during generation:
        blade, profile = rack.swap("harmlessness")
        generator.blade_model = blade.blade_model
        generator.blade = blade

        # Profile the swap:
        print(profile)

    Parameters
    ----------
    cfg : SwissKnifeConfig
        Configuration — blade_sources defines which blades to load.
    tokenizer : PreTrainedTokenizer
        Shared tokenizer.
    base_model : PreTrainedModel
        Frozen reference model (π_ref).
    """

    def __init__(
        self,
        cfg: SwissKnifeConfig,
        tokenizer: PreTrainedTokenizer,
        base_model: PreTrainedModel,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.base_model = base_model
        self._blades: Dict[str, DPOBlade] = {}
        self._active_blade_name: Optional[str] = None
        self._load_times: Dict[str, float] = {}

    # ── Loading ──────────────────────────────────────────────────────────

    def load_blade(self, blade_name: str) -> DPOBlade:
        """Load a single blade from HuggingFace and cache it.

        If already cached, returns the cached instance immediately.
        """
        if blade_name in self._blades:
            logger.debug("Blade '%s' already in rack (cached).", blade_name)
            return self._blades[blade_name]

        logger.info("Loading blade '%s' into rack...", blade_name)
        t0 = time.perf_counter()
        blade_model = load_blade_model(self.cfg, blade_name)
        blade = DPOBlade(self.cfg, self.base_model, blade_model, self.tokenizer)
        elapsed = (time.perf_counter() - t0) * 1000
        self._blades[blade_name] = blade
        self._load_times[blade_name] = elapsed
        logger.info("Blade '%s' loaded in %.1f ms.", blade_name, elapsed)
        return blade

    def load_all(self) -> None:
        """Pre-load all blades defined in cfg.blade_sources."""
        for blade_name in self.cfg.blade_sources:
            self.load_blade(blade_name)
        logger.info(
            "BladeRack: %d blades loaded: %s",
            len(self._blades), list(self._blades.keys()),
        )

    def get_stochastic_auditor(self, blade_name: str, auditor_cfg = None) -> "StochasticAuditor":
        """Get a StochasticAuditor instance for a loaded blade."""
        from .stochastic_auditor import StochasticAuditor, StochasticAuditorConfig
        blade = self.load_blade(blade_name)
        if auditor_cfg is None:
            auditor_cfg = StochasticAuditorConfig(
                mode=self.cfg.stochastic_mode,
                dropout_p=self.cfg.stochastic_dropout_p,
                proj_epsilon=self.cfg.stochastic_proj_epsilon,
                head_frac=self.cfg.stochastic_head_frac,
                num_layers_to_mask=self.cfg.stochastic_num_layers_to_mask,
            )
        return StochasticAuditor(blade, self.cfg, auditor_cfg)


    # ── Hot-swap ─────────────────────────────────────────────────────────

    def swap(self, blade_name: str) -> Tuple[DPOBlade, ReconfigurationProfile]:
        """Hot-swap the active alignment blade.

        This is a pointer swap — no model weights are moved or reloaded.
        All blades must be pre-loaded (call load_all() or load_blade() first).

        Parameters
        ----------
        blade_name : str
            Name of the target blade (must be in cfg.blade_sources and loaded).

        Returns
        -------
        (DPOBlade, ReconfigurationProfile)
            The new active blade and profiling data for this operation.

        Raises
        ------
        KeyError
            If the blade has not been loaded yet. Call load_blade(name) first.
        """
        if blade_name not in self._blades:
            raise KeyError(
                f"Blade '{blade_name}' not in rack. "
                f"Available: {list(self._blades.keys())}. "
                f"Call load_blade('{blade_name}') first."
            )

        from_blade = self._active_blade_name or "<none>"

        # ── Measure memory before swap ────────────────────────────────────
        mem_before = _get_memory_mb()

        # ── The actual swap: pointer reassignment (O(1)) ──────────────────
        t0 = time.perf_counter()
        new_blade = self._blades[blade_name]
        self._active_blade_name = blade_name
        swap_time_ms = (time.perf_counter() - t0) * 1000

        # ── Measure memory after swap ─────────────────────────────────────
        mem_after = _get_memory_mb()

        # Count LoRA adapter parameters
        adapter_params = _count_lora_params(new_blade.blade_model)

        profile = ReconfigurationProfile(
            from_blade=from_blade,
            to_blade=blade_name,
            swap_time_ms=swap_time_ms,
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=mem_after - mem_before,
            adapter_params=adapter_params,
        )

        logger.info(str(profile))
        return new_blade, profile

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def active_blade_name(self) -> Optional[str]:
        return self._active_blade_name

    @property
    def loaded_blades(self):
        return list(self._blades.keys())

    def get_load_time(self, blade_name: str) -> float:
        """Return the initial download+load time in ms for a blade."""
        return self._load_times.get(blade_name, 0.0)

    def print_summary(self) -> None:
        """Print a formatted summary of the rack contents and load times."""
        print("\n" + "═" * 60)
        print("  BladeRack Summary")
        print("═" * 60)
        print(f"  Active blade : {self._active_blade_name or '(none)'}")
        print(f"  Loaded blades: {len(self._blades)}")
        print()
        for name, blade in self._blades.items():
            params = _count_lora_params(blade.blade_model)
            load_t = self._load_times.get(name, 0.0)
            active_marker = " ← ACTIVE" if name == self._active_blade_name else ""
            print(
                f"  {name:20s} | {params:>12,} LoRA params"
                f" | loaded in {load_t:>8.0f} ms{active_marker}"
            )
        print()
        # MoD comparison
        mod_est = MoDStyleRetrainEstimate()
        summary = mod_est.summary()
        print("  MoD Retrain Cost (if adding a new objective):")
        for k, v in summary.items():
            if k != "note":
                print(f"    {k:30s}: {v}")
        print(f"    {'note':30s}: {summary['note']}")
        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_memory_mb() -> float:
    """Return current allocated GPU memory in MB, or 0.0 on CPU-only."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def _count_lora_params(model: PeftModel) -> int:
    """Count the number of LoRA-specific trainable parameters in a PeftModel."""
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if "lora" in name.lower()
    )
