"""
Swiss Knife — Model Loading

Handles loading the frozen base/draft model, tokenizer, and DPO blade adapters.

The base model is an SFT-merged Qwen2.5 checkpoint hosted as a HuggingFace
dataset at  divyajot5005/ndna → SFT/Qwen_SFT_merged/.
We use snapshot_download to fetch the model files, then load locally.

Memory budget:
    Qwen2.5-3B in float32  ≈ 13 GB  (draft copy)
    + blade copy            ≈ 13 GB
    Total needed            ≈ 26 GB RAM (CPU) or VRAM (GPU)

    In float16/bfloat16     ≈  6.5 GB each → 13 GB VRAM (GPU recommended)

The base model serves dual purpose:
    1. Draft model  — generates K candidate spans
    2. Reference π_ref — the un-adapted policy for DPO log-ratio computation

Each blade is a PEFT LoRA adapter loaded on a *separate* copy of the base model,
yielding π_blade.  This avoids adapter-swapping during batched scoring.
"""

import logging
import os
from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

from .config import SwissKnifeConfig

logger = logging.getLogger(__name__)

# ── dtype mapping ──────────────────────────────────────────────────────────
_DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}


def _resolve_dtype(cfg: SwissKnifeConfig) -> torch.dtype:
    dtype = _DTYPE_MAP.get(cfg.dtype)
    if dtype is None:
        raise ValueError(f"Unknown dtype '{cfg.dtype}'. Choose from {list(_DTYPE_MAP)}")
    if dtype == torch.float16 and not torch.cuda.is_available():
        logger.warning(
            "float16 requested but no CUDA GPU found — falling back to float32. "
            "Pass --dtype bfloat16 or float32 explicitly to suppress this."
        )
        return torch.float32
    return dtype


def _resolve_device(cfg: SwissKnifeConfig):
    """Resolve device for ``from_pretrained``.

    Returns ``{'': 0}`` on GPU to pin the entire model onto cuda:0 (rather
    than letting accelerate shard layers across CPU/GPU). Sharding produces
    index-vs-tensor device mismatches in the candidate-scoring forward
    passes. With 13 GB total VRAM needed (two bf16 copies of Qwen2.5-3B),
    pinning is safe on any GPU with ≥16 GB free."""
    if cfg.device == "auto":
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected — pinning model to cuda:0")
            return {"": 0}
        logger.info("No CUDA GPU — using CPU. Expect slow inference.")
        return "cpu"
    return cfg.device


# Cache the download path so we don't call snapshot_download multiple times
_base_model_cache: dict = {}


def _download_base_model(cfg: SwissKnifeConfig) -> str:
    """Download the SFT-merged base model from the HuggingFace dataset repo.

    The model is stored inside a dataset repo subfolder, so we use
    snapshot_download with allow_patterns to fetch only the relevant files.
    Results are cached in memory for the process lifetime.

    Returns
    -------
    str
        Local path to the downloaded model directory.
    """
    cache_key = (cfg.base_model_id, cfg.base_model_subfolder)
    if cache_key in _base_model_cache:
        return _base_model_cache[cache_key]

    subfolder = cfg.base_model_subfolder
    logger.info(
        "Downloading base model from dataset repo: %s / %s",
        cfg.base_model_id, subfolder,
    )
    local_dir = snapshot_download(
        repo_id=cfg.base_model_id,
        repo_type="dataset",
        allow_patterns=[f"{subfolder}/*"],
    )
    model_path = os.path.join(local_dir, subfolder)
    _base_model_cache[cache_key] = model_path
    logger.info("Base model cached at: %s", model_path)
    return model_path


# ── Public loaders ─────────────────────────────────────────────────────────

def load_tokenizer(cfg: SwissKnifeConfig) -> PreTrainedTokenizer:
    """Load the tokenizer for the base model, with correct padding setup."""
    model_path = _download_base_model(cfg)
    logger.info("Loading tokenizer from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # left-pad for batched generation
    return tokenizer


def load_base_model(cfg: SwissKnifeConfig) -> PreTrainedModel:
    """Load the frozen base / draft model (π_draft = π_ref).

    Used for candidate span generation AND as the reference policy
    in the DPO implicit reward  r = β·log(π_blade/π_ref).
    """
    model_path = _download_base_model(cfg)
    dtype = _resolve_dtype(cfg)
    device = _resolve_device(cfg)

    logger.info(
        "Loading base model (draft + ref): %s  [dtype=%s, device=%s]",
        model_path, dtype, device,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Base model loaded and frozen. Params: %s", f"{n_params:,}")
    return model


def load_blade_model(
    cfg: SwissKnifeConfig,
    blade_name: str,
) -> PeftModel:
    """Load a DPO LoRA adapter (blade) on a fresh copy of the base model.

    Supports adapters hosted in either a HuggingFace *model* repo (passed
    directly to PeftModel) or a *dataset* repo (downloaded first via
    snapshot_download, since PEFT does not understand dataset repos).

    Parameters
    ----------
    blade_name : str
        Key into ``cfg.blade_sources``. E.g. ``"helpfulness"``,
        ``"harmlessness"``, ``"truthfulness"``.

    Returns
    -------
    PeftModel
        Base model + LoRA adapter. Forward passes yield π_blade.
    """
    if blade_name not in cfg.blade_sources:
        raise ValueError(
            f"Unknown blade '{blade_name}'. "
            f"Available: {list(cfg.blade_sources)}"
        )
    source = cfg.blade_sources[blade_name]
    repo_id   = source["repo_id"]
    repo_type = source["repo_type"]
    subfolder = source["subfolder"]

    model_path = _download_base_model(cfg)
    dtype = _resolve_dtype(cfg)
    device = _resolve_device(cfg)

    logger.info(
        "Loading blade '%s' base copy [dtype=%s, device=%s]...",
        blade_name, dtype, device,
    )
    base_for_blade = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    if repo_type == "model":
        logger.info(
            "Attaching LoRA adapter from model repo %s / %s ...",
            repo_id, subfolder,
        )
        blade_model = PeftModel.from_pretrained(
            base_for_blade,
            repo_id,
            subfolder=subfolder,
            torch_dtype=dtype,
        )
    elif repo_type == "dataset":
        logger.info(
            "Downloading dataset-hosted adapter %s / %s ...",
            repo_id, subfolder,
        )
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{subfolder}/*"],
        )
        adapter_path = os.path.join(local_dir, subfolder)
        logger.info("Attaching LoRA adapter from local path: %s", adapter_path)
        blade_model = PeftModel.from_pretrained(
            base_for_blade,
            adapter_path,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unknown repo_type '{repo_type}' for blade '{blade_name}'. "
            f"Expected 'model' or 'dataset'."
        )

    blade_model.eval()
    for param in blade_model.parameters():
        param.requires_grad = False

    adapter_params = sum(
        p.numel() for n, p in blade_model.named_parameters()
        if "lora" in n.lower()
    )
    logger.info("Blade '%s' loaded. LoRA params: %s", blade_name, f"{adapter_params:,}")
    return blade_model


def load_all(
    cfg: SwissKnifeConfig,
    blade_name: str,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, PeftModel]:
    """Convenience: load tokenizer, base model, and one blade in one call."""
    tokenizer   = load_tokenizer(cfg)
    base_model  = load_base_model(cfg)
    blade_model = load_blade_model(cfg, blade_name)
    return tokenizer, base_model, blade_model
