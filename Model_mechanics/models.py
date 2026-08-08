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
    """Download the base model from HuggingFace dataset or model repo.

    If base_model_repo_type is 'model' and there is no subfolder, we return the
    base_model_id directly to let HuggingFace's standard cache mechanisms handle it.
    Otherwise, we use snapshot_download to fetch the model files.
    Results are cached in memory for the process lifetime.

    Returns
    -------
    str
        Local path to the downloaded model directory or HuggingFace repo ID.
    """
    repo_type = getattr(cfg, "base_model_repo_type", "dataset")
    cache_key = (cfg.base_model_id, cfg.base_model_subfolder, repo_type)
    if cache_key in _base_model_cache:
        return _base_model_cache[cache_key]

    subfolder = cfg.base_model_subfolder

    if repo_type == "model" and not subfolder:
        # For standard model repositories without subfolders, return base_model_id
        # directly so AutoModel/AutoTokenizer loads from HuggingFace cache.
        _base_model_cache[cache_key] = cfg.base_model_id
        logger.info("Using base model ID from model repository directly: %s", cfg.base_model_id)
        return cfg.base_model_id

    logger.info(
        "Downloading base model from %s repo: %s / %s",
        repo_type, cfg.base_model_id, subfolder,
    )
    
    allow_patterns = [f"{subfolder}/*"] if subfolder else None
    local_dir = snapshot_download(
        repo_id=cfg.base_model_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
    )
    model_path = os.path.join(local_dir, subfolder) if subfolder else local_dir
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
    attn_kwargs = {}
    if dtype in [torch.float16, torch.bfloat16] and torch.cuda.is_available():
        try:
            import flash_attn
            attn_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        **attn_kwargs,
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
    base_model: Optional[PreTrainedModel] = None,
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

    if base_model is not None and getattr(cfg, "shared_base_model", False):
        logger.info("Using shared base model for blade '%s' [device=%s]...", blade_name, device)
        base_for_blade = base_model
    else:
        logger.info("Loading blade '%s' base copy [dtype=%s, device=%s]...", blade_name, dtype, device)
        attn_kwargs = {}
        if dtype in [torch.float16, torch.bfloat16] and torch.cuda.is_available():
            try:
                import flash_attn
                attn_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        base_for_blade = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            **attn_kwargs,
        )

    adapter_path_resolved = ""
    if repo_type == "model":
        logger.info("Attaching LoRA adapter from model repo %s / %s ...", repo_id, subfolder)
        adapter_path_resolved = repo_id
    elif repo_type == "dataset":
        logger.info("Downloading dataset-hosted adapter %s / %s ...", repo_id, subfolder)
        local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=[f"{subfolder}/*"])
        adapter_path_resolved = os.path.join(local_dir, subfolder)
        logger.info("Attaching LoRA adapter from local path: %s", adapter_path_resolved)
    elif repo_type == "local":
        adapter_path_resolved = repo_id if os.path.isabs(repo_id) else os.path.abspath(repo_id)
        logger.info("Attaching LoRA adapter from local filesystem path: %s", adapter_path_resolved)
    else:
        raise ValueError(
            f"Unknown repo_type '{repo_type}' for blade '{blade_name}'. "
            f"Expected 'model', 'dataset', or 'local'."
        )

    if isinstance(base_for_blade, PeftModel):
        base_for_blade.load_adapter(
            adapter_path_resolved,
            adapter_name=blade_name,
            subfolder=subfolder if repo_type == "model" else None
        )
        blade_model = base_for_blade
    else:
        blade_model = PeftModel.from_pretrained(
            base_for_blade,
            adapter_path_resolved,
            subfolder=subfolder if repo_type == "model" else None,
            adapter_name=blade_name,
            torch_dtype=dtype,
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


class DisabledAdapterView:
    """Callable wrapper exposing a PEFT model's *base* (LoRA-disabled)
    forward pass, without loading a second copy of the base model.

    ``load_blade_model`` loads a fresh ~7.6B-parameter base model and
    attaches a small (~40M-param) LoRA adapter to make ``blade_model``.
    Historically, callers *also* loaded a completely separate ~7.6B-param
    copy via ``load_verifier_model``/``load_base_model`` to act as π_ref —
    doubling weight memory (~15 GB → ~30 GB in bf16) for two copies of
    numerically identical weights.

    Since PEFT's ``LoraLayer``s wrap (rather than replace) the base
    ``nn.Linear`` modules, the underlying base weights are still present
    inside ``blade_model`` and reachable via ``disable_adapter()``. This
    view makes that accessible with the same call signature
    (``model(input_ids=..., attention_mask=...) -> outputs.logits``) that
    ``DPOBlade`` / ``compute_logprob`` / ``compute_logprobs_batched``
    already expect, so it's a drop-in replacement for a separately loaded
    verifier/base model — no other code needs to change.

    Usage
    -----
    >>> blade_model = load_blade_model(cfg, "harmlessness")
    >>> verifier_model = DisabledAdapterView(blade_model)  # no 2nd copy
    """

    def __init__(self, peft_model):
        self._m = peft_model

    def __call__(self, *args, **kwargs):
        with self._m.disable_adapter():
            return self._m(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self._m.disable_adapter():
            return self._m.generate(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._m.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._m.named_parameters(*args, **kwargs)

    def eval(self):
        self._m.eval()
        return self

    def to(self, *args, **kwargs):
        self._m.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self._m.parameters()).device

    @property
    def config(self):
        return self._m.config


def load_verifier_model_shared(blade_model: PeftModel) -> "DisabledAdapterView":
    """Preferred alternative to ``load_verifier_model``/``load_base_model``
    when a blade model is already loaded: reuses its weights via
    ``DisabledAdapterView`` instead of downloading and loading a second
    ~7.6B-parameter copy onto the GPU. Saves ~15 GB VRAM in bf16."""
    return DisabledAdapterView(blade_model)


def load_all(
    cfg: SwissKnifeConfig,
    blade_name: str,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, PeftModel]:
    """Convenience: load tokenizer, base model, and one blade in one call."""
    tokenizer   = load_tokenizer(cfg)
    base_model  = load_base_model(cfg)
    # If shared_base_model is True, base_model becomes a PeftModel here
    blade_model = load_blade_model(cfg, blade_name, base_model=base_model if getattr(cfg, "shared_base_model", False) else None)
    return tokenizer, base_model, blade_model


def load_drafter_tokenizer(cfg: SwissKnifeConfig) -> PreTrainedTokenizer:
    """Load the tokenizer for the drafter model."""
    logger.info("Loading drafter tokenizer from: %s", cfg.drafter_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.drafter_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_drafter_model(cfg: SwissKnifeConfig) -> PreTrainedModel:
    """Load the frozen drafter model (π_S)."""
    dtype = _resolve_dtype(cfg)
    device = _resolve_device(cfg)
    logger.info("Loading drafter model: %s  [dtype=%s, device=%s]", cfg.drafter_model_id, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.drafter_model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_verifier_tokenizer(cfg: SwissKnifeConfig) -> PreTrainedTokenizer:
    """Load the tokenizer for the verifier model."""
    # Build custom config with verifier model settings
    temp_cfg = SwissKnifeConfig(
        base_model_id=cfg.verifier_model_id,
        base_model_subfolder=cfg.verifier_model_subfolder,
        base_model_repo_type=cfg.verifier_model_repo_type
    )
    return load_tokenizer(temp_cfg)


def load_verifier_model(cfg: SwissKnifeConfig) -> PreTrainedModel:
    """Load the verifier model (π_B)."""
    temp_cfg = SwissKnifeConfig(
        base_model_id=cfg.verifier_model_id,
        base_model_subfolder=cfg.verifier_model_subfolder,
        base_model_repo_type=cfg.verifier_model_repo_type
    )
    return load_base_model(temp_cfg)


def load_rrm_model(cfg: SwissKnifeConfig) -> PreTrainedModel:
    """Load the Reward Reasoning Model (RRM) judge model."""
    dtype = _resolve_dtype(cfg)
    device = _resolve_device(cfg)
    logger.info("Loading RRM model: %s  [dtype=%s, device=%s]", cfg.rrm_model_id, dtype, device)
    attn_kwargs = {}
    if dtype in [torch.float16, torch.bfloat16] and torch.cuda.is_available():
        try:
            import flash_attn
            attn_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        cfg.rrm_model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        **attn_kwargs,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_rrm_tokenizer(cfg: SwissKnifeConfig) -> PreTrainedTokenizer:
    """Load the tokenizer for the RRM judge model.

    RRM-7B is a Qwen2.5-based model and exposes its own tokenizer directly
    from the model repo. This loader fetches it explicitly rather than sharing
    the verifier tokenizer, so that special tokens (pad, eos) are correctly
    configured for the RRM model.
    """
    logger.info("Loading RRM tokenizer from: %s", cfg.rrm_model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.rrm_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer

