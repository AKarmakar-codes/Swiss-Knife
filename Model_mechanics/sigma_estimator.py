"""
Swiss Knife — Sigma Estimator and Running Percentile Threshold Calibrator
========================================================================

Implements:
1. Selective activation/deactivation of PEFT LoRA dropout layers at test time.
2. RunningPercentileThreshold for calibrating u as the sum of empirical
   percentiles of the blade reward and KL terms separately.
3. estimate_mu_sigma for candidates under different sigma_modes.
"""

import math
import logging
import torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def set_lora_dropout(model: torch.nn.Module, training: bool = True, dropout_p: Optional[float] = None) -> dict:
    """Selectively toggle only the LoRA dropout layers to train/eval mode.

    Parameters
    ----------
    model : torch.nn.Module
        The model containing PEFT LoRA layers.
    training : bool
        If True, set dropout layers to train mode. If False, set to eval mode.
    dropout_p : float, optional
        If provided, override the dropout probability of the layers.

    Returns
    -------
    dict
        A mapping from dropout module reference to its original dropout probability p,
        allowing restoration if needed.
    """
    original_ps = {}
    for module in model.modules():
        # Check for LoraLayer or any module containing lora_dropout
        if module.__class__.__name__ == "LoraLayer" or hasattr(module, "lora_dropout"):
            if hasattr(module, "lora_dropout") and isinstance(module.lora_dropout, torch.nn.ModuleDict):
                for key, drop in module.lora_dropout.items():
                    if isinstance(drop, torch.nn.Dropout):
                        original_ps[drop] = drop.p
                        if training:
                            drop.train()
                            if dropout_p is not None:
                                drop.p = dropout_p
                        else:
                            drop.eval()
    return original_ps


def restore_lora_dropout(original_ps: dict):
    """Restore original dropout probabilities to overridden dropout modules."""
    for drop, orig_p in original_ps.items():
        drop.p = orig_p


class RunningPercentileThreshold:
    """Calibrates threshold u as a percentile of the empirical distribution

    on a running slice, computed for the blade reward and KL terms separately:
        u = percentile(B_blade, p) + percentile(B_kl, p)
    """

    def __init__(self, percentile: float = 10.0, buffer_size: int = 20, initial_threshold: float = -1e6):
        self.percentile = percentile
        self.buffer_size = buffer_size
        self.initial_threshold = initial_threshold
        self.buffer_blade = []
        self.buffer_kl = []

    def get_threshold(self, default_threshold: float) -> float:
        """Get the current calibrated threshold, falling back to default if buffers are small."""
        if len(self.buffer_blade) < 5:
            return default_threshold

        p_blade = float(np.percentile(self.buffer_blade, self.percentile))
        p_kl = float(np.percentile(self.buffer_kl, self.percentile))
        calibrated = p_blade + p_kl
        logger.debug(
            "Calibrated threshold: %.4f (blade 10th=%.4f, kl 10th=%.4f, buffers=%d)",
            calibrated, p_blade, p_kl, len(self.buffer_blade)
        )
        return calibrated

    def update(self, r_blade_val: float, kl_val: float):
        """Update the running buffers with a new accepted step's terms."""
        self.buffer_blade.append(r_blade_val)
        self.buffer_kl.append(kl_val)
        if len(self.buffer_blade) > self.buffer_size:
            self.buffer_blade.pop(0)
            self.buffer_kl.pop(0)


def estimate_mu_sigma(
    prefix_ids: torch.Tensor,
    step_token_ids_list: List[torch.Tensor],
    blade,
    sigma_mode: str = "none",
    K: int = 5,
    dropout_p: Optional[float] = None,
    draft_logprobs: Optional[torch.Tensor] = None,
    verifier_logprobs: Optional[torch.Tensor] = None,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate mu (mean reward) and sigma (uncertainty) for each candidate.

    Parameters
    ----------
    prefix_ids : torch.Tensor
        Tokenized prompt + accepted history.
    step_token_ids_list : list of torch.Tensor
        Candidate steps.
    blade : DPOBlade
        The DPO blade containing verifier and blade models.
    sigma_mode : str
        One of "none", "mc_dropout", or "log_ratio_proxy".
    K : int
        Number of forward passes for "mc_dropout".
    dropout_p : float, optional
        Custom test-time dropout probability for "mc_dropout".
    draft_logprobs : torch.Tensor, optional
        Precomputed draft log-probabilities. Needed for "log_ratio_proxy".
    verifier_logprobs : torch.Tensor, optional
        Precomputed verifier log-probabilities. Needed for "log_ratio_proxy".
    beta : float
        Implicit reward scaling factor beta.

    Returns
    -------
    mu : torch.Tensor
        Shape ``[n]`` - clean DPO blade rewards.
    sigma : torch.Tensor
        Shape ``[n]`` - uncertainty estimate per step.
    """
    n = len(step_token_ids_list)
    device = prefix_ids.device

    if n == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    # 1. Always compute clean mu under standard eval mode
    # Ensure lora dropout is disabled first
    set_lora_dropout(blade.blade_model, training=False)
    mu = blade.score_reasoning_steps(prefix_ids, step_token_ids_list)

    if sigma_mode == "none":
        sigma = torch.zeros_like(mu)
        return mu, sigma

    elif sigma_mode == "log_ratio_proxy":
        if draft_logprobs is None:
            raise ValueError("draft_logprobs must be provided in log_ratio_proxy mode")

        # Compute verifier logprobs if not already precomputed
        if verifier_logprobs is None:
            # Import compute_logprob helper
            from evaluation.retokenisation_llama_to_qwen import compute_logprob
            verifier_logprobs_list = []
            prefix_ids_squeezed = prefix_ids.squeeze(0)
            for step_ids in step_token_ids_list:
                verifier_lp = compute_logprob(blade.base_model, prefix_ids_squeezed, step_ids)
                verifier_logprobs_list.append(verifier_lp)
            verifier_logprobs = torch.tensor(verifier_logprobs_list, dtype=torch.float, device=device)

        # Formula: sigma = | r_blade - (1/beta)*(log pi_verifier - log pi_draft) |
        sigma = (mu - (verifier_logprobs - draft_logprobs) / beta).abs()
        return mu, sigma

    elif sigma_mode == "mc_dropout":
        # Enable LoRA dropout layers
        original_ps = set_lora_dropout(blade.blade_model, training=True, dropout_p=dropout_p)
        try:
            samples = []
            for _ in range(K):
                sample_rewards = blade.score_reasoning_steps(prefix_ids, step_token_ids_list)
                samples.append(sample_rewards)
            
            samples_tensor = torch.stack(samples, dim=0) # [K, n]
            # Compute empirical standard deviation
            sigma = samples_tensor.std(dim=0, unbiased=True)
        finally:
            # Disable LoRA dropout and restore original probabilities
            set_lora_dropout(blade.blade_model, training=False)
            restore_lora_dropout(original_ps)

        return mu, sigma

    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")
