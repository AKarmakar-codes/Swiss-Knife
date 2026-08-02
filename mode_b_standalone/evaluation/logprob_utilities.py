"""
Swiss Knife — GSI Retokenisation and Logprob Utilities
======================================================

Contains the shared logic for aligning mismatched tokenizers and computing
step-level log probabilities. Used across all GSI decoding strategies.
"""

import torch

def compute_logprob(model, prefix_ids, step_ids):
    """Compute the mean per-token log-probability of step_ids conditioned on prefix_ids.

    Returns the **mean** (not sum) over step tokens so that the tilted reward
    penalty ``(1/β) * (qwen_lp - draft_lp)`` is independent of step length.
    Without length normalization, long steps accumulate large negative log-prob
    sums and are almost always rejected by the threshold, causing the override
    rate to vary wildly with prompt difficulty regardless of β or u.

    Parameters
    ----------
    model : PreTrainedModel
    prefix_ids : torch.Tensor
        1D tensor of prefix token IDs.
    step_ids : torch.Tensor
        1D tensor of step token IDs.

    Returns
    -------
    float
        Mean log-probability per token of the step.
    """
    if step_ids.shape[0] == 0:
        return 0.0
    prefix_len = prefix_ids.shape[0]
    # Concatenate prefix and step token IDs
    full_ids = torch.cat([prefix_ids, step_ids]).unsqueeze(0)  # [1, prefix_len + step_len]
    attention_mask = torch.ones_like(full_ids)

    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)  # [prefix_len + step_len, vocab_size]

    # The logit at index t predicts token at index t+1.
    pred_positions = torch.arange(
        prefix_len - 1,
        prefix_len + step_ids.shape[0] - 1,
        device=prefix_ids.device
    )
    log_probs = torch.log_softmax(logits[pred_positions].float(), dim=-1)

    # Gather step token log-probabilities and return their mean
    step_logprobs = log_probs.gather(dim=-1, index=step_ids.unsqueeze(-1)).squeeze(-1)
    return step_logprobs.mean().item()  # per-token mean, not sum


def retokenize_step(tokenizer, prefix_text, step_text, prefix_ids, device):
    """Retokenize a step text and extract step IDs for the target tokenizer.
    
    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Target tokenizer (e.g. verifier tokenizer).
    prefix_text : str
        The prefix text.
    step_text : str
        The step text to append and tokenize.
    prefix_ids : torch.Tensor
        1D tensor of target prefix token IDs.
    device : torch.device or str
        Device to map tensors to.
        
    Returns
    -------
    torch.Tensor
        1D tensor of step token IDs under the target tokenizer.
    """
    full_ids = tokenizer.encode(
        prefix_text + step_text, add_special_tokens=True, return_tensors="pt"
    ).squeeze(0).to(device)
    
    if full_ids.shape[0] <= prefix_ids.shape[0]:
        step_ids = torch.tensor([], dtype=torch.long, device=device)
    else:
        step_ids = full_ids[prefix_ids.shape[0]:]
    return step_ids


def compute_logprobs_batched(model, prefix_ids, step_ids_list, pad_token_id=0):
    """Compute the mean per-token log-probability of multiple step_ids conditioned on prefix_ids in a single batch.

    Parameters
    ----------
    model : PreTrainedModel
    prefix_ids : torch.Tensor
        1D tensor of prefix token IDs.
    step_ids_list : list of torch.Tensor
        List of 1D tensors of step token IDs.
    pad_token_id : int
        Token ID to use for padding.

    Returns
    -------
    list of float
        List of mean log-probabilities per step.
    """
    if len(step_ids_list) == 0:
        return []

    prefix_len = prefix_ids.shape[0]
    device = prefix_ids.device

    # Find max step length
    max_step_len = max(step_ids.shape[0] for step_ids in step_ids_list)
    if max_step_len == 0:
        return [0.0] * len(step_ids_list)

    batch_size = len(step_ids_list)
    full_len = prefix_len + max_step_len

    # Create padded batch and attention mask
    batch_ids = torch.full((batch_size, full_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, full_len), dtype=torch.long, device=device)

    for i, step_ids in enumerate(step_ids_list):
        step_len = step_ids.shape[0]
        # Copy prefix
        batch_ids[i, :prefix_len] = prefix_ids
        attention_mask[i, :prefix_len] = 1
        # Copy step
        if step_len > 0:
            batch_ids[i, prefix_len : prefix_len + step_len] = step_ids
            attention_mask[i, prefix_len : prefix_len + step_len] = 1

    with torch.no_grad():
        outputs = model(input_ids=batch_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, full_len, vocab_size]

    # The logit at index t predicts token at index t+1.
    pred_positions = torch.arange(
        prefix_len - 1,
        full_len - 1,
        device=device
    )
    
    # [batch_size, max_step_len, vocab_size]
    step_logits = logits[:, pred_positions, :]
    log_probs = torch.log_softmax(step_logits.float(), dim=-1)

    mean_logprobs = []
    for i, step_ids in enumerate(step_ids_list):
        step_len = step_ids.shape[0]
        if step_len == 0:
            mean_logprobs.append(0.0)
            continue
            
        cand_log_probs = log_probs[i, :step_len, :]
        cand_step_logprobs = cand_log_probs.gather(dim=-1, index=step_ids.unsqueeze(-1)).squeeze(-1)
        mean_logprobs.append(cand_step_logprobs.mean().item())

    return mean_logprobs
