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
    step_len = step_ids.shape[0]
    full_ids = torch.cat([prefix_ids, step_ids]).unsqueeze(0)  # [1, prefix_len + step_len]
    attention_mask = torch.ones_like(full_ids)

    logits_to_keep = step_len + 1
    with torch.no_grad():
        try:
            outputs = model(input_ids=full_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep)
            logits = outputs.logits.squeeze(0)  # [logits_to_keep, vocab_size]
            step_logits = logits[:step_len].float()
        except TypeError:
            outputs = model(input_ids=full_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0)  # [prefix_len + step_len, vocab_size]
            pred_positions = torch.arange(
                prefix_len - 1,
                prefix_len + step_len - 1,
                device=prefix_ids.device
            )
            step_logits = logits[pred_positions].float()

    step_logsumexp = torch.logsumexp(step_logits, dim=-1)
    step_targets = step_logits.gather(dim=-1, index=step_ids.unsqueeze(-1)).squeeze(-1)
    step_logprobs = step_targets - step_logsumexp
    del logits, outputs
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
    logits_to_keep = max_step_len + 1

    # Create padded batch and attention mask
    batch_ids = torch.full((batch_size, full_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, full_len), dtype=torch.long, device=device)

    for i, step_ids in enumerate(step_ids_list):
        step_len = step_ids.shape[0]
        batch_ids[i, :prefix_len] = prefix_ids
        attention_mask[i, :prefix_len] = 1
        if step_len > 0:
            batch_ids[i, prefix_len : prefix_len + step_len] = step_ids
            attention_mask[i, prefix_len : prefix_len + step_len] = 1

    with torch.no_grad():
        try:
            outputs = model(input_ids=batch_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep)
            logits = outputs.logits  # [batch_size, logits_to_keep, vocab_size]
            use_kept = True
        except TypeError:
            outputs = model(input_ids=batch_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, full_len, vocab_size]
            use_kept = False

    mean_logprobs = []
    for i, step_ids in enumerate(step_ids_list):
        step_len = step_ids.shape[0]
        if step_len == 0:
            mean_logprobs.append(0.0)
            continue

        if use_kept:
            row_logits = logits[i, :step_len, :].float()
        else:
            pred_positions = torch.arange(
                prefix_len - 1,
                prefix_len - 1 + step_len,
                device=device
            )
            row_logits = logits[i, pred_positions, :].float()

        row_logsumexp = torch.logsumexp(row_logits, dim=-1)
        row_targets = row_logits.gather(dim=-1, index=step_ids.unsqueeze(-1)).squeeze(-1)
        row_logprobs = row_targets - row_logsumexp
        mean_logprobs.append(row_logprobs.mean().item())

    del logits, outputs
    return mean_logprobs
