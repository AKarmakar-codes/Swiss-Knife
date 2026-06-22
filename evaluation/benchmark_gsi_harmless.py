"""
Swiss Knife — GSI Retokenisation and Logprob Utilities
======================================================

Contains the shared logic for aligning mismatched tokenizers and computing
step-level log probabilities. Used across all GSI decoding strategies.
"""

import torch

def compute_logprob(model, prefix_ids, step_ids):
    """Compute log-probability of step_ids conditioned on prefix_ids.
    
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
        The sum of log-probabilities of step tokens.
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
    
    # Gather step_ids probabilities
    step_logprobs = log_probs.gather(dim=-1, index=step_ids.unsqueeze(-1)).squeeze(-1)
    return step_logprobs.sum().item()


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
