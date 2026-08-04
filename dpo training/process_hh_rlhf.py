"""
Process Anthropic HH-RLHF into DPO-ready JSONL files with train/eval splits.

Outputs (into dpo_datasets/):
  hh_helpfulness_train.jsonl   hh_helpfulness_eval.jsonl
  hh_harmlessness_train.jsonl  hh_harmlessness_eval.jsonl

Usage:
    python process_hh_rlhf.py
"""

import json
import os
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATASET_NAME = "Anthropic/hh-rlhf"
SUBSETS = ["harmless-base", "helpful-base"]

OUTPUT_DIR = "./dpo_datasets"

# Map HH subset names to objective names matching the rest of the pipeline
SUBSET_TO_OBJECTIVE = {
    "harmless-base": "hh_harmlessness",
    "helpful-base": "hh_helpfulness",
}

EVAL_SPLIT_RATIO = 0.05
SEED = 42
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

# -----------------------------
# Load tokenizer (for chat template)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

tokenizer.pad_token = tokenizer.pad_token or "<|endoftext|>"
tokenizer.eos_token = "<|im_end|>"
tokenizer.padding_side = "right"


# -----------------------------
# Helpers
# -----------------------------
def parse_hh_conversation(text):
    """
    Convert raw HH-RLHF string into [(role, content), ...].

    Raw format:  "\\n\\nHuman: msg\\n\\nAssistant: msg\\n\\nHuman: msg..."
    Some entries start with "Human: " without leading newlines — we handle both.
    """
    # Normalise: ensure the text starts with \n\nHuman: so the split is uniform
    text = text.strip()
    if text.startswith("Human: "):
        text = "\n\n" + text

    turns = []
    parts = text.split("\n\nHuman: ")

    for part in parts:
        if not part.strip():
            continue

        sub_parts = part.split("\n\nAssistant: ")

        for i, sp in enumerate(sub_parts):
            sp = sp.strip()
            if not sp:
                continue

            # Safety net: strip any residual "Human: " or "Assistant: " prefix
            if i == 0:
                sp = re.sub(r"^Human:\s*", "", sp)
                turns.append(("user", sp))
            else:
                sp = re.sub(r"^Assistant:\s*", "", sp)
                turns.append(("assistant", sp))

    return turns


def convert_example(example):
    """
    Convert one HH example into DPO format.
    """
    chosen_turns = parse_hh_conversation(example["chosen"])
    rejected_turns = parse_hh_conversation(example["rejected"])

    if len(chosen_turns) < 2 or len(rejected_turns) < 2:
        return None

    if chosen_turns[-1][0] != "assistant":
        return None

    if rejected_turns[-1][0] != "assistant":
        return None

    # Prompt = all turns except the last assistant turn
    prompt_msgs = [
        {"role": role, "content": content}
        for role, content in chosen_turns[:-1]
    ]

    chosen_resp = chosen_turns[-1][1]
    rejected_resp = rejected_turns[-1][1]

    if not chosen_resp or not rejected_resp:
        return None

    # Filter out rows where the response is actually a misidentified Human turn
    if chosen_resp.startswith("Human:") or rejected_resp.startswith("Human:"):
        return None

    # Apply chat template
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt_text,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
        "source": "hh-rlhf",
    }


# -----------------------------
# Main processing loop
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for subset in SUBSETS:
    objective = SUBSET_TO_OBJECTIVE[subset]
    print(f"\n{'='*60}")
    print(f"Loading subset: {subset} -> {objective}")

    # Load both train and test splits from HH-RLHF
    train_raw = load_dataset(DATASET_NAME, data_dir=subset, split="train")
    test_raw = load_dataset(DATASET_NAME, data_dir=subset, split="test")

    print(f"Raw sizes — train: {len(train_raw)}, test: {len(test_raw)}")

    # Process train split
    train_records = []
    for ex in tqdm(train_raw, desc=f"{objective} train"):
        item = convert_example(ex)
        if item is not None:
            train_records.append(item)

    # Process test/eval split
    eval_records = []
    for ex in tqdm(test_raw, desc=f"{objective} eval"):
        item = convert_example(ex)
        if item is not None:
            eval_records.append(item)

    # Write train file
    train_path = os.path.join(OUTPUT_DIR, f"{objective}_train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write eval file
    eval_path = os.path.join(OUTPUT_DIR, f"{objective}_eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_records)} -> {train_path}")
    print(f"Eval:  {len(eval_records)} -> {eval_path}")

print(f"\nDone. All files saved to {OUTPUT_DIR}/")
