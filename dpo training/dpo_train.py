"""
DPO training on pre-processed chat-template datasets from dpo_datasets/.

Usage:
    # PKU / UltraFeedback objectives (single file, auto train/eval split)
    python dpo_train.py --objective helpfulness

    # HH-RLHF objectives (separate train + eval files)
    python dpo_train.py --objective hh_harmlessness
    python dpo_train.py --objective hh_helpfulness

Available objectives:
    helpfulness      - PKU-SafeRLHF (safe,safe) pairs; chosen = better_response_id
    harmlessness     - PKU-SafeRLHF (safe,unsafe) pairs; chosen = safe response
    informativeness  - UltraFeedback truthfulness axis
    hh_helpfulness   - Anthropic HH-RLHF helpful-base
    hh_harmlessness  - Anthropic HH-RLHF harmless-base
"""

import argparse
import os
import sys
import json
import logging
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DPO training from local JSONL datasets")
parser.add_argument("--objective", type=str, required=True,
                    choices=["helpfulness", "harmlessness", "informativeness",
                             "hh_helpfulness", "hh_harmlessness"],
                    help="Which objective dataset to train on")
parser.add_argument("--model_path", type=str, default="/root/ndna/SFT/Qwen_SFT_merged",
                    help="Path to base / SFT-merged model")
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Fallback tokenizer if model_path tokenizer fails")
parser.add_argument("--data_dir", type=str, default="./dpo_datasets",
                    help="Directory containing JSONL dataset files")
parser.add_argument("--output_root", type=str, default="./dpo_out",
                    help="Root output directory")
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--max_prompt_length", type=int, default=512)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--eval_ratio", type=float, default=0.05,
                    help="Eval split ratio (only used when no separate eval file exists)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Paths & logging
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(args.output_root, args.objective)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Objective: {args.objective}")
logger.info(f"Output:    {OUTPUT_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# Dtype
# ──────────────────────────────────────────────────────────────────────────────
DTYPE = (torch.bfloat16
         if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
         else torch.float16)

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    logger.info(f"Loaded tokenizer from {args.model_path}")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    logger.info(f"Loaded fallback tokenizer from {args.tokenizer_name}")

tokenizer.pad_token = tokenizer.pad_token or "<|endoftext|>"
tokenizer.eos_token = "<|im_end|>"
tokenizer.padding_side = "right"

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
logger.info(f"Loading model from {args.model_path} …")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=DTYPE,
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.config.use_cache = False
logger.info("Model loaded.")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load a JSONL file into a list of dicts, filtering broken rows."""
    records = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chosen = row["chosen"].strip()
            rejected = row["rejected"].strip()
            # Skip rows where chosen/rejected is a misidentified Human turn
            if chosen.startswith("Human:") or rejected.startswith("Human:"):
                skipped += 1
                continue
            # Skip degenerate pairs
            if chosen == rejected:
                skipped += 1
                continue
            records.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    if skipped:
        logger.info(f"  Filtered {skipped} broken/degenerate rows from {path}")
    return records


# Check for separate train/eval files (hh_* objectives) or single file (pku/ultrafeedback)
train_path = os.path.join(args.data_dir, f"{args.objective}_train.jsonl")
eval_path = os.path.join(args.data_dir, f"{args.objective}_eval.jsonl")
single_path = os.path.join(args.data_dir, f"{args.objective}.jsonl")

if os.path.exists(train_path) and os.path.exists(eval_path):
    # Separate train + eval files (HH-RLHF)
    logger.info(f"Loading train: {train_path}")
    logger.info(f"Loading eval:  {eval_path}")
    train_dataset = Dataset.from_list(load_jsonl(train_path)).shuffle(seed=args.seed)
    eval_dataset = Dataset.from_list(load_jsonl(eval_path)).shuffle(seed=args.seed)

elif os.path.exists(single_path):
    # Single file — auto split (PKU / UltraFeedback)
    logger.info(f"Loading dataset: {single_path}")
    dataset = Dataset.from_list(load_jsonl(single_path)).shuffle(seed=args.seed)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

else:
    logger.error(f"No dataset found. Looked for:")
    logger.error(f"  {train_path} + {eval_path}")
    logger.error(f"  {single_path}")
    sys.exit(1)

logger.info(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

# Print a sample for verification
ex = train_dataset[0]
logger.info(f"Sample prompt (first 200 chars): {ex['prompt'][:200]}")
logger.info(f"Sample chosen (first 200 chars): {ex['chosen'][:200]}")
logger.info(f"Sample rejected (first 200 chars): {ex['rejected'][:200]}")

# ──────────────────────────────────────────────────────────────────────────────
# LoRA config
# ──────────────────────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# ──────────────────────────────────────────────────────────────────────────────
# DPO training config
# ──────────────────────────────────────────────────────────────────────────────
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    beta=args.beta,
    max_length=args.max_length,
    learning_rate=args.lr,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=(DTYPE == torch.bfloat16),
    fp16=(DTYPE == torch.float16),
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=25,
    report_to="tensorboard",
    logging_dir=os.path.join(OUTPUT_DIR, "tb_logs"),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    seed=args.seed,
    remove_unused_columns=False,
)

est_steps = len(train_dataset) // (args.batch_size * args.grad_accum)
logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
logger.info(f"DPO config: lr={args.lr}, beta={args.beta}, max_length={args.max_length}")
logger.info(f"Estimated training steps: {est_steps}")

# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
trainer = DPOTrainer(
    model=model,
    ref_model=None,          # with LoRA, DPOTrainer uses the frozen base as ref
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.model.print_trainable_parameters()

# ──────────────────────────────────────────────────────────────────────────────
# Train (resume-aware)
# ──────────────────────────────────────────────────────────────────────────────
resume = None
if os.path.exists(OUTPUT_DIR):
    ckpts = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ])
    if ckpts:
        resume = os.path.join(OUTPUT_DIR, ckpts[-1])
        logger.info(f"Resuming from {resume}")

trainer.train(resume_from_checkpoint=resume)

# ──────────────────────────────────────────────────────────────────────────────
# Save final adapter
# ──────────────────────────────────────────────────────────────────────────────
final_path = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
logger.info(f"Done. Adapter saved to {final_path}")
