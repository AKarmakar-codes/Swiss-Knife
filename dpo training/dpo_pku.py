"""
DPO on PKU-SafeRLHF with selectable objective.

OBJECTIVE ∈ {"safety", "helpfulness", "harmlessness"}
  - safety:       chosen = response with safer_response_id
  - helpfulness:  chosen = response with better_response_id
  - harmlessness: keep only pairs where exactly one response is safe;
                  chosen = the safe one

Run three times with different OBJECTIVE + OUTPUT_DIR to get three adapters
sharing the same SFT-merged base.
"""

import os
import sys
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OBJECTIVE = os.environ.get("OBJECTIVE", "safety")   # safety | helpfulness | harmlessness
assert OBJECTIVE in {"safety", "helpfulness", "harmlessness"}

MERGED_MODEL_DIR = "/root/ndna/SFT/Qwen_SFT_merged"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"

OUTPUT_DIR = f"./qwen25_dpo_{OBJECTIVE}"
EVAL_SPLIT_RATIO = 0.05

NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
DPO_BETA = 0.1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 25
SEED = 42
DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
         else torch.float16)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(f"dpo_{OBJECTIVE}.log", mode="a"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"OBJECTIVE = {OBJECTIVE}")
logger.info(f"OUTPUT_DIR = {OUTPUT_DIR}")

# --------------------------------------------------------------------------
# Tokenizer + model
# --------------------------------------------------------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

tokenizer.pad_token = tokenizer.pad_token or "<|endoftext|>"
tokenizer.eos_token = "<|im_end|>"
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_DIR,
    torch_dtype=DTYPE,
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.config.use_cache = False

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
raw = load_dataset(DATASET_NAME, split="train")
logger.info(f"Raw PKU-SafeRLHF size: {len(raw)}")

def pick_chosen_rejected(example):
    r0, r1 = example["response_0"], example["response_1"]

    if OBJECTIVE == "safety":
        chosen_id = example["safer_response_id"]
    elif OBJECTIVE == "helpfulness":
        chosen_id = example["better_response_id"]
    elif OBJECTIVE == "harmlessness":
        s0, s1 = example["is_response_0_safe"], example["is_response_1_safe"]
        # Keep only pairs with clear safe/unsafe disagreement
        if s0 == s1:
            return {"prompt": "", "chosen": "", "rejected": ""}
        chosen_id = 0 if s0 else 1
    else:
        raise ValueError(OBJECTIVE)

    chosen = r0 if chosen_id == 0 else r1
    rejected = r1 if chosen_id == 0 else r0

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}

dataset = raw.map(pick_chosen_rejected, remove_columns=raw.column_names,
                  desc=f"Building DPO pairs ({OBJECTIVE})", num_proc=4)
dataset = dataset.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"]
                                   and x["chosen"] != x["rejected"])
logger.info(f"Valid pairs for {OBJECTIVE}: {len(dataset)}")

dataset = dataset.shuffle(seed=SEED)
split = dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=SEED)
train_dataset, eval_dataset = split["train"], split["test"]
logger.info(f"Train={len(train_dataset)}  Eval={len(eval_dataset)}")

# --------------------------------------------------------------------------
# LoRA + DPO config
# --------------------------------------------------------------------------
peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type=TaskType.CAUSAL_LM,
    target_modules=LORA_TARGET_MODULES,
)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    beta=DPO_BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=(DTYPE == torch.bfloat16),
    fp16=(DTYPE == torch.float16),
    save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=3,
    eval_strategy="steps", eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/tb_logs",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    seed=SEED,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.model.print_trainable_parameters()

# --------------------------------------------------------------------------
# Train (resume-aware)
# --------------------------------------------------------------------------
resume = None
if os.path.exists(OUTPUT_DIR):
    ckpts = sorted([d for d in os.listdir(OUTPUT_DIR)
                    if d.startswith("checkpoint-")
                    and os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    if ckpts:
        resume = os.path.join(OUTPUT_DIR, ckpts[-1])
        logger.info(f"Resuming from {resume}")

trainer.train(resume_from_checkpoint=resume)

trainer.save_model(f"{OUTPUT_DIR}/final_dpo_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_dpo_adapter")
logger.info(f"Done. Adapter -> {OUTPUT_DIR}/final_dpo_adapter")
