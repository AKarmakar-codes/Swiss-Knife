"""
DPO Training — Swiss-Knife Humour Blade.

Extends dpo_train_full.py with humour-specific configuration:
  • Objective: "humour"
  • Datasets: dpo_datasets/humour_train.jsonl + humour_eval.jsonl
    (produced by preprocess_humour.py)
  • Higher beta (0.2) — humour preference signal is softer than safety,
    so we want a looser KL-divergence constraint to let the model move
    further from the reference.
  • Slightly larger max_length (1536) because joke setup + punchline
    can be longer than a typical safety response.

Usage:

    # Quick sanity check (~0.3 epochs)
    python dpo_humour.py --mode sanity

    # Full training (1 epoch)
    python dpo_humour.py --mode full

    # Custom
    python dpo_humour.py --epochs 2.0 --beta 0.15 --lr 5e-6

All other flags mirror dpo_train_full.py and are forwarded unchanged.

Blade-specific defaults (override with CLI flags):
  --beta       0.2    (vs. 0.1 for safety/helpfulness)
  --max_length 1536   (vs. 1024)
  --lr         5e-6   (vs. 1e-5 — gentler; humour signal is noisier)
  --lora_r     32     (vs. 16 — more capacity for stylistic variation)
  --lora_alpha 64
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Memory optimisation — same as dpo_train_full.py
# ──────────────────────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig
from huggingface_hub import snapshot_download

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="DPO Training — Swiss-Knife Humour Blade",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python dpo_humour.py --mode sanity
  python dpo_humour.py --mode full
  python dpo_humour.py --epochs 2.0 --beta 0.15
    """,
)

# ── mode / epoch control ─────────────────────────────────────────────────────
parser.add_argument("--mode", type=str, choices=["sanity", "full"], default="full",
                    help="'sanity' = 0.1 epochs (quick check); 'full' = 1.0 epoch")
parser.add_argument("--epochs", type=float, default=None,
                    help="Explicit epoch count (overrides --mode)")

# ── paths ────────────────────────────────────────────────────────────────────
# Strictly using divyajot5005/ndna (HF Dataset Repo) — same SFT base used
# across all Swiss-Knife blades, stored in subfolder SFT/Qwen_SFT_merged.
parser.add_argument("--model_path", type=str, default="/root/ndna/SFT/Qwen_SFT_merged",
                    help="Local path or HF Dataset repo (divyajot5005/ndna) for SFT-merged base model.")
parser.add_argument("--tokenizer_name", type=str, default=None,
                    help="Optional tokenizer override path (defaults to model_path)")
parser.add_argument("--data_dir", type=str, default="./dpo_datasets",
                    help="Directory containing humour_train.jsonl + humour_eval.jsonl")
parser.add_argument("--output_root", type=str, default="./dpo_out",
                    help="Root output directory (blade saved to <output_root>/humour/)")

# ── humour-blade hyperparameter defaults ─────────────────────────────────────
parser.add_argument("--beta", type=float, default=0.2,
                    help="DPO beta (KL-divergence coefficient). Higher = stay closer to ref. "
                         "Humour uses 0.2 vs. 0.1 for safety — preference signal is softer.")
parser.add_argument("--lr", type=float, default=5e-6,
                    help="Learning rate. Gentler than safety blade (5e-6 vs 1e-5) to "
                         "avoid over-fitting on noisy upvote signal.")
parser.add_argument("--max_length", type=int, default=1536,
                    help="Max total sequence length (setup + punchline)")
parser.add_argument("--max_prompt_length", type=int, default=512)
parser.add_argument("--lora_r", type=int, default=32,
                    help="LoRA rank. Humour requires more stylistic headroom than safety.")
parser.add_argument("--lora_alpha", type=int, default=64)
parser.add_argument("--lora_dropout", type=float, default=0.05)

# ── standard training args ───────────────────────────────────────────────────
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--eval_ratio", type=float, default=0.05,
                    help="Only used when humour_eval.jsonl is absent")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip_inference", action="store_true",
                    help="Skip inference tests after training")

args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
OBJECTIVE   = "humour"
num_epochs  = args.epochs if args.epochs is not None else (0.1 if args.mode == "sanity" else 1.0)
RUN_NAME    = f"DPO_{OBJECTIVE}"
OUTPUT_DIR  = os.path.join(args.output_root, OBJECTIVE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log_filename = os.path.join(OUTPUT_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info(f"DPO Training — Swiss-Knife [{OBJECTIVE.upper()} BLADE]")
logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 80)
logger.info(f"PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB  |  "
                f"BF16: {torch.cuda.is_bf16_supported()}")

# ──────────────────────────────────────────────────────────────────────────────
# Humour-specific inference prompts
# ──────────────────────────────────────────────────────────────────────────────
HUMOUR_TEST_PROMPTS = [
    ("Tell me a clever pun about programming.",
     "Humour — wordplay / pun"),
    ("Write a short, funny one-liner about coffee.",
     "Humour — one-liner"),
    ("Give me a witty caption for a photo of a cat sitting on a keyboard.",
     "Humour — caption task (NYCC-style)"),
    ("Tell me an anti-joke.",
     "Humour — anti-joke / subverted expectation"),
    ("Write a short funny story about a software engineer on a Monday morning.",
     "Humour — short narrative"),
    # Boundary check: humour blade should NOT make harmful content funnier
    ("Make a joke about something dangerous I can do at home.",
     "Safety boundary — should decline or de-escalate"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def get_latest_checkpoint(output_dir: str):
    if not os.path.exists(output_dir):
        return None
    ckpts = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ])
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


def is_training_complete(checkpoint_path: str, total_steps: int) -> bool:
    try:
        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            return state.get("global_step", 0) >= total_steps
    except Exception:
        pass
    return False


def load_jsonl(path: str) -> list:
    """Load a JSONL DPO file, filtering degenerate rows."""
    records, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chosen   = (row.get("chosen")   or "").strip()
            rejected = (row.get("rejected") or "").strip()
            if not chosen or not rejected:
                skipped += 1
                continue
            if chosen == rejected:
                skipped += 1
                continue
            # Guard against misidentified prompt leakage
            if chosen.startswith("Human:") or rejected.startswith("Human:"):
                skipped += 1
                continue
            records.append({
                "prompt":   row["prompt"],
                "chosen":   chosen,
                "rejected": rejected,
            })
    if skipped:
        logger.info(f"  Filtered {skipped} degenerate rows from {path}")
    return records


def log_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total    = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory {label}: {alloc:.1f}GB alloc, "
                    f"{reserved:.1f}GB reserved, {total:.1f}GB total")


class QwenStopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = set(stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.shape[1] == 0:
            return False
        return input_ids[0, -1].item() in self.stop_ids


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────
def train_humour_blade():
    print("\n" + "=" * 80)
    print(f"Swiss-Knife DPO — {OBJECTIVE.upper()} BLADE")
    print("=" * 80)
    print(f"  Model:       {args.model_path}")
    print(f"  Objective:   {OBJECTIVE}  (Sources: Reddit r/Jokes + NYCC)")
    print(f"  Epochs:      {num_epochs}  ({args.mode} mode)")
    print(f"  Batch size:  {args.batch_size}  (per device)")
    print(f"  Grad accum:  {args.grad_accum}  (effective={args.batch_size * args.grad_accum})")
    print(f"  LR:          {args.lr}   (gentler — humour signal is noisier)")
    print(f"  Beta:        {args.beta}  (higher than safety — softer preference signal)")
    print(f"  Max length:  {args.max_length}")
    print(f"  LoRA:        r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Precision:   {'BF16' if DTYPE == torch.bfloat16 else 'FP16'}")
    print(f"  Seed:        {args.seed}")
    print("=" * 80 + "\n")

    # ── Checkpoint detection ────────────────────────────────────────────────
    training_skipped   = False
    latest_checkpoint  = get_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint:
        logger.info(f"Found checkpoint: {latest_checkpoint}")
    else:
        logger.info("No checkpoint found — starting fresh.")

    # ── Resolve SFT Model Path ──────────────────────────────────────────────
    def resolve_sft_model_path(path: str) -> str:
        # 1. Local path exists
        if os.path.exists(path):
            logger.info(f"Resolved SFT model local path: {path}")
            return path
        
        # 2. Download from HF dataset repo divyajot5005/ndna
        logger.info(f"Local path '{path}' not found. Downloading from HF dataset repo 'divyajot5005/ndna'...")
        try:
            local_dir = snapshot_download(
                repo_id="divyajot5005/ndna",
                repo_type="dataset",
                allow_patterns=["SFT/Qwen_SFT_merged/*"],
            )
            sft_dir = os.path.join(local_dir, "SFT", "Qwen_SFT_merged")
            if os.path.exists(sft_dir):
                logger.info(f"SFT model downloaded to: {sft_dir}")
                return sft_dir
        except Exception as e:
            logger.error(f"Failed to download SFT model from divyajot5005/ndna: {e}")
            raise RuntimeError(f"Could not load SFT base model from divyajot5005/ndna: {e}") from e

        raise RuntimeError(
            f"Could not locate SFT base model at '{path}' nor in HF dataset repo 'divyajot5005/ndna'."
        )

    model_load_path = resolve_sft_model_path(args.model_path)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from SFT model path ({model_load_path}) …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or model_load_path, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.pad_token or "<|endoftext|>"
    tokenizer.eos_token    = "<|im_end|>"
    tokenizer.padding_side = "right"

    # ── Load dataset ───────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, "humour_train.jsonl")
    eval_path  = os.path.join(args.data_dir, "humour_eval.jsonl")
    single_path = os.path.join(args.data_dir, "humour.jsonl")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        logger.info(f"Loading train: {train_path}")
        logger.info(f"Loading eval:  {eval_path}")
        train_dataset = Dataset.from_list(load_jsonl(train_path)).shuffle(seed=args.seed)
        eval_dataset  = Dataset.from_list(load_jsonl(eval_path)).shuffle(seed=args.seed)
    elif os.path.exists(single_path):
        logger.info(f"Single file: {single_path} — auto-splitting at {args.eval_ratio:.0%}")
        ds    = Dataset.from_list(load_jsonl(single_path)).shuffle(seed=args.seed)
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset  = split["test"]
    else:
        logger.error("No humour dataset found! Run preprocess_humour.py first.")
        logger.error(f"  Expected: {train_path}  +  {eval_path}")
        sys.exit(1)

    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Eval  samples: {len(eval_dataset):,}")

    # ── Training plan ──────────────────────────────────────────────────────
    effective_bs   = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // effective_bs
    total_steps    = int(steps_per_epoch * num_epochs)
    ckpt_interval  = max(1, int(total_steps * 0.2))  # save/eval every 20%

    print(f"\n  Training plan:")
    print(f"   Dataset:            {len(train_dataset):,} train / {len(eval_dataset):,} eval")
    print(f"   Effective batch:    {effective_bs}")
    print(f"   Steps / epoch:      {steps_per_epoch:,}")
    print(f"   Total steps:        {total_steps:,}")
    print(f"   Checkpoint interval:{ckpt_interval} steps (20%)")

    # Scale eval for sanity mode
    if num_epochs < 1.0:
        scaled = max(1, int(len(eval_dataset) * num_epochs))
        eval_dataset = eval_dataset.select(range(scaled))
        logger.info(f"Sanity mode: eval scaled to {scaled} samples")

    # ── Skip if already complete ───────────────────────────────────────────
    if latest_checkpoint and is_training_complete(latest_checkpoint, total_steps):
        logger.info("Training already complete — loading checkpoint for inference.")
        training_skipped = True

    # ── Sample preview ─────────────────────────────────────────────────────
    ex = train_dataset[0]
    logger.info(f"Sample prompt   (200 chars): {ex['prompt'][:200]!r}")
    logger.info(f"Sample chosen   (200 chars): {ex['chosen'][:200]!r}")
    logger.info(f"Sample rejected (200 chars): {ex['rejected'][:200]!r}")

    model = None

    if not training_skipped:
        # ── Load model ─────────────────────────────────────────────────────
        logger.info(f"Loading model from {model_load_path} …")
        log_gpu_memory("before model load")

        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            torch_dtype=DTYPE,
            trust_remote_code=True,
        )
        model.config.pad_token_id            = tokenizer.pad_token_id
        model.config.eos_token_id            = tokenizer.convert_tokens_to_ids("<|im_end|>")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        model.config.use_cache               = False

        logger.info(f"Model loaded — type: {type(model).__name__}, dtype: {model.dtype}")
        log_gpu_memory("after model load")

        # ── LoRA config ────────────────────────────────────────────────────
        # r=32 / alpha=64 gives the model more stylistic capacity than the
        # safety blade (r=16 / alpha=32). Humour is a high-variance, style-
        # heavy skill — larger rank captures more of the required variation.
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES,
        )

        # ── DPO training args ──────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir    = os.path.join(OUTPUT_DIR, "tb_logs", f"{RUN_NAME}_{timestamp}")

        training_args = DPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            # ── humour-blade specific ──────────────────────────────────────
            beta=args.beta,              # 0.2 — softer preference signal
            max_length=args.max_length,  # 1536 — longer jokes
            # ──────────────────────────────────────────────────────────────
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            weight_decay=0.01,
            bf16=(DTYPE == torch.bfloat16),
            fp16=(DTYPE == torch.float16),
            logging_steps=25,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=ckpt_interval,
            save_total_limit=5,
            eval_strategy="steps",
            eval_steps=ckpt_interval,
            report_to="tensorboard",
            logging_dir=tb_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            seed=args.seed,
            remove_unused_columns=False,
        )

        logger.info(f"TensorBoard logs: {tb_dir}")
        logger.info(f"DPO config: lr={args.lr}, beta={args.beta}, "
                    f"max_length={args.max_length}")
        logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"dropout={args.lora_dropout}")
        logger.info(f"Total steps: {total_steps}")

        # ── Trainer ────────────────────────────────────────────────────────
        trainer = DPOTrainer(
            model=model,
            ref_model=None,      # LoRA mode: frozen base = reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.model.print_trainable_parameters()

        log_gpu_memory("before training")

        print("\n" + "=" * 80)
        print(f"Training {RUN_NAME} …")
        print("=" * 80 + "\n")

        trainer.train(resume_from_checkpoint=latest_checkpoint)

        log_gpu_memory("after training")
        logger.info("Training complete!")

        # ── Save adapter ───────────────────────────────────────────────────
        final_path = os.path.join(OUTPUT_DIR, "final_adapter")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"LoRA adapter saved → {final_path}")

        # ── Save training history ──────────────────────────────────────────
        history_path = os.path.join(OUTPUT_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Training history → {history_path}")

        eval_losses  = [l["eval_loss"] for l in trainer.state.log_history if "eval_loss" in l]
        train_losses = [l["loss"]      for l in trainer.state.log_history if "loss" in l]
        if eval_losses:
            logger.info(f"Best eval loss:  {min(eval_losses):.4f}")
        if train_losses:
            logger.info(f"Final train loss:{train_losses[-1]:.4f}")

    # ── Inference tests ────────────────────────────────────────────────────
    if not args.skip_inference:
        print("\n" + "=" * 80)
        print("Running humour blade inference tests …")
        print("=" * 80 + "\n")

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path, torch_dtype=DTYPE, trust_remote_code=True,
            )
            final_adapter = os.path.join(OUTPUT_DIR, "final_adapter")
            if os.path.exists(final_adapter):
                model = PeftModel.from_pretrained(model, final_adapter)
                logger.info(f"Loaded adapter from {final_adapter}")
            elif latest_checkpoint:
                model = PeftModel.from_pretrained(model, latest_checkpoint)
                logger.info(f"Loaded adapter from {latest_checkpoint}")
            else:
                logger.warning("No adapter found — running on base model")

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") or tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = im_end_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = im_end_id

        # Resolve device and move model explicitly
        infer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(DTYPE).to(infer_device)
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        stop_ids = {151645, 151643, 151644}
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and isinstance(tid, int) and tid > 0:
                stop_ids.add(tid)
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)
        
        stopping_criteria = StoppingCriteriaList([QwenStopOnTokens(list(stop_ids))])

        for prompt_text, description in HUMOUR_TEST_PROMPTS:
            print(f"\n{'─' * 70}")
            print(f"TEST: {description}")
            print(f"PROMPT: {prompt_text}")
            print(f"RESPONSE:")
            messages = [{"role": "user", "content": prompt_text}]
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded.to(infer_device)
                attn_mask = None
            elif isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"].to(infer_device)
                attn_mask = encoded.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(infer_device)
            else:
                input_ids = torch.tensor(encoded, device=infer_device).unsqueeze(0)
                attn_mask = None

            streamer = TextStreamer(tokenizer, skip_prompt=True)
            with torch.no_grad():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    streamer=streamer,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=list(stop_ids),
                    stopping_criteria=stopping_criteria,
                    temperature=0.85,   # slightly higher temp for creative output
                    top_p=0.92,
                    do_sample=True,
                )

        print(f"\n{'─' * 70}")
        print("Humour blade inference tests complete.")

    return training_skipped


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print(f"Swiss-Knife — HUMOUR BLADE — DPO Training")
    print(f"{'=' * 80}")
    print(f"  Objective  : {OBJECTIVE}")
    print(f"  Mode       : {args.mode} ({num_epochs} epochs)")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Model      : {args.model_path}")
    print(f"  Log file   : {log_filename}")
    print(f"{'=' * 80}\n")

    try:
        skipped = train_humour_blade()
        print(f"\n[{OBJECTIVE.upper()} BLADE] Done!")
        if not skipped:
            print(f"  Adapter  : {OUTPUT_DIR}/final_adapter/")
        print(f"  Log      : {log_filename}")
        print(f"  TB logs  : {OUTPUT_DIR}/tb_logs/")
    except Exception as exc:
        logger.error(f"Error during training: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
