"""
Preprocess a DPO dataset for the *honesty* objective from the UltraFeedback
honesty axis (NOT the truthfulness axis used by `informativeness`).

Difference vs. informativeness
------------------------------
  informativeness : completions[i]["annotations"]["truthfulness"]["Rating"]
  honesty         : completions[i]["annotations"]["honesty"]["Rating"]

Honesty (UltraFeedback definition) rewards appropriate *calibration* — a
response that does not overclaim and signals uncertainty when warranted —
rather than raw factual correctness. Truthfulness ~ "is it factually right";
honesty ~ "does it express appropriate confidence".

Pair construction
-----------------
  For each prompt with >=2 rated completions:
    chosen   = completion with the HIGHEST honesty rating
    rejected = completion with the LOWEST  honesty rating
  Then filter:
    * honesty gap >= --min_honesty_gap        (default 2; 1 is label noise)
    * chosen != rejected                       (drop degenerate pairs)
    * (optional) --max_confound_gap            keep helpfulness / instruction-
                                               following gaps small, so the
                                               signal is honesty and not just
                                               "the better answer overall"
    * (optional) --max_len_ratio               drop pairs where one response is
                                               N x longer than the other, so DPO
                                               can't exploit length as a proxy
    * (optional) --drop_truthfulqa             drop rows sourced from TruthfulQA
                                               (do this if you plan to EVALUATE
                                               on TruthfulQA — avoids leakage)

Output (JSONL, one record per line; matches dpo_train_full.py's separate-file
branch -> {objective}_train.jsonl + {objective}_eval.jsonl):
  dpo_datasets/honesty_train.jsonl
  dpo_datasets/honesty_eval.jsonl
  {"prompt": "<chat-templated>", "chosen": "...", "rejected": "...", "source": "ultrafeedback"}

Run:
  python preprocess_honesty.py
  python preprocess_honesty.py --min_honesty_gap 2 --max_confound_gap 1 --drop_truthfulqa

Requires:
  pip install datasets transformers huggingface_hub matplotlib numpy

NOTE: after running, add "honesty" to the --objective choices in
dpo_train_full.py so the trainer can pick up honesty_train/eval.jsonl.
"""

import os
import sys
import json
import hashlib
import random
import logging
import argparse
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Build the honesty-objective DPO dataset from the UltraFeedback honesty axis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset", type=str, default="openbmb/UltraFeedback",
                    help="HF dataset with per-axis annotations")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--min_honesty_gap", type=int, default=2,
                    help="Min (max_honesty - min_honesty) per prompt to keep a pair")
parser.add_argument("--max_confound_gap", type=int, default=None,
                    help="If set, drop pairs where helpfulness OR instruction_following "
                         "differ by more than this between chosen and rejected "
                         "(isolates honesty). Try 1. Default off (preserves yield).")
parser.add_argument("--max_len_ratio", type=float, default=4.0,
                    help="Drop pairs where the longer response is > this x the shorter. "
                         "Set <=0 to disable.")
parser.add_argument("--drop_truthfulqa", action="store_true",
                    help="Drop rows whose UltraFeedback source is truthful_qa "
                         "(recommended if you evaluate on TruthfulQA)")
parser.add_argument("--max_samples", type=int, default=None,
                    help="Cap on total kept pairs (before split). Default: keep all")
parser.add_argument("--eval_ratio", type=float, default=0.05,
                    help="Fraction of kept pairs held out for eval")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="./dpo_datasets")
parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--no_plots", action="store_true", help="Skip diagnostic plots")


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers (import-safe: no dataset / tokenizer load at module top level)
# ──────────────────────────────────────────────────────────────────────────────
def _rating(completion, axis):
    """Extract a 1..5 int rating for `axis` from a UltraFeedback completion,
    or None if missing / 'N/A' / unparseable."""
    try:
        ann = completion.get("annotations") or {}
        node = ann.get(axis) or {}
        rating = node.get("Rating")
        if rating is None:
            return None
        rating = str(rating).strip()
        if rating in {"N/A", ""}:
            return None
        return int(rating)
    except (ValueError, AttributeError, TypeError):
        return None


def select_pair(ex, cfg):
    """Turn one UltraFeedback row into a (chosen, rejected) selection on the
    honesty axis, applying all filters.

    Returns (hi_resp, lo_resp, gap, reason) where reason is one of:
      "kept", "no_honesty", "small_gap", "degenerate", "confounded", "length_bias"
    hi_resp / lo_resp are None unless reason == "kept". gap may be None.
    """
    completions = ex.get("completions") or []

    scored = []
    for c in completions:
        h = _rating(c, "honesty")
        resp = c.get("response")
        if h is None or not resp or not resp.strip():
            continue
        scored.append({
            "honesty": h,
            "help":  _rating(c, "helpfulness"),
            "instr": _rating(c, "instruction_following"),
            "resp":  resp.strip(),
        })

    if len(scored) < 2:
        return None, None, None, "no_honesty"

    scored.sort(key=lambda x: x["honesty"])          # ascending
    lo, hi = scored[0], scored[-1]
    gap = hi["honesty"] - lo["honesty"]

    if gap < cfg["min_honesty_gap"]:
        return None, None, gap, "small_gap"
    if hi["resp"] == lo["resp"]:
        return None, None, gap, "degenerate"

    # Confound control: keep the OTHER axes' gaps small so we train honesty,
    # not general quality. Missing ratings can't be checked -> treated as pass.
    if cfg["max_confound_gap"] is not None:
        for k in ("help", "instr"):
            if hi[k] is not None and lo[k] is not None:
                if abs(hi[k] - lo[k]) > cfg["max_confound_gap"]:
                    return None, None, gap, "confounded"

    # Length-bias guard.
    if cfg["max_len_ratio"] and cfg["max_len_ratio"] > 0:
        lc, lr = len(hi["resp"]), len(lo["resp"])
        if max(lc, lr) / max(1, min(lc, lr)) > cfg["max_len_ratio"]:
            return None, None, gap, "length_bias"

    return hi["resp"], lo["resp"], gap, "kept"


def prompt_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, "preprocess_honesty.log"), mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    cfg = {
        "min_honesty_gap": args.min_honesty_gap,
        "max_confound_gap": args.max_confound_gap,
        "max_len_ratio": args.max_len_ratio,
    }

    # Tokenizer (chat template only) ------------------------------------------
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def chat_templated(prompt_str):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_str}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Load dataset ------------------------------------------------------------
    from datasets import load_dataset
    logger.info("=" * 70)
    logger.info(f"SOURCE — UltraFeedback honesty axis ({args.dataset} [{args.split}])")
    logger.info("=" * 70)
    uf = load_dataset(args.dataset, split=args.split)
    logger.info(f"Raw UltraFeedback size: {len(uf):,}")
    logger.info(f"Columns: {uf.column_names}")

    # Build pairs -------------------------------------------------------------
    reasons = Counter()
    gap_hist = Counter()
    kept = []            # list of record dicts (with meta for plotting)
    seen_prompts = set()

    for ex in uf:
        if args.drop_truthfulqa and (ex.get("source") or "").lower() == "truthful_qa":
            reasons["dropped_truthfulqa"] += 1
            continue

        hi_resp, lo_resp, gap, reason = select_pair(ex, cfg)
        if gap is not None:
            gap_hist[gap] += 1
        reasons[reason] += 1

        if reason != "kept":
            continue

        prompt = chat_templated(ex["instruction"])
        h = prompt_hash(prompt)
        if h in seen_prompts:                # dedup by prompt
            reasons["dropped_dup_prompt"] += 1
            continue
        seen_prompts.add(h)

        kept.append({
            "prompt": prompt,
            "chosen": hi_resp,
            "rejected": lo_resp,
            "source": "ultrafeedback",
            "_gap": gap,                     # meta, stripped before writing
        })

    logger.info("\nHonesty-axis yield:")
    for k in ["kept", "no_honesty", "small_gap", "degenerate", "confounded",
              "length_bias", "dropped_dup_prompt", "dropped_truthfulqa"]:
        if reasons.get(k):
            logger.info(f"  {k:22s} {reasons[k]:>8,d}")

    if not kept:
        logger.error("No pairs kept — loosen --min_honesty_gap / --max_confound_gap and retry.")
        sys.exit(1)

    # Optional cap ------------------------------------------------------------
    random.shuffle(kept)
    if args.max_samples is not None and len(kept) > args.max_samples:
        kept = kept[:args.max_samples]
        logger.info(f"Capped to --max_samples: {len(kept):,}")

    # Train / eval split ------------------------------------------------------
    n_eval = max(1, int(len(kept) * args.eval_ratio))
    eval_pairs = kept[:n_eval]
    train_pairs = kept[n_eval:]
    logger.info(f"\nSplit -> train: {len(train_pairs):,}  |  eval: {len(eval_pairs):,}")

    def _strip(p):
        return {k: v for k, v in p.items() if not k.startswith("_")}

    train_path = os.path.join(args.out_dir, "honesty_train.jsonl")
    eval_path = os.path.join(args.out_dir, "honesty_eval.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(_strip(p), ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for p in eval_pairs:
            f.write(json.dumps(_strip(p), ensure_ascii=False) + "\n")
    logger.info(f"Written -> {train_path}")
    logger.info(f"Written -> {eval_path}")

    # Plots -------------------------------------------------------------------
    if not args.no_plots:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
            })
            HONESTY_C, CHOSEN_C, REJECTED_C = "#6A4C93", "#4C9A2A", "#C44536"

            # (1) honesty-gap histogram
            fig, ax = plt.subplots(figsize=(6.5, 4))
            gaps = sorted(gap_hist.keys())
            ax.bar(gaps, [gap_hist[g] for g in gaps], color=HONESTY_C, edgecolor="white")
            ax.axvline(args.min_honesty_gap - 0.5, ls="--", color=REJECTED_C,
                       label=f"cutoff (gap >= {args.min_honesty_gap})")
            ax.set_xlabel("max honesty rating - min honesty rating (per prompt)")
            ax.set_ylabel("prompts")
            ax.set_title("UltraFeedback: honesty-axis rating gap")
            ax.set_xticks(gaps)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "honesty_01_gap.png"), bbox_inches="tight")
            plt.close(fig)

            # (2) chosen vs rejected length
            fig, ax = plt.subplots(figsize=(8, 4))
            cl = [len(p["chosen"]) for p in kept]
            rl = [len(p["rejected"]) for p in kept]
            clip = int(np.percentile(np.array(cl + rl), 99)) if (cl + rl) else 1
            ax.hist(np.clip(cl, 0, clip), bins=40, alpha=0.6, color=CHOSEN_C,
                    label="chosen", edgecolor="white")
            ax.hist(np.clip(rl, 0, clip), bins=40, alpha=0.6, color=REJECTED_C,
                    label="rejected", edgecolor="white")
            ax.set_xlabel("response length (chars, clipped at p99)")
            ax.set_ylabel("count")
            ax.set_title("Honesty blade — chosen vs rejected length")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "honesty_02_lengths.png"), bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plots -> {plots_dir}/")
        except Exception as e:
            logger.warning(f"Plotting skipped ({e})")

    # Summary -----------------------------------------------------------------
    summary = {
        "objective": "honesty",
        "axis": "honesty",
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "filters": {
            "min_honesty_gap": args.min_honesty_gap,
            "max_confound_gap": args.max_confound_gap,
            "max_len_ratio": args.max_len_ratio,
            "drop_truthfulqa": args.drop_truthfulqa,
        },
        "raw_size": len(uf),
        "reasons": dict(reasons),
        "honesty_gap_histogram": dict(sorted(gap_hist.items())),
        "kept_total": len(kept),
        "train": len(train_pairs),
        "eval": len(eval_pairs),
        "paths": {"train": train_path, "eval": eval_path},
    }
    with open(os.path.join(args.out_dir, "honesty_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Preview -----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE PAIR (train[0]):")
    ex0 = train_pairs[0]
    logger.info(f"  Prompt   : {ex0['prompt'][:120]!r}")
    logger.info(f"  Chosen   : {ex0['chosen'][:120]!r}")
    logger.info(f"  Rejected : {ex0['rejected'][:120]!r}")
    logger.info("=" * 70)
    logger.info("Done — honesty blade dataset ready.")


if __name__ == "__main__":
    main()
