"""
Preprocess DPO datasets for the Swiss-Knife Humour auditor blade.

Two complementary academic sources are combined:

  1. Reddit r/Jokes (primary, ~550k jokes)
     HuggingFace: "SocialGrep/one-million-reddit-jokes"
     Signal: upvote score → high-score (chosen) vs. low-score (rejected)
     We require a minimum score gap to ensure clear preference signal.

  2. New Yorker Caption Contest (secondary, ~500 prompts × ~5 captions each)
     HuggingFace: "jmhessel/newyorker_caption_contest"
     Signal: contest ranking → winner caption (chosen) vs. worst entry (rejected)
     Provides witty, structured humour with clean human-preference signal.

Why these two?
─────────────
• Reddit r/Jokes is the de-facto academic standard for computational humor
  research (used in Mihalcea & Strapparava 2005; Hossain et al. 2020;
  Weller & Seppi 2019; Yang et al. 2015) and has the largest coverage of
  joke styles (puns, one-liners, anti-jokes, dark comedy).
• The New Yorker Caption Contest is a uniquely *discriminative* benchmark —
  pairs are rated by thousands of humans, so the chosen/rejected signal is
  far cleaner than upvote-noisy Reddit scores. It is widely used in humor
  AI research (Radev et al. 2015; Hessel et al. 2023 — "Do Androids
  Laugh at Electric Sheep?").

Output (JSONL, one record per line):
  dpo_datasets/humour_train.jsonl
  dpo_datasets/humour_eval.jsonl

Run:
  python preprocess_humour.py [--min_score_gap 200] [--max_samples 30000]

Requires:
  pip install datasets transformers huggingface_hub tqdm matplotlib numpy
"""

import os
import json
import argparse
import hashlib
import random
import logging
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Build humour-blade DPO dataset from Reddit r/Jokes + NYCC",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--min_score_gap", type=int, default=15,
                    help="Minimum upvote gap for Reddit chosen/rejected pairs")
parser.add_argument("--max_samples_reddit", type=int, default=30_000,
                    help="Max Reddit pairs to keep (after gap filter)")
parser.add_argument("--eval_ratio", type=float, default=0.05,
                    help="Fraction of total data to use as eval split")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--out_dir", type=str, default="./dpo_datasets",
                    help="Output directory for JSONL files")
parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF tokenizer name for chat template")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.out_dir, "preprocess_humour.log"), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(args.out_dir, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────
logger.info(f"Loading tokenizer: {args.tokenizer}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

def chat_templated(prompt_str: str) -> str:
    """Wrap a plain prompt string through the Qwen chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_str}],
        tokenize=False,
        add_generation_prompt=True,
    )

def prompt_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Helper: basic quality filters
# ──────────────────────────────────────────────────────────────────────────────
MIN_RESPONSE_CHARS = 15   # reject very short jokes
MAX_RESPONSE_CHARS = 2000 # ~512 tokens max response length


def _is_acceptable(text: str) -> bool:
    """Return True if a joke body passes basic quality checks."""
    if not text or not isinstance(text, str):
        return False
    t = text.strip().strip("\ufeff\u200b\u00a0")  # strip BOM + zero-width + nbsp
    if len(t) < MIN_RESPONSE_CHARS or len(t) > MAX_RESPONSE_CHARS:
        return False
    # Filter deleted/removed Reddit artefacts
    if t.lower() in {"[deleted]", "[removed]", ""}:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Source 1: Reddit r/Jokes
# ──────────────────────────────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("SOURCE 1 — Reddit r/Jokes (SocialGrep/one-million-reddit-jokes)")
logger.info("=" * 70)

reddit_ds = load_dataset(
    "SocialGrep/one-million-reddit-jokes",
    split="train",
)
logger.info(f"Raw Reddit size: {len(reddit_ds):,}")
logger.info(f"Columns: {reddit_ds.column_names}")

# ── Build full valid-joke pool ─────────────────────────────────────────────
# Old strategy: required the same title posted 2+ times → only ~16k titles
# eligible → 8k pairs at gap=25.
#
# New strategy (Option B): score-based cross-topic pairing.
#   • Collect ALL jokes with positive scores and valid bodies (~450k rows).
#   • Split into HIGH-score pool (chosen) and LOW-score pool (rejected)
#     using percentile thresholds.
#   • Group both pools by topic key (first N meaningful title words).
#   • For each chosen joke, pair with the worst-scoring joke on the same topic.
#   • Prompt: "Tell me a funny joke about [topic]" — generic enough that
#     both chosen and rejected are plausible responses.
# This unlocks the full 622k+ unique titles and easily yields 30k+ pairs.

STOPWORDS = {
    "a", "an", "the", "i", "my", "me", "you", "we", "it", "is", "was",
    "are", "what", "why", "how", "when", "who", "do", "did", "if", "so",
    "just", "got", "get", "gets", "at", "to", "in", "on", "of", "for",
    "with", "has", "have", "had", "not", "be", "by", "but", "this", "that",
    "can", "he", "she", "they", "his", "her", "our", "its", "your",
}
TOPIC_WORDS = 2   # number of keywords that define a "topic bucket"


def topic_key(title: str, n: int = TOPIC_WORDS) -> str:
    """Return a short keyword key representing the joke topic."""
    words = [
        w.lower().strip(".,!?\"'")
        for w in title.split()
        if w.lower().strip(".,!?\"'") not in STOPWORDS
        and len(w.strip(".,!?\"'")) > 2
    ]
    return " ".join(words[:n]) if words else title[:20].lower()


all_valid: list = []
skipped_reddit = Counter()

for ex in tqdm(reddit_ds, desc="Indexing Reddit jokes"):
    title = (ex.get("title") or "").strip()
    body  = (ex.get("selftext") or "").strip()
    score = ex.get("score") or 0

    if not title:
        skipped_reddit["no_title"] += 1
        continue
    if score <= 0:
        skipped_reddit["non_positive_score"] += 1
        continue
    if not _is_acceptable(body) or body.lower() == title.lower():
        skipped_reddit["bad_response"] += 1
        continue

    all_valid.append({
        "title": title,
        "body":  body,
        "score": score,
        "topic": topic_key(title),
    })

logger.info(f"Valid jokes (with body + positive score): {len(all_valid):,}")
logger.info(f"Skipped rows: {dict(skipped_reddit)}")

# ── Score-percentile split ──────────────────────────────────────────────────
scores_arr = np.array([j["score"] for j in all_valid])
CHOSEN_PCT   = 70   # top 30% → chosen pool
REJECTED_PCT = 30   # bottom 30% → rejected pool

chosen_threshold   = float(np.percentile(scores_arr, CHOSEN_PCT))
rejected_threshold = float(np.percentile(scores_arr, REJECTED_PCT))

chosen_pool   = [j for j in all_valid if j["score"] >= chosen_threshold]
rejected_pool = [j for j in all_valid if j["score"] <= rejected_threshold]

logger.info(f"Chosen pool  (score ≥ {chosen_threshold:.0f}): {len(chosen_pool):,}")
logger.info(f"Rejected pool (score ≤ {rejected_threshold:.0f}): {len(rejected_pool):,}")

# ── Group rejected pool by topic ────────────────────────────────────────────
from collections import defaultdict
rejected_by_topic: dict = defaultdict(list)
for j in rejected_pool:
    rejected_by_topic[j["topic"]].append(j)

# Sort each topic bucket by score ascending so we always pick the worst first
for topic in rejected_by_topic:
    rejected_by_topic[topic].sort(key=lambda x: x["score"])

# ── Build preference pairs ──────────────────────────────────────────────────
# For each chosen joke, find the worst-scoring rejected joke on the same topic.
# Shuffle chosen pool so the cap samples evenly across topics.

random.shuffle(chosen_pool)
reddit_pairs = []
used_rejected_ids: set = set()
gap_hist = Counter()

for chosen in tqdm(chosen_pool, desc="Building Reddit pairs"):
    if len(reddit_pairs) >= args.max_samples_reddit:
        break

    key = chosen["topic"]
    candidates = [
        j for j in rejected_by_topic.get(key, [])
        if id(j) not in used_rejected_ids and j["body"] != chosen["body"]
    ]
    if not candidates:
        continue

    rejected = candidates[0]   # already sorted: worst-scoring first
    gap = chosen["score"] - rejected["score"]
    gap_hist[min(gap // 50 * 50, 1000)] += 1

    if gap < args.min_score_gap:
        continue

    used_rejected_ids.add(id(rejected))

    # Prompt is topic-based — generic so both chosen and rejected are valid responses
    prompt = chat_templated(f"Tell me a funny joke about: {key}")
    reddit_pairs.append({
        "prompt":   prompt,
        "chosen":   chosen["body"],
        "rejected": rejected["body"],
        "source":   "reddit_jokes",
        "meta": {
            "chosen_title":   chosen["title"],
            "rejected_title": rejected["title"],
            "topic_key":      key,
            "hi_score":       chosen["score"],
            "lo_score":       rejected["score"],
            "gap":            gap,
        },
    })

logger.info(f"Reddit pairs built (capped at {args.max_samples_reddit}): {len(reddit_pairs):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Source 2: New Yorker Caption Contest (NYCC)
# ──────────────────────────────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("SOURCE 2 — New Yorker Caption Contest (jmhessel/newyorker_caption_contest)")
logger.info("=" * 70)

try:
    nycc_ds = load_dataset(
        "jmhessel/newyorker_caption_contest",
        "ranking",
        split="train",
    )
    logger.info(f"Raw NYCC size: {len(nycc_ds):,}")
    logger.info(f"Columns: {nycc_ds.column_names}")

    nycc_pairs = []
    nycc_skipped = Counter()

    for ex in tqdm(nycc_ds, desc="Processing NYCC"):
        description = (ex.get("image_description") or "").strip()
        candidates  = ex.get("caption_choices") or []
        raw_label   = ex.get("label")

        if not description or len(candidates) < 2:
            nycc_skipped["insufficient_candidates"] += 1
            continue
        if raw_label is None:
            nycc_skipped["no_label"] += 1
            continue

        # The `label` field in jmhessel/newyorker_caption_contest is a LETTER
        # ("A", "B", "C", "D", "E") corresponding to the winning caption index.
        # Priority order:
        #   1. Letter → index ("A"→0, "B"→1, ...)
        #   2. Integer index (fallback)
        #   3. Normalised text search through candidates
        label_str = raw_label.strip() if isinstance(raw_label, str) else str(raw_label).strip()

        winner_idx = None

        # 1. Letter label (most common format in this dataset)
        if len(label_str) == 1 and label_str.upper() in "ABCDE":
            letter_idx = ord(label_str.upper()) - ord("A")
            if 0 <= letter_idx < len(candidates):
                winner_idx = letter_idx

        # 2. Integer / numeric string
        if winner_idx is None and label_str.isdigit():
            idx = int(label_str)
            if 0 <= idx < len(candidates):
                winner_idx = idx

        # 3. Normalised text search (if label is the caption text itself)
        if winner_idx is None:
            label_norm = label_str.lower()
            for i, c in enumerate(candidates):
                if (c or "").strip().lower() == label_norm:
                    winner_idx = i
                    break

        if winner_idx is None:
            nycc_skipped["bad_label"] += 1
            continue

        chosen_caption = (candidates[winner_idx] or "").strip()
        other_captions = [
            (candidates[i] or "").strip()
            for i in range(len(candidates))
            if i != winner_idx and candidates[i]
        ]
        rejected_caption = other_captions[-1] if other_captions else None

        if not chosen_caption or not rejected_caption:
            nycc_skipped["empty_caption"] += 1
            continue
        if chosen_caption.lower() == rejected_caption.lower():
            nycc_skipped["identical_captions"] += 1
            continue

        prompt = chat_templated(
            f"Write a funny, witty single-line caption for the following scene:\n\n{description}"
        )
        nycc_pairs.append({
            "prompt":   prompt,
            "chosen":   chosen_caption,
            "rejected": rejected_caption,
            "source":   "nycc",
            "meta": {
                "description": description[:100],
                "winner_idx":  winner_idx,
            },
        })

    logger.info(f"NYCC pairs kept: {len(nycc_pairs):,}")
    logger.info(f"NYCC skipped: {dict(nycc_skipped)}")

except Exception as exc:
    logger.warning(f"Failed to load NYCC dataset: {exc}")
    logger.warning("Proceeding with Reddit-only data.")
    nycc_pairs = []

# ──────────────────────────────────────────────────────────────────────────────
# Merge, deduplicate, split
# ──────────────────────────────────────────────────────────────────────────────
all_pairs = reddit_pairs + nycc_pairs
logger.info(f"Total pairs before dedup: {len(all_pairs):,}")

# Deduplicate by prompt hash + filter near-identical chosen/rejected pairs
# (e.g. "Thank you sir" vs "Thank you, sir." — almost zero DPO signal)
def _similarity_ratio(a: str, b: str) -> float:
    """Rough character-overlap ratio — no external deps needed."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    matches = sum(c in longer for c in shorter)
    return matches / len(longer)

seen_prompts: set = set()
deduped_pairs = []
near_dup_dropped = 0
for pair in all_pairs:
    h = prompt_hash(pair["prompt"])
    if h in seen_prompts:
        continue
    # Drop near-duplicate chosen/rejected (similarity > 85%)
    if _similarity_ratio(pair["chosen"], pair["rejected"]) > 0.85:
        near_dup_dropped += 1
        continue
    seen_prompts.add(h)
    deduped_pairs.append(pair)

logger.info(f"Total pairs after dedup:  {len(deduped_pairs):,}  (near-dups dropped: {near_dup_dropped})")

# Strip internal meta before writing (not needed by DPOTrainer)
def _strip_meta(pair: dict) -> dict:
    return {k: v for k, v in pair.items() if k != "meta"}

random.shuffle(deduped_pairs)
n_eval  = max(1, int(len(deduped_pairs) * args.eval_ratio))
n_train = len(deduped_pairs) - n_eval

train_pairs = deduped_pairs[:n_train]
eval_pairs  = deduped_pairs[n_train:]

logger.info(f"Train: {len(train_pairs):,}  |  Eval: {len(eval_pairs):,}")

train_path = os.path.join(args.out_dir, "humour_train.jsonl")
eval_path  = os.path.join(args.out_dir, "humour_eval.jsonl")

with open(train_path, "w", encoding="utf-8") as f:
    for pair in train_pairs:
        f.write(json.dumps(_strip_meta(pair), ensure_ascii=False) + "\n")

with open(eval_path, "w", encoding="utf-8") as f:
    for pair in eval_pairs:
        f.write(json.dumps(_strip_meta(pair), ensure_ascii=False) + "\n")

logger.info(f"Written → {train_path}")
logger.info(f"Written → {eval_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})

PALETTE = {
    "reddit": "#E84855",
    "nycc":   "#3A86FF",
    "chosen": "#4CAF50",
    "rejected": "#F44336",
}


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved plot → {path}")


# (1) Source composition
fig, ax = plt.subplots(figsize=(6, 4))
sources  = ["Reddit r/Jokes", "NYCC"]
counts   = [len(reddit_pairs), len(nycc_pairs)]
colors   = [PALETTE["reddit"], PALETTE["nycc"]]
bars = ax.bar(sources, counts, color=colors, edgecolor="white")
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("preference pairs")
ax.set_title("Humour Blade — DPO pairs by source")
_save(fig, "humour_01_source_composition.png")

# (2) Reddit score-gap histogram
if gap_hist:
    fig, ax = plt.subplots(figsize=(8, 4))
    gaps_sorted = sorted(gap_hist.keys())
    vals = [gap_hist[g] for g in gaps_sorted]
    ax.bar([str(g) for g in gaps_sorted], vals, color=PALETTE["reddit"], edgecolor="white")
    ax.axvline(
        x=[str(g) for g in gaps_sorted].index(
            str(min(gaps_sorted, key=lambda x: abs(x - args.min_score_gap)))
        ),
        color="navy", linestyle="--", label=f"cutoff = {args.min_score_gap}",
    )
    ax.set_xlabel("Score gap bucket (upvotes)")
    ax.set_ylabel("Number of titles")
    ax.set_title("Reddit — upvote score gap distribution")
    ax.legend(frameon=False)
    plt.xticks(rotation=45)
    _save(fig, "humour_02_reddit_score_gap.png")

# (3) Response length distributions
chosen_lens   = [len(p["chosen"])   for p in deduped_pairs]
rejected_lens = [len(p["rejected"]) for p in deduped_pairs]
fig, ax = plt.subplots(figsize=(8, 4))
bins = 40
ax.hist(chosen_lens,   bins=bins, alpha=0.6, color=PALETTE["chosen"],   label="chosen",   edgecolor="white")
ax.hist(rejected_lens, bins=bins, alpha=0.6, color=PALETTE["rejected"], label="rejected", edgecolor="white")
ax.set_xlabel("Response length (chars)")
ax.set_ylabel("Count")
ax.set_title("Humour Blade — chosen vs. rejected length distribution")
ax.legend(frameon=False)
_save(fig, "humour_03_response_lengths.png")

# ──────────────────────────────────────────────────────────────────────────────
# Summary JSON
# ──────────────────────────────────────────────────────────────────────────────
summary = {
    "blade":       "humour",
    "tokenizer":   args.tokenizer,
    "seed":        args.seed,
    "sources": {
        "reddit_jokes": {
            "hf_dataset":         "SocialGrep/one-million-reddit-jokes",
            "academic_refs":      ["Mihalcea & Strapparava 2005", "Weller & Seppi 2019",
                                   "Hossain et al. 2020 (SemEval-2020 Task 7)"],
            "signal":             "upvote score — high score = chosen",
            "min_score_gap":      args.min_score_gap,
            "pairs_before_cap":   len(reddit_pairs),   # already capped above
            "pairs_kept":         len(reddit_pairs),
        },
        "nycc": {
            "hf_dataset":         "jmhessel/newyorker_caption_contest",
            "academic_refs":      ["Hessel et al. 2023 — Do Androids Laugh at Electric Sheep?",
                                   "Radev et al. 2015 — Humor in Collective Discourse"],
            "signal":             "contest ranking — winner = chosen, worst = rejected",
            "pairs_kept":         len(nycc_pairs),
        },
    },
    "total_after_dedup": len(deduped_pairs),
    "train":             len(train_pairs),
    "eval":              len(eval_pairs),
    "paths": {
        "train": train_path,
        "eval":  eval_path,
    },
}

summary_path = os.path.join(args.out_dir, "humour_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Summary → {summary_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Quick sample preview
# ──────────────────────────────────────────────────────────────────────────────
logger.info("\n" + "=" * 70)
logger.info("SAMPLE PAIR (train[0]):")
ex = train_pairs[0]
logger.info(f"  Source:   {ex['source']}")
logger.info(f"  Prompt:   {ex['prompt'][:120]!r}")
logger.info(f"  Chosen:   {ex['chosen'][:120]!r}")
logger.info(f"  Rejected: {ex['rejected'][:120]!r}")
logger.info("=" * 70)
logger.info("Done — humour blade dataset ready.")
