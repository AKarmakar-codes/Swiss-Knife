"""
Swiss-Knife — Frozen HHH Evaluation Prompt Set
===============================================

Builds ONE immutable prompt file that every method, every weight-vector, and
every judge in the HHH Pareto experiment reads from.

Why a frozen file instead of per-script sampling:
    The existing scripts each sample their own prompts. `benchmark_HH_harmlessness.py`
    shuffles harmless-base with seed 42; `benchmarking/run_benchmarks.py` takes the
    first N rows unshuffled with a different prompt-extraction rule; and
    `benchmark_multi_blade.py` uses 4 hand-written strings. Rows produced by those
    three are therefore NOT paired, so no per-prompt bootstrap, no paired win-rate,
    and no honest Pareto comparison is possible across them. This script fixes the
    prompt set once, writes a sha256 over it, and every downstream script keys on
    the `id` field. Any run whose manifest hash differs is not comparable, by
    construction.

Composition — one axis of the HHH triangle per source, so steering toward one
objective visibly costs the others ON THE SAME PROMPTS:

    helpfulness   Anthropic/hh-rlhf  helpful-base  (test)   — task-completion pressure
    harmlessness  Anthropic/hh-rlhf  harmless-base (test)   — safety pressure
    honesty       truthfulqa/truthful_qa generation (val)   — misconception pressure

MOD (Shi et al., 2024) evaluates its Helpful Assistant task on Anthropic-HH and its
DPO/safety task on 200 prompts. We keep Anthropic-HH for two of the three axes so
our numbers sit on the same dataset family, and substitute TruthfulQA for their
third axis (humour → honesty), which is the HHH axis this paper actually claims.

Harmless-base prompts additionally carry the six pre-registered safety categories
from `evaluation/classify_prompts.py`, so the safety tail can be reported
separately without any post-hoc labelling.

Run:
    # default: 40 per axis = 120 prompts
    python evaluation/build_hhh_prompt_set.py

    # MOD-matched size (200 prompts, ~67 per axis)
    python evaluation/build_hhh_prompt_set.py --per-axis 67 --out data/hhh_eval_prompts.jsonl

    # verify an existing file still hashes to what a run claimed
    python evaluation/build_hhh_prompt_set.py --verify data/hhh_eval_prompts.jsonl
"""

import os
import sys
import json
import random
import hashlib
import logging
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

from evaluation.classify_prompts import classify_prompt, CATEGORY_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("BuildHHHPrompts")

AXES = ("helpfulness", "harmlessness", "honesty")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_hh_prompt(text: str) -> str:
    """Everything up to and including the last '\\n\\nAssistant:'.

    Identical to extract_prompt() in evaluation/benchmark_HH_harmlessness.py —
    kept byte-for-byte compatible so prompts built here match the generations
    already on disk from the 125-prompt ablation run.
    """
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:"
    return text


def format_truthfulqa_prompt(question: str) -> str:
    """Wrap a TruthfulQA question in the same Human/Assistant frame as HH-RLHF.

    Without this the honesty subset would differ from the other two axes in
    surface form as well as in content, and any axis effect would be confounded
    with prompt formatting.
    """
    return f"\n\nHuman: {question.strip()}\n\nAssistant:"


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (online, then local-cache fallback — same two-stage pattern as the
# existing benchmark scripts, so a cached box works offline)
# ─────────────────────────────────────────────────────────────────────────────

def _load_with_fallback(loader_kwargs: dict):
    try:
        return load_dataset(**loader_kwargs)
    except Exception as e:
        logger.warning("Online load failed (%s). Retrying from local cache...", e)
        from datasets import DownloadConfig
        return load_dataset(**loader_kwargs, download_config=DownloadConfig(local_files_only=True))


def load_hh_axis(data_dir: str, n: int, seed: int) -> list:
    ds = _load_with_fallback(dict(path="Anthropic/hh-rlhf", data_dir=data_dir, split="test"))
    ds = ds.shuffle(seed=seed)
    out, seen = [], set()
    for row in ds:
        prompt = extract_hh_prompt(row["chosen"])
        # De-duplicate: hh-rlhf repeats prompts across preference pairs, and a
        # duplicated prompt would be double-counted in every paired statistic.
        key = prompt.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(prompt)
        if len(out) >= n:
            break
    return out


def load_truthfulqa_axis(n: int, seed: int) -> list:
    ds = _load_with_fallback(dict(path="truthfulqa/truthful_qa", name="generation", split="validation"))
    ds = ds.shuffle(seed=seed)
    out, seen = [], set()
    for row in ds:
        q = (row.get("question") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(format_truthfulqa_prompt(q))
        if len(out) >= n:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def manifest_hash(records: list) -> str:
    """sha256 over (id, prompt) pairs in file order.

    Every downstream script writes this hash into its output. Two runs are
    comparable iff their hashes match.
    """
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r['id']}\x00{r['prompt']}\x1e".encode("utf-8"))
    return h.hexdigest()


def build(per_axis: dict, seed: int) -> list:
    records = []
    next_id = 0

    for axis in AXES:
        n = per_axis[axis]
        if n <= 0:
            continue
        logger.info("Loading %d prompts for axis '%s'...", n, axis)

        if axis == "helpfulness":
            prompts = load_hh_axis("helpful-base", n, seed)
            source = "Anthropic/hh-rlhf:helpful-base:test"
        elif axis == "harmlessness":
            prompts = load_hh_axis("harmless-base", n, seed)
            source = "Anthropic/hh-rlhf:harmless-base:test"
        else:
            prompts = load_truthfulqa_axis(n, seed)
            source = "truthfulqa/truthful_qa:generation:validation"

        if len(prompts) < n:
            logger.warning("Axis '%s': only %d unique prompts available (asked %d).",
                           axis, len(prompts), n)

        for p in prompts:
            # Safety categories are only meaningful on the harmlessness axis;
            # the classifier is keyword-based and would mislabel benign
            # helpfulness/honesty prompts as 'benign_informational' by default.
            category = classify_prompt(p) if axis == "harmlessness" else None
            records.append({
                "id": next_id,
                "axis": axis,
                "source": source,
                "category": category,
                "category_label": CATEGORY_LABELS.get(category) if category else None,
                "prompt": p,
            })
            next_id += 1

    return records


def write_out(records: list, out_path: str, seed: int, per_axis: dict):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = manifest_hash(records)
    meta = {
        "prompt_set_sha256": digest,
        "num_prompts": len(records),
        "seed": seed,
        "requested_per_axis": per_axis,
        "actual_per_axis": dict(Counter(r["axis"] for r in records)),
        "category_counts": dict(Counter(
            r["category"] for r in records if r["category"] is not None
        )),
        "extraction": "last '\\n\\nAssistant:' for hh-rlhf; Human/Assistant frame for TruthfulQA",
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wrote %d prompts → %s", len(records), out_path)
    logger.info("Metadata → %s", meta_path)
    print("\n" + "=" * 72)
    print("  FROZEN HHH PROMPT SET")
    print("=" * 72)
    print(f"  prompts        : {len(records)}")
    print(f"  per axis       : {meta['actual_per_axis']}")
    print(f"  sha256         : {digest}")
    for cat, cnt in sorted(meta["category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<26} {cnt}")
    print("=" * 72)
    print("  Pass --prompt-file to benchmark_hhh_pareto.py. Every output JSON")
    print("  records this sha256; runs with a different hash are not comparable.")
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Public loader used by the other scripts
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_set(path: str) -> tuple:
    """Read a frozen prompt file. Returns (records, sha256)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records, manifest_hash(records)


def main():
    p = argparse.ArgumentParser(description="Build the frozen HHH evaluation prompt set")
    p.add_argument("--per-axis", type=int, default=40,
                   help="Prompts per HHH axis (default 40 → 120 total). Use 67 for MOD's 200.")
    p.add_argument("--n-helpfulness", type=int, default=None, help="Override --per-axis for this axis")
    p.add_argument("--n-harmlessness", type=int, default=None, help="Override --per-axis for this axis")
    p.add_argument("--n-honesty", type=int, default=None, help="Override --per-axis for this axis")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/hhh_eval_prompts.jsonl")
    p.add_argument("--verify", type=str, default=None,
                   help="Print the sha256 of an existing prompt file and exit.")
    args = p.parse_args()

    if args.verify:
        records, digest = load_prompt_set(args.verify)
        print(f"{args.verify}: {len(records)} prompts, sha256={digest}")
        print(f"  per axis: {dict(Counter(r['axis'] for r in records))}")
        return

    per_axis = {
        "helpfulness":  args.n_helpfulness  if args.n_helpfulness  is not None else args.per_axis,
        "harmlessness": args.n_harmlessness if args.n_harmlessness is not None else args.per_axis,
        "honesty":      args.n_honesty      if args.n_honesty      is not None else args.per_axis,
    }
    records = build(per_axis, args.seed)
    write_out(records, args.out, args.seed, per_axis)


if __name__ == "__main__":
    main()
