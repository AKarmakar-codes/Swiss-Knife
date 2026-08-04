"""
Find duplicate prompts within each objective JSONL file.

For each objective:
  - counts how many prompts appear more than once
  - prints the top-K most duplicated prompts with their repeat count
  - saves a full report to pku_dpo_datasets/duplicate_prompts.json
"""

import os
import json
import hashlib
from collections import Counter, defaultdict

DATA_DIR = "./pku_dpo_datasets"
FILES = {
    "helpfulness":  os.path.join(DATA_DIR, "pku_helpfulness.jsonl"),
    "safety":       os.path.join(DATA_DIR, "pku_safety.jsonl"),
    "harmlessness": os.path.join(DATA_DIR, "pku_harmlessness.jsonl"),
}
OUT_FILE = os.path.join(DATA_DIR, "duplicate_prompts.json")
TOP_K = 10
PROMPT_PREVIEW_CHARS = 220


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def extract_user_turn(prompt: str) -> str:
    """Strip Qwen chat template to show just the user message for readability."""
    marker = "<|im_start|>user\n"
    end = "<|im_end|>"
    i = prompt.find(marker)
    if i == -1:
        return prompt
    j = prompt.find(end, i)
    return prompt[i + len(marker): j if j != -1 else None].strip()


def analyse(name, path):
    if not os.path.exists(path):
        print(f"WARN: {path} missing, skipping {name}")
        return None

    counts = Counter()
    first_seen = {}
    for line_no, line in enumerate(open(path, "r", encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = h(norm(rec["prompt"]))
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = rec["prompt"]

    total = sum(counts.values())
    unique = len(counts)
    dup_keys = [k for k, c in counts.items() if c > 1]
    dup_rows = sum(counts[k] for k in dup_keys)

    print(f"\n=== {name} ===")
    print(f"  total rows         : {total:,}")
    print(f"  unique prompts     : {unique:,}")
    print(f"  prompts repeated   : {len(dup_keys):,}")
    print(f"  rows in duplicates : {dup_rows:,}  ({100*dup_rows/total:.1f}% of file)")

    top = sorted(dup_keys, key=lambda k: -counts[k])[:TOP_K]
    print(f"\n  top {TOP_K} repeated prompts:")
    top_records = []
    for k in top:
        preview = extract_user_turn(first_seen[k])[:PROMPT_PREVIEW_CHARS]
        print(f"    x{counts[k]:<4d}  {preview!r}")
        top_records.append({"count": counts[k], "prompt_preview": preview})

    return {
        "total_rows": total,
        "unique_prompts": unique,
        "duplicate_prompt_count": len(dup_keys),
        "rows_in_duplicates": dup_rows,
        "pct_rows_in_duplicates": round(100 * dup_rows / total, 2) if total else 0.0,
        "top_duplicates": top_records,
    }


def main():
    report = {}
    for name, path in FILES.items():
        res = analyse(name, path)
        if res is not None:
            report[name] = res

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
