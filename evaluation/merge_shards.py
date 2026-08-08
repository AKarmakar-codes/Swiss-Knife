"""
Swiss Knife — Merge Sharded Generation Outputs
===============================================

After running generation across N GPUs with:

    for i in 0..N-1:
        CUDA_VISIBLE_DEVICES=<gpu_i> python evaluation/run_hh_experiments.py \\
            --test all --mode generate --num_samples 50 --max_new_tokens 768 \\
            --num-shards N --shard-id i

...each shard writes its own suffixed files (e.g. elo_real_sigma_shard3.jsonl,
step_sigma_stats_shard3.json). This script merges those back into the
single, un-suffixed files that `--mode analyze` and Tribunal expect.

Usage:
    python evaluation/merge_shards.py --num-shards 8
"""

import os
import json
import glob
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# (glob_pattern_without_shard_tag, merged_output_path, kind)
# kind: "jsonl_by_id" = list of {"id": ..., ...} lines, re-sorted by id
#       "results_json" = {"responses": [...]}, concatenated + sorted by prompt_idx
#       "stats_json"   = {"prompt_stats": [...]}, concatenated (order doesn't matter)
_JSONL_TARGETS = [
    ("tribunal/inputs/sigma_validity/elo_real_sigma", ".jsonl"),
    ("tribunal/inputs/sigma_validity/elo_shuffled_sigma", ".jsonl"),
    ("tribunal/inputs/sigma_validity/elo_zero_sigma", ".jsonl"),
    ("tribunal/inputs/tournament_value/gsi_elo_thurstonian", ".jsonl"),
    ("tribunal/inputs/tournament_value/gsi_elo_bt_w0", ".jsonl"),
    ("tribunal/inputs/tournament_value/gsi_elo_softmax_blade", ".jsonl"),
]

_RESULTS_JSON_TARGETS = [
    ("runs/sigma_validity/elo_real_sigma_results", ".json"),
    ("runs/sigma_validity/elo_shuffled_sigma_results", ".json"),
    ("runs/sigma_validity/elo_zero_sigma_results", ".json"),
    ("runs/tournament_value/gsi_elo_thurstonian_results", ".json"),
    ("runs/tournament_value/gsi_elo_baseline_results", ".json"),
    ("runs/tournament_value/gsi_elo_softmax_blade_results", ".json"),
]

_STATS_JSON_TARGETS = [
    ("runs/sigma_validity/step_sigma_stats", ".json"),
    ("runs/tournament_value/step_tournament_stats", ".json"),
]


def _shard_paths(base: str, ext: str, num_shards: int):
    paths = []
    for i in range(num_shards):
        p = f"{base}_shard{i}{ext}"
        if os.path.exists(p):
            paths.append(p)
    return paths


def merge_jsonl(base: str, ext: str, num_shards: int):
    paths = _shard_paths(base, ext, num_shards)
    if not paths:
        return
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda r: r.get("id", 0))
    out_path = f"{base}{ext}"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info("Merged %d shard file(s) -> %s (%d rows)", len(paths), out_path, len(rows))


def merge_results_json(base: str, ext: str, num_shards: int):
    paths = _shard_paths(base, ext, num_shards)
    if not paths:
        return
    responses = []
    for p in paths:
        with open(p) as f:
            responses.extend(json.load(f).get("responses", []))
    responses.sort(key=lambda r: r.get("prompt_idx", 0))
    out_path = f"{base}{ext}"
    with open(out_path, "w") as f:
        json.dump({"responses": responses}, f, indent=2)
    logger.info("Merged %d shard file(s) -> %s (%d responses)", len(paths), out_path, len(responses))


def merge_stats_json(base: str, ext: str, num_shards: int):
    paths = _shard_paths(base, ext, num_shards)
    if not paths:
        return
    prompt_stats = []
    for p in paths:
        with open(p) as f:
            prompt_stats.extend(json.load(f).get("prompt_stats", []))
    out_path = f"{base}{ext}"
    with open(out_path, "w") as f:
        json.dump({"prompt_stats": prompt_stats}, f, indent=2)
    logger.info("Merged %d shard file(s) -> %s (%d prompt entries)", len(paths), out_path, len(prompt_stats))


def main():
    parser = argparse.ArgumentParser(description="Merge sharded generation outputs")
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    for base, ext in _JSONL_TARGETS:
        merge_jsonl(base, ext, args.num_shards)
    for base, ext in _RESULTS_JSON_TARGETS:
        merge_results_json(base, ext, args.num_shards)
    for base, ext in _STATS_JSON_TARGETS:
        merge_stats_json(base, ext, args.num_shards)

    logger.info("Done. You can now run: python evaluation/run_hh_experiments.py --test all --mode analyze")


if __name__ == "__main__":
    main()
