"""
Swiss-Knife — Format Benchmark Outputs into Tribunal JSONL Inputs
===================================================================

Gathers JSON benchmark outputs from honesty, humour, and multi-blade experiment
runs and converts them into standardized `.jsonl` model files inside `tribunal/inputs/all_experiments/`.

Usage:
    python evaluation/prepare_tribunal_inputs.py \
        --input-dirs runs/gsi_honesty_benchmark/qwen7B runs/gsi_humour_benchmark/qwen7B runs/gsi_multiblade_benchmark/qwen7B \
        --output-dir tribunal/inputs/all_experiments

Then run Tribunal judge in a SINGLE PASS:
    python -m tribunal.run_eval --input tribunal/inputs/all_experiments --output tribunal/eval_results/all_experiments
"""

import os
import json
import glob
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PrepareTribunal")


def process_json_file(file_path: str, out_dir: str, prefix: str = ""):
    p = Path(file_path)
    if p.name in ["summary.json", "gsi_humour_benchmark_summary.json"]:
        return

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            logger.warning("Could not read %s: %s", file_path, e)
            return

    responses = data.get("responses", [])
    if not responses:
        return

    strat_name = data.get("strategy") or data.get("profile") or p.stem.replace("_results", "")
    model_label = f"{prefix}_{strat_name}" if prefix else strat_name
    out_file = os.path.join(out_dir, f"{model_label}.jsonl")

    count = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(responses):
            prompt = item.get("prompt", "")
            resp = item.get("generated") or item.get("response") or ""
            if not prompt or not resp:
                continue
            rec = {
                "id": item.get("prompt_idx", idx),
                "prompt": prompt,
                "response": resp,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Formatted %d records for '%s' → %s", count, model_label, out_file)


def main():
    parser = argparse.ArgumentParser(description="Prepare Tribunal JSONL inputs from experiment runs")
    parser.add_argument(
        "--input-dirs", nargs="+",
        default=[
            "runs/gsi_honesty_benchmark/qwen7B",
            "runs/gsi_humour_benchmark/qwen7B",
            "runs/gsi_multiblade_benchmark/qwen7B",
        ],
        help="Directories containing benchmark JSON outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="tribunal/inputs/all_experiments",
        help="Destination directory for Tribunal .jsonl files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for d in args.input_dirs:
        if not os.path.exists(d):
            logger.warning("Directory does not exist, skipping: %s", d)
            continue

        # Determine prefix from directory name
        dirname = Path(d).parts[-2] if len(Path(d).parts) >= 2 else Path(d).name
        if "honesty" in dirname:
            prefix = "honesty"
        elif "humour" in dirname or "humor" in dirname:
            prefix = "humour"
        elif "multiblade" in dirname or "multi_blade" in dirname:
            prefix = "multiblade"
        else:
            prefix = dirname

        json_files = glob.glob(os.path.join(d, "*.json"))
        for jf in sorted(json_files):
            process_json_file(jf, args.output_dir, prefix=prefix)

    logger.info("Ready for Tribunal evaluation! All .jsonl files prepared in: %s", args.output_dir)


if __name__ == "__main__":
    main()
