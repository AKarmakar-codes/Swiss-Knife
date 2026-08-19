"""
run_tribunal_eval.py — CLI Wrapper Entrypoint for Tribunal Evaluation
======================================================================

Forwarding CLI wrapper matching RUN_PIPELINE.md invocation:
    python tribunal/run_tribunal_eval.py --input-dir tribunal/inputs/hhh_pareto --output-dir tribunal/eval_results/hhh_pareto --parallel
"""

import os
import sys
import argparse

# Ensure local repository modules are on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tribunal.config import CONFIG
from tribunal import pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run Tribunal G-Eval Judging Pipeline")
    p.add_argument("--input-dir", "--input", dest="input", help="Path to input .jsonl file or directory of model files.")
    p.add_argument("--output-dir", "--output", dest="output", help="Output folder for evaluation results CSVs.")
    p.add_argument("--sample-size", type=int, help="Maximum records to score per file.")
    p.add_argument("--judge-url", help="vLLM judge server URL (default http://localhost:8000/v1).")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Judge model identifier.")
    p.add_argument("--no-detoxify", action="store_true", help="Skip Detoxify cross-check.")
    p.add_argument("--include-humour", action="store_true", help="Include humour rubrics in evaluation.")
    p.add_argument("--overwrite", action="store_true", help="Force clean re-evaluation, overwriting existing evaluation CSVs.")
    
    honesty_group = p.add_mutually_exclusive_group()
    honesty_group.add_argument("--no-honesty", dest="include_honesty", action="store_false",
                               default=None, help="Skip honesty rubrics.")
    honesty_group.add_argument("--include-honesty", dest="include_honesty", action="store_true",
                               default=None, help="Include honesty rubrics in evaluation.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.input:
        CONFIG["input_path"] = args.input
    if args.output:
        CONFIG["output_folder"] = args.output
    if args.sample_size is not None:
        CONFIG["sample_size"] = args.sample_size
    if args.judge_url:
        CONFIG["vllm_url"] = args.judge_url
    if args.judge_model:
        CONFIG["judge_model"] = args.judge_model
    if args.no_detoxify:
        CONFIG["use_detoxify"] = False
    if args.include_humour:
        CONFIG["include_humour"] = True
    if args.include_honesty is not None:
        CONFIG["include_honesty"] = args.include_honesty
    if args.overwrite:
        CONFIG["overwrite"] = True
    CONFIG["max_workers"] = 1

    pipeline.run(CONFIG)


if __name__ == "__main__":
    main()
