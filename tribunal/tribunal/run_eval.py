"""Command-line entrypoint.

    python -m tribunal.run_eval                       # uses defaults in config.py
    python -m tribunal.run_eval --input my_runs/      # override the input path
    python -m tribunal.run_eval --sample-size 200 --no-detoxify
"""

import argparse

from .config import CONFIG
from . import pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Score LLM responses on safety and quality rubrics.")
    p.add_argument("--task", choices=["harmlessness", "helpfulness"], default="harmlessness",
                   help="Evaluation task to run. Adjusts default input/output directories.")
    p.add_argument("--input", help="Path to a .jsonl file or a folder of them.")
    p.add_argument("--output", help="Folder to write results to.")
    p.add_argument("--sample-size", type=int, help="Max records to score per file.")
    p.add_argument("--judge-url", help="vLLM server URL (default http://localhost:8000/v1).")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct", help="Judge model identifier.")
    p.add_argument("--no-detoxify", action="store_true", help="Skip the Detoxify cross-check.")
    p.add_argument("--include-humour", action="store_true", help="Include humour rubrics in evaluation.")
    p.add_argument("--overwrite", action="store_true", help="Force clean re-evaluation, overwriting existing evaluation CSVs.")
    
    honesty_group = p.add_mutually_exclusive_group()
    honesty_group.add_argument("--no-honesty", dest="include_honesty", action="store_false",
                               default=None, help="Skip honesty rubrics (truthfulness, non_deception, epistemic_honesty).")
    honesty_group.add_argument("--include-honesty", dest="include_honesty", action="store_true",
                               default=None, help="Include honesty rubrics in evaluation.")

    p.add_argument("--max-workers", type=int, default=1, help="Number of parallel worker threads sending requests to vLLM.")
    return p.parse_args()


def main():
    args = parse_args()
    task = args.task

    if args.input:
        CONFIG["input_path"] = args.input
    else:
        CONFIG["input_path"] = f"inputs/{task}"

    if args.output:
        CONFIG["output_folder"] = args.output
    else:
        CONFIG["output_folder"] = f"eval_results/{task}"

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
    if args.max_workers is not None:
        CONFIG["max_workers"] = args.max_workers

    pipeline.run(CONFIG)



if __name__ == "__main__":
    main()
