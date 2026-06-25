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
    p.add_argument("--input", help="Path to a .jsonl file or a folder of them.")
    p.add_argument("--output", help="Folder to write results to.")
    p.add_argument("--sample-size", type=int, help="Max records to score per file.")
    p.add_argument("--judge-url", help="vLLM server URL (default http://localhost:8000/v1).")
    p.add_argument("--no-detoxify", action="store_true", help="Skip the Detoxify cross-check.")
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
    if args.no_detoxify:
        CONFIG["use_detoxify"] = False

    pipeline.run(CONFIG)


if __name__ == "__main__":
    main()
