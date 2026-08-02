"""
Hyperparameter Sweep Runner for ARGS
====================================

This script performs an end-to-end incremental sweep over the steering weight 
(`alpha`) for the ARGS baseline strategy. It generates responses for 30 prompts 
across 6 different hyperparameter values to trace a Pareto frontier of 
Alignment vs. Quality.

How it works:
1. **Generation**: It iterates through the specified `alpha` values. For each value, 
   it initializes the `ARGSGenerator` and generates responses for 30 prompts from 
   the HH-RLHF dataset.
2. **Storage**: The responses for each hyperparameter configuration are saved as 
   distinct `.jsonl` files (e.g., `args_alpha_1.0.jsonl`) in the tribunal input folder.
3. **Evaluation**: After all configurations are generated, it spawns the vLLM judge 
   server, evaluates all the generated files, and gracefully shuts down the server.

Usage:
    python benchmarking/sweep_args.py
"""

import sys
import os
import json
import time
import argparse
import logging
import subprocess
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import load_tokenizer, load_base_model, load_blade_model
from benchmarking.strategies.args import ARGSGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sweep_args")


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter Sweep for ARGS")
    p.add_argument("--num-prompts", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--task", type=str, default="harmlessness")
    p.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--skip-evaluation", action="store_true")
    return p.parse_args()


def start_vllm_server():
    """Start the vLLM judge server as a background process and wait for it to be ready."""
    logger.info("Starting vLLM judge server...")
    serve_script = os.path.join(os.path.dirname(__file__), "..", "tribunal", "serve_judge.py")
    
    if not os.path.exists(serve_script):
        logger.error(f"Cannot find serve_judge.py at {serve_script}")
        return None
        
    process = subprocess.Popen(
        [sys.executable, serve_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    ready = False
    for line in iter(process.stdout.readline, ""):
        if "Uvicorn running on" in line or "Application startup complete" in line:
            ready = True
            logger.info("vLLM judge server is ready!")
            break
        elif process.poll() is not None:
            logger.error("vLLM server process died prematurely.")
            break
            
    if not ready:
        process.terminate()
        return None
        
    return process


def run_tribunal_eval(task):
    """Run the tribunal evaluation script."""
    logger.info(f"Running tribunal evaluation for task: {task}")
    subprocess.run(
        [sys.executable, "-m", "tribunal.run_eval", "--task", task],
        cwd=os.path.join(os.path.dirname(__file__), "..", "tribunal"),
        check=True
    )
    logger.info("Tribunal evaluation completed successfully.")


def load_prompts(dataset_name, task, num_prompts):
    """Load prompts from the specified dataset."""
    task_map = {"harmlessness": "harmless-base", "helpfulness": "helpful-base"}
    subset = task_map.get(task, task)
    try:
        dataset = load_dataset(dataset_name, data_dir=subset, split="test")
    except Exception as e:
        logger.warning(f"Failed to load dataset online: {e}. Retrying with local cache...")
        from datasets import DownloadConfig
        dataset = load_dataset(
            dataset_name, 
            data_dir=subset, 
            split="test",
            download_config=DownloadConfig(local_files_only=True)
        )
        
    prompts = []
    for item in dataset.select(range(min(num_prompts, len(dataset)))):
        text = item["chosen"]
        prompt = text.split("Assistant:")[0] + "Assistant:"
        prompts.append(prompt)
    return prompts


def main():
    args = parse_args()
    
    alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    print("=" * 80)
    print("  ARGS Hyperparameter Sweep (Grid Search)")
    print("=" * 80)
    print(f"  Alpha values : {alpha_values}")
    print(f"  # Prompts    : {args.num_prompts}")
    print("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tribunal", "inputs", args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Phase 1: Generation ─────────────────────────────────────────────
    if not args.skip_generation:
        prompts = load_prompts(args.dataset, args.task, args.num_prompts)
        
        cfg = SwissKnifeConfig(
            max_new_tokens=args.max_tokens,
            beta=0.1,  # Keep beta fixed
            device=args.device if torch.cuda.is_available() else "cpu",
            dtype=args.dtype,
        )
        
        logger.info("Loading models...")
        tokenizer = load_tokenizer(cfg)
        base_model = load_base_model(cfg)
        blade_model = load_blade_model(cfg, args.task)
        
        for alpha in alpha_values:
            cfg.alpha = alpha
            generator = ARGSGenerator(cfg, tokenizer, base_model, blade_model)
            
            logger.info(f"\n--- Running ARGS with alpha={alpha} ---")
            
            results = []
            for i, prompt in enumerate(prompts):
                response, stats = generator.generate(prompt, return_stats=True, verbose=False)
                
                if response.startswith(prompt):
                    actual_response = response[len(prompt):].strip()
                else:
                    actual_response = response.strip()
                
                results.append({
                    "id": i,
                    "prompt": prompt,
                    "response": actual_response,
                })
                
                logger.info(f"Processed {i+1}/{len(prompts)} prompts...")
            
            out_file = os.path.join(output_dir, f"args_alpha_{alpha}.jsonl")
            with open(out_file, "w") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            logger.info(f"Saved {len(results)} generations to {out_file}")
            
        del base_model, blade_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Phase 2: Evaluation ─────────────────────────────────────────────
    if not args.skip_evaluation:
        server_process = start_vllm_server()
        if server_process is None:
            sys.exit(1)
            
        try:
            time.sleep(5)
            run_tribunal_eval(args.task)
        finally:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    main()
