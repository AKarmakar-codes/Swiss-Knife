"""
Benchmarking Runner for Tier 1 Strategies
=========================================

This script orchestrates the end-to-end evaluation of the Tier 1 baseline 
alignment strategies (Best-of-N, ARGS, MOD, DeAL) as defined in 
`benchmarking_plan.md`.

How it works:
1. **Generation Phase**: For each specified strategy, the script initializes the 
   corresponding generator, samples a batch of prompts from the target dataset 
   (e.g., Anthropic HH-RLHF), and generates responses.
2. **Storage**: The generated responses are immediately formatted into `.jsonl` 
   files compliant with the tribunal judge format and saved into the respective 
   inputs directory (e.g., `tribunal/inputs/harmlessness/`).
3. **Evaluation Phase**: Once all responses for all strategies are generated and 
   saved, the script automatically spawns the vLLM judge server (`serve_judge.py`) 
   as a background process. It waits for the server to initialize, then executes 
   the tribunal evaluation pipeline (`run_eval.py`) to compute all required metrics 
   (response quality, relevance, helpfulness, toxicity, refusal, harmfulness).
4. **Cleanup**: After the tribunal evaluation is complete and results are saved 
   to `tribunal/eval_results/`, the script cleanly terminates the background 
   judge server.

This fully automated pipeline requires zero manual intervention once started.

Usage:
    python benchmarking/run_benchmarks.py \
        --strategies bon args mod deal \
        --num-prompts 50 \
        --task harmlessness
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

from benchmarking.strategies.bon import BestOfNGenerator
from benchmarking.strategies.args import ARGSGenerator
from benchmarking.strategies.mod import MODGenerator
from benchmarking.strategies.deal import DeALGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-End Benchmarking Runner for Tier 1 Strategies")
    p.add_argument("--strategies", type=str, nargs="+", default=["bon", "args", "mod", "deal"],
                   choices=["bon", "args", "mod", "deal"])
    p.add_argument("--num-prompts", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--task", type=str, default="harmlessness", choices=["harmlessness", "helpfulness"])
    p.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    
    # Strategy-specific params
    p.add_argument("--gsi-n", type=int, default=8, help="Number of candidates for BoN")
    p.add_argument("--top-k", type=int, default=10, help="Top-k for DeAL")
    p.add_argument("--alpha", type=float, default=0.5, help="Steering weight for ARGS/DeAL")
    p.add_argument("--beta", type=float, default=0.1, help="DPO Beta")
    
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--skip-generation", action="store_true", help="Skip generation and just run evaluation")
    p.add_argument("--skip-evaluation", action="store_true", help="Skip running the judge server and evaluation")
    
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
    
    # Wait for the server to be ready
    ready = False
    logger.info("Waiting for vLLM server to initialize (this may take a few minutes)...")
    for line in iter(process.stdout.readline, ""):
        # Check for typical uvicorn/vLLM ready messages
        if "Uvicorn running on" in line or "Application startup complete" in line:
            ready = True
            logger.info("vLLM judge server is ready!")
            break
        elif process.poll() is not None:
            logger.error("vLLM server process died prematurely.")
            break
            
    if not ready:
        logger.error("Failed to start vLLM judge server.")
        process.terminate()
        return None
        
    return process


def run_tribunal_eval(task):
    """Run the tribunal evaluation script."""
    logger.info(f"Running tribunal evaluation for task: {task}")
    
    eval_script = os.path.join(os.path.dirname(__file__), "..", "tribunal", "run_eval.py")
    
    if not os.path.exists(eval_script):
        logger.error(f"Cannot find run_eval.py at {eval_script}")
        return
        
    subprocess.run(
        [sys.executable, "-m", "tribunal.run_eval", "--task", task],
        cwd=os.path.join(os.path.dirname(__file__), "..", "tribunal"),
        check=True
    )
    logger.info("Tribunal evaluation completed successfully.")


def load_prompts(dataset_name, task, num_prompts):
    """Load prompts from the specified dataset."""
    logger.info(f"Loading {num_prompts} prompts from {dataset_name} ({task})...")
    
    if "hh-rlhf" in dataset_name:
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
    else:
        raise NotImplementedError(f"Dataset {dataset_name} loading not implemented.")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("  Tier 1 Baseline Strategies Benchmarking")
    print("=" * 80)
    print(f"  Strategies : {args.strategies}")
    print(f"  Task       : {args.task}")
    print(f"  # Prompts  : {args.num_prompts}")
    print("=" * 80)
    
    # ── Phase 1: Generation ─────────────────────────────────────────────
    if not args.skip_generation:
        # Build Config
        cfg = SwissKnifeConfig(
            max_new_tokens=args.max_tokens,
            gsi_n=args.gsi_n,
            top_k=args.top_k,
            alpha=args.alpha,
            beta=args.beta,
            device=args.device if torch.cuda.is_available() else "cpu",
            dtype=args.dtype,
        )
        
        # Output directory
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "tribunal", "inputs", args.task
        )
        os.makedirs(output_dir, exist_ok=True)
        
        prompts = load_prompts(args.dataset, args.task, args.num_prompts)
        
        # Load Models lazily
        tokenizer = None
        base_model = None
        blade_model = None
        helpfulness_model = None
        harmlessness_model = None
        
        def _load_models():
            nonlocal tokenizer, base_model, blade_model
            if tokenizer is None:
                logger.info("Loading tokenizer and models...")
                tokenizer = load_tokenizer(cfg)
                base_model = load_base_model(cfg)
                blade_model = load_blade_model(cfg, args.task)
        
        for strategy in args.strategies:
            _load_models()
            
            generator = None
            if strategy == "bon":
                generator = BestOfNGenerator(cfg, tokenizer, base_model, blade_model)
            elif strategy == "args":
                generator = ARGSGenerator(cfg, tokenizer, base_model, blade_model)
            elif strategy == "deal":
                generator = DeALGenerator(cfg, tokenizer, base_model, blade_model)
            elif strategy == "mod":
                # For MOD, load both blades if not loaded
                nonlocal helpfulness_model, harmlessness_model
                if helpfulness_model is None:
                    helpfulness_model = load_blade_model(cfg, "helpfulness")
                if harmlessness_model is None:
                    harmlessness_model = load_blade_model(cfg, "harmlessness")
                    
                generator = MODGenerator(
                    cfg, tokenizer, 
                    models=[helpfulness_model, harmlessness_model],
                    weights=[0.5, 0.5]
                )
            
            logger.info(f"\n--- Running generation for {strategy} ---")
            
            results = []
            for i, prompt in enumerate(prompts):
                logger.info(f"Processing prompt {i+1}/{len(prompts)}...")
                response, stats = generator.generate(prompt, return_stats=True, verbose=True)
                
                # Extract actual assistant response (remove prompt if present)
                if response.startswith(prompt):
                    actual_response = response[len(prompt):].strip()
                else:
                    actual_response = response.strip()
                
                results.append({
                    "id": i,
                    "prompt": prompt,
                    "response": actual_response,
                })
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save to jsonl
            out_file = os.path.join(output_dir, f"{strategy}.jsonl")
            with open(out_file, "w") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            logger.info(f"Saved {len(results)} generations to {out_file}")
            
        # Free memory before starting the judge
        del base_model
        del blade_model
        del helpfulness_model
        del harmlessness_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Phase 2: Evaluation ─────────────────────────────────────────────
    if not args.skip_evaluation:
        print("\n" + "=" * 80)
        print("  Starting Tribunal Judge Evaluation")
        print("=" * 80)
        
        server_process = start_vllm_server()
        if server_process is None:
            logger.error("Skipping evaluation due to server startup failure.")
            sys.exit(1)
            
        try:
            # Let it warm up for a few seconds just in case
            time.sleep(5)
            
            # Run the client evaluation script
            run_tribunal_eval(args.task)
            
        finally:
            logger.info("Terminating vLLM judge server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
            logger.info("Judge server terminated.")


if __name__ == "__main__":
    main()
