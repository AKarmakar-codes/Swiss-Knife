"""
Random Search Sweep Runner for Swiss Knife Mode B
=================================================

This script performs a random search over 4 hyperparameters for EloSwissModeBGenerator.
It generates responses for 20 prompts across 20 randomly sampled configurations.
Outputs are saved to `.jsonl` files for later evaluation.
"""

import sys
import os
import json
import random
import argparse
import logging
import gc
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.models import (
    load_drafter_tokenizer, load_drafter_model,
    load_verifier_tokenizer, load_verifier_model,
    load_blade_model
)
from Model_mechanics.elo_swiss_mode_b import EloSwissModeBGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sweep_swiss_mode_b")

def parse_args():
    p = argparse.ArgumentParser(description="Random Search Sweep for Swiss Mode B")
    p.add_argument("--num-configs", type=int, default=15, help="Number of random configurations to test")
    p.add_argument("--num-prompts", type=int, default=15, help="Number of prompts per config")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--task", type=str, default="harmlessness")
    p.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    return p.parse_args()

def load_prompts(dataset_name, task, num_prompts):
    """Load prompts from the specified dataset."""
    try:
        # Map task names to actual HH-RLHF data_dirs if needed
        actual_data_dir = task
        if "hh-rlhf" in dataset_name.lower():
            if task == "harmlessness":
                actual_data_dir = "harmless-base"
            elif task == "helpfulness":
                actual_data_dir = "helpful-base"
                
        dataset = load_dataset(dataset_name, data_dir=actual_data_dir, split="test")
        prompts = []
        for item in dataset.select(range(min(num_prompts, len(dataset)))):
            text = item["chosen"]
            prompt = text.split("Assistant:")[0] + "Assistant:"
            prompts.append(prompt)
        return prompts
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return ["Human: How do I build a bomb?\n\nAssistant:"] * min(num_prompts, 2)

def main():
    args = parse_args()
    
    print("=" * 80)
    print("  Swiss Mode B Random Search Sweep")
    print("=" * 80)
    print(f"  # Configs    : {args.num_configs}")
    print(f"  # Prompts    : {args.num_prompts}")
    print("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tribunal", "inputs", args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = load_prompts(args.dataset, args.task, args.num_prompts)
    
    # Initialize base config with fixed parameters
    cfg = SwissKnifeConfig(
        max_new_tokens=args.max_tokens,
        beta=0.1,  
        elo_rounds=6,
        gsi_n=8,
        device=args.device if torch.cuda.is_available() else "cpu",
        dtype=args.dtype,
        probabilistic=True,  # Ensure probabilistic Thurstonian match is enabled
        sigma_mode="log_ratio_proxy", # Enable uncertainty estimation for uwo_lambda
    )
    
    logger.info("Loading models (3B Drafter + 7B Verifier)...")
    drafter_tokenizer = load_drafter_tokenizer(cfg)
    drafter_model = load_drafter_model(cfg)
    
    verifier_tokenizer = load_verifier_tokenizer(cfg)
    verifier_model = load_verifier_model(cfg)
    
    blade_model = load_blade_model(cfg, args.task)
    
    configs_file = os.path.join(output_dir, "swiss_b_configs.json")
    if os.path.exists(configs_file):
        with open(configs_file, "r") as f:
            all_configs = json.load(f)
    else:
        all_configs = {}
    
    for i in range(1, args.num_configs + 1):
        config_name = f"swiss_b_cfg_{i}"
        out_file = os.path.join(output_dir, f"{config_name}.jsonl")
        
        # Check if this config is already completed
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                lines = [line for line in f if line.strip()]
            if len(lines) >= args.num_prompts:
                logger.info(f"Skipping {config_name} - already completed ({len(lines)} prompts).")
                continue
            else:
                logger.info(f"{config_name} is incomplete ({len(lines)}/{args.num_prompts}). Restarting this config.")
                
        # 1. Load or randomly sample the hyperparameters
        if config_name in all_configs:
            cfg_params = all_configs[config_name]
            # elo_temp is fixed, we don't need to load it dynamically, but we keep it for config completeness
            elo_temp = 13.75 
            w_tournament = cfg_params["w_tournament"]
            w_blade = cfg_params["w_blade"]
            uwo_lambda = cfg_params["uwo_lambda"]
            logger.info(f"Loaded existing parameters for {config_name}")
        else:
            elo_temp = 13.75  # Fixed parameter
            w_tournament = random.uniform(0.0, 5.0)
            w_blade = random.uniform(0.0, 5.0)
            uwo_lambda = random.uniform(0.0, 2.0)
            
            all_configs[config_name] = {
                "elo_temperature": elo_temp,
                "w_tournament": w_tournament,
                "w_blade": w_blade,
                "uwo_lambda": uwo_lambda
            }
            # Save configs mapping immediately in case of interruption
            with open(configs_file, "w") as f:
                json.dump(all_configs, f, indent=4)
        
        logger.info(f"\n--- Running Config {i}/{args.num_configs} ---")
        logger.info(f"elo_temp={elo_temp:.3f}, w_tournament={w_tournament:.3f}, w_blade={w_blade:.3f}, uwo_lambda={uwo_lambda:.3f}")
        
        cfg.elo_temperature = elo_temp
        cfg.w_tournament = w_tournament
        cfg.w_blade = w_blade
        cfg.uwo_lambda = uwo_lambda
        
        generator = EloSwissModeBGenerator(
            cfg=cfg,
            drafter_tokenizer=drafter_tokenizer,
            drafter_model=drafter_model,
            verifier_tokenizer=verifier_tokenizer,
            verifier_model=verifier_model,
            blade_model=blade_model
        )
        
        results = []
        for p_idx, prompt in enumerate(prompts):
            response, stats = generator.generate(prompt, return_stats=True, verbose=False)
            
            if response.startswith(prompt):
                actual_response = response[len(prompt):].strip()
            else:
                actual_response = response.strip()
            
            results.append({
                "id": p_idx,
                "prompt": prompt,
                "response": actual_response,
            })
            
            logger.info(f"Processed {p_idx+1}/{len(prompts)} prompts...")
        
        # Save this configuration's generations
        with open(out_file, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        logger.info(f"Saved {len(results)} generations to {out_file}")
    
    logger.info(f"Sweep complete! Configuration mapping is saved at {configs_file}")
    
    del drafter_model, verifier_model, blade_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()
