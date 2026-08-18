# AAAI Benchmarking Pipeline — 8-GPU Cluster Guide

This document contains the step-by-step instructions and exact bash commands to run the **3D HHH (Helpfulness, Harmlessness, Honesty) Pareto Benchmark** across all 6 alignment strategies:

- **`swiss`**: Swiss-Knife Multi-Blade Mode B (CBN + Thurstonian Elo, ours)
- **`mod`**: Multi-Objective Decoding (Shi et al., NeurIPS 2024)
- **`rs`**: Rewarded Soups (Ramé et al., NeurIPS 2023)
- **`args`**: Alignment as Reward-Guided Search (Shi et al., 2023)
- **`bon`**: Best-of-N (compute-matched to Swiss-Knife candidate budget `N=7`)
- **`base`**: Frozen SFT Backbone (unsteered control baseline)

---

## Environment Setup

Ensure the `myenv` conda environment is active before running commands:

```bash
conda activate myenv
```

---

## Step 1: Generate Frozen 3D HHH Prompt Set (CPU-Only)

Builds the balanced dataset of 120 prompts (40 helpfulness, 40 harmlessness, 40 honesty) and saves it to `data/hhh_eval_prompts.jsonl`.

```bash
python benchmarking/build_hhh_dataset.py --per-axis 40 --out data/hhh_eval_prompts.jsonl
```

---

## Step 2: Parallel 8-GPU Generation (Shards 0 to 7)

Launches 8 background processes in parallel across GPUs 0 through 7. Each GPU processes 1/8th of the 120 prompts across the 13 Pareto grid points.

```bash
mkdir -p runs/logs

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python benchmarking/benchmark_hhh_pareto.py \
    --prompts data/hhh_eval_prompts.jsonl \
    --methods swiss mod rs args bon base \
    --grid symmetric7 \
    --num-shards 8 \
    --shard-id $i \
    > runs/logs/pareto_shard_$i.log 2>&1 &
done; wait
```

To monitor shard execution logs:

```bash
tail -f runs/logs/pareto_shard_0.log
```

---

## Step 3: Merge GPU Shards & Prepare Judge Files

Merges responses from all 8 shards, verifies that all 120 prompts are present without duplication, sorts them strictly by prompt ID, and exports judge-ready JSONL files into `tribunal/inputs/hhh_pareto/`.

```bash
python benchmarking/benchmark_hhh_pareto.py \
  --prompts data/hhh_eval_prompts.jsonl \
  --methods swiss mod rs args bon base \
  --grid symmetric7 \
  --num-shards 8 \
  --merge-shards
```

---

## Step 4: Run G-Eval Judging (Tribunal)

Runs parallel LLM-as-a-judge scoring on all generated responses to produce Quality, Safety, and Honesty rubrics.

```bash
python tribunal/run_tribunal_eval.py \
  --input-dir tribunal/inputs/hhh_pareto \
  --output-dir tribunal/eval_results/hhh_pareto \
  --parallel
```

---

## Step 5: Compute Schott Spacing Delta, Bootstrap CIs & 2D Pareto Plots

Analyzes the G-Eval outputs, calculates Harmonic Mean F1, Schott's Spacing Metric (Δ) for frontier uniformity, paired bootstrap 95% confidence intervals against Swiss-Knife, and generates 2D Pareto projection plots.

```bash
python benchmarking/analyze_hhh_pareto.py \
  --summary tribunal/eval_results/hhh_pareto/model_summary.csv \
  --out-dir runs/hhh_pareto/plots
```

---

## Key Output Locations

- **Merged Benchmark Responses**: `runs/hhh_pareto/<method>/w_*.json`
- **Judge Input Prompts**: `tribunal/inputs/hhh_pareto/<method>__w_*.jsonl`
- **Judge Evaluation Output**: `tribunal/eval_results/hhh_pareto/model_summary.csv`
- **Summary Table CSV**: `runs/hhh_pareto/plots/summary_table.csv`
- **Pareto Projection Plot**: `runs/hhh_pareto/plots/pareto_2d_projections.png`
