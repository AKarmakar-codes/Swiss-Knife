# HHH Multi-Objective Alignment Benchmark Pipeline Execution Guide

This guide outlines the verified, end-to-end execution workflow for the **Swiss-Knife HHH Multi-Objective Alignment Benchmarks** for AAAI submission.

---

## 1. Dataset Generation
Generate the fixed, 60-prompt dataset (20 Helpfulness, 20 Harmlessness, 20 Honesty) using random seed `42`:

```bash
python benchmarking/build_hhh_dataset.py --per-axis 20 --seed 42 --out data/hhh_eval_prompts.jsonl
```

---

## 2. Benchmark Response Generation

### Option A: 8-GPU Parallel Cluster Execution (Recommended)

#### 1. Pareto Grid Generation (8 GPUs in Parallel)
```bash
for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES=$gpu python benchmarking/benchmark_hhh_pareto.py \
    --prompts data/hhh_eval_prompts.jsonl \
    --methods swiss mod args bon rs base \
    --grid symmetric7 \
    --max-tokens 512 \
    --num-shards 8 \
    --shard-id $gpu &
done
wait
```

#### 2. Merge Pareto Shards
```bash
python benchmarking/benchmark_hhh_pareto.py --merge-shards --num-shards 8
```

#### 3. Ablation Suite Generation (8 GPUs in Parallel)
```bash
for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES=$gpu python benchmarking/benchmark_hhh_ablations.py \
    --prompts data/hhh_eval_prompts.jsonl \
    --max-tokens 512 \
    --num-shards 8 \
    --shard-id $gpu &
done
wait
```

#### 4. Merge Ablation Shards
```bash
python benchmarking/benchmark_hhh_ablations.py --merge-shards --num-shards 8
```

---

### Option B: Single-GPU Sequential Execution

#### 1. Pareto Grid Generation
```bash
python benchmarking/benchmark_hhh_pareto.py \
  --prompts data/hhh_eval_prompts.jsonl \
  --methods swiss mod args bon rs base \
  --grid symmetric7 \
  --max-tokens 512
```

#### 2. Ablation Suite Generation
```bash
python benchmarking/benchmark_hhh_ablations.py \
  --prompts data/hhh_eval_prompts.jsonl \
  --max-tokens 512
```

---

## 3. Tribunal G-Eval LLM Judging

#### 1. Launch vLLM Judge Server
In a separate terminal or background process, launch the judge server:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --tensor-parallel-size 1
```

#### 2. Evaluate Pareto Grid Outputs
```bash
python tribunal/run_tribunal_eval.py \
  --input tribunal/inputs/hhh_pareto \
  --output tribunal/eval_results/hhh_pareto \
  --parallel \
  --max-workers 8
```

#### 3. Evaluate Ablation Suite Outputs
```bash
python tribunal/run_tribunal_eval.py \
  --input tribunal/inputs/hhh_ablations \
  --output tribunal/eval_results/hhh_ablations \
  --parallel \
  --max-workers 8
```

---

## 4. Statistical Analysis & Reporting

Run statistical metrics extraction (Pareto Hypervolume, 3-Axis Harmonic F1, Schott's Spacing Metric, 95% Bootstrap CIs):

```bash
python benchmarking/analyze_hhh_pareto.py \
  --results-dir tribunal/eval_results/hhh_pareto \
  --out-dir results/hhh_pareto_analysis
```

---

## Technical Audit Verification Summary

1. **Logit & Score Mixing Formulations**:
   - **Swiss-Knife**: Candidate-Batch Normalization (CBN) normalizes rewards ($\mu$) and scale-normalizes uncertainties ($\sigma$) per blade independently before convex mixing. Elo ratings and UWO scores are combined via $w_{\text{tourn}} Z(R_i) + w_{\text{blade}} (Z(\mu) - \lambda Z(\sigma))$.
   - **MOD**: Token-level linear log-probability mixing: $\sum w_i \log p_{\text{blade}_i}(t|x)$.
   - **ARGS**: Token-level reward-guided logit offset: $\log p_{\text{base}}(t) + \alpha \beta \sum w_i (\log p_{\text{blade}_i}(t) - \log p_{\text{base}}(t))$.
   - **RS (Rewarded Soups)**: Linear weight-space averaging of PEFT adapters.
   - **BoN**: Trajectory-level scoring ($N=7$, compute-matched to Swiss-Knife).
2. **Response Token Limits**: Hard capped at 512 tokens across all strategies.
3. **Prompt Leakage Prevention**: All model outputs are stripped of prompt preambles via `extract_response()` prior to Tribunal evaluation.
