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

### 8-GPU Parallel Cluster Execution (Single-Pass for Baselines + CBN Ablation)

#### 1. Launch 8 Parallel GPU Shards
This single command runs all 6 Pareto strategies (`swiss`, `mod`, `rs`, `args`, `bon`, `base`) **plus** the Candidate-Batch Normalization ablation (`swiss_no_cbn`) across 8 GPUs in parallel:

```bash
mkdir -p runs/logs

for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES=$gpu python benchmarking/benchmark_hhh_pareto.py \
    --prompts data/hhh_eval_prompts.jsonl \
    --methods swiss mod rs args bon base swiss_no_cbn \
    --grid symmetric7 \
    --max-tokens 512 \
    --num-shards 8 \
    --shard-id $gpu \
    > runs/logs/pareto_shard_$gpu.log 2>&1 &
done
wait
```

#### 2. Merge GPU Shards
Merge the output JSON shards into final evaluation JSON and judge-ready JSONL files:

```bash
python benchmarking/benchmark_hhh_pareto.py \
  --prompts data/hhh_eval_prompts.jsonl \
  --methods swiss mod rs args bon base swiss_no_cbn \
  --grid symmetric7 \
  --num-shards 8 \
  --merge-shards
```

---

## 3. Tribunal G-Eval LLM Judging (Qwen 2.5 32B Judge)

#### 1. Launch 4-Bit Quantized Qwen 2.5 32B Judge Server
In a dedicated terminal or background process, start the vLLM judge server:

```bash
python tribunal/serve_judge.py --port 8000
```

#### 2. Execute G-Eval Evaluation
Evaluate all generated responses across quality, harmlessness, and honesty rubrics:

```bash
python tribunal/run_tribunal_eval.py \
  --input-dir tribunal/inputs/hhh_pareto \
  --output-dir tribunal/eval_results/hhh_pareto \
  --overwrite
```

---

## 4. Statistical Analysis & Visualization

Run automated Pareto analysis to extract Hypervolume, Schott's Spacing ($S$), Spearman steerability correlations, and logit scale ratio diagnostics:

```bash
python benchmarking/analyze_hhh_pareto.py \
  --summary tribunal/eval_results/hhh_pareto/model_summary.csv \
  --out-dir runs/hhh_pareto/plots
```

---

## Technical Audit Verification Summary

1. **Logit & Score Mixing Formulations**:
   - **Swiss-Knife (`swiss`)**: Candidate-Batch Normalization (CBN, `normalize_scores=True`) standardizes candidate rewards ($\mu = 0, \sigma = 1$) and scale-normalizes uncertainties ($\sigma$) per blade independently prior to convex logit mixing. Elo ratings and UWO scores are combined via $w_{\text{tourn}} Z(R_i) + w_{\text{blade}} (Z(\mu) - \lambda Z(\sigma))$.
   - **CBN Ablation (`swiss_no_cbn`)**: Disables CBN (`normalize_scores=False`), mixing unstandardized raw DPO blade rewards directly.
   - **MOD**: Token-level linear log-probability mixing: $\sum w_i \log p_{\text{blade}_i}(t|x)$.
   - **ARGS**: Token-level reward-guided logit offset: $\log p_{\text{base}}(t) + \alpha \beta \sum w_i (\log p_{\text{blade}_i}(t) - \log p_{\text{base}}(t))$.
   - **RS (Rewarded Soups)**: Linear weight-space averaging of PEFT adapters.
   - **BoN**: Trajectory-level scoring ($N=7$, compute-matched to Swiss-Knife).
   - **Base**: Unsteered SFT baseline.
2. **Response Token Limits**: Hard capped at 512 tokens across all strategies.
3. **Prompt Leakage Prevention**: All model outputs are stripped of prompt preambles and turn markers via `extract_response()` prior to Tribunal evaluation.
4. **Diagnostic Logging**: Every generation step automatically logs `raw_blade_mu_stds`, `raw_blade_mu_means`, `logit_scale_ratio`, `tournament_term_std`, and `blade_term_std` directly into JSON response stats for zero-GPU post-hoc verification.
