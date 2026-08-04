# Swiss-Knife: Elo Tournament Selection for Decoding-Time Alignment

> AAAI 2027 Submission — Anonymous

Swiss-Knife is a **step-level speculative decoding framework** for alignment at inference time.
The core strategy (**Mode B**) runs a probabilistic Elo tournament over candidate reasoning steps
drawn from a fast Drafter model, selects a champion unconditionally, and avoids any expensive
Verifier resampling loop.

---

## Overview

Standard decoding-time alignment methods (ARGS, DeAL, MOD) operate at the **token level** and
require either a Verifier forward pass per token or a multi-objective logit blend that couples
all objectives simultaneously.

Swiss-Knife instead operates at the **step level**: at each reasoning step, `n` full candidate
continuations are drafted, scored by a lightweight DPO-trained Blade reward model, and ranked
via a **Thurstonian Elo tournament** (Case-V model) that respects score *uncertainty* —
allowing lower-scoring but more confident candidates to beat high-scoring uncertain ones.

### Key properties of Mode B (the proposed strategy)

| Property | Detail |
|---|---|
| **No acceptance gate** | The tournament winner is accepted unconditionally — no Verifier threshold, no rejection sampling |
| **Probabilistic tournament** | Win probability: `P(A>B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B²))` |
| **Uncertainty-weighted selection** | Champion drawn from softmax over `w_tournament·R_elo + w_blade·(μ − λσ)` |
| **Pure Drafter latency** | No secondary Verifier forward passes; cost = Drafter + Blade scoring only |

---

## Repository Structure

```
Swiss-Knife/
│
├── Model_mechanics/            # Core strategy implementation
│   ├── elo_swiss_mode_b.py     ← THE PROPOSED STRATEGY (Mode B, unconditional acceptance)
│   ├── elo_swiss.py            # Parent class: EloSwissGenerator (Mode A baseline)
│   ├── elo_system.py           # elo_bracket(): Thurstonian / Bradley-Terry match mechanics
│   ├── sigma_estimator.py      # estimate_mu_sigma(): uncertainty estimation (mc_dropout, log_ratio_proxy)
│   ├── blade_rack.py           # BladeRack: multi-blade reward model registry
│   ├── blades.py               # DPO Blade reward model wrapper
│   ├── models.py               # Model loading utilities (Drafter, Verifier, Blade)
│   ├── config.py               # SwissKnifeConfig: all CLI flags and hyper-parameters
│   └── __init__.py
│
├── evaluation/                 # Benchmarking and parameter search scripts
│   ├── benchmark_gsi_strategies_harmlessness.py   # HH-RLHF harmlessness benchmark
│   ├── benchmark_gsi_strategies_helpfulness.py    # HH-RLHF helpfulness benchmark
│   ├── benchmark_gsi_strategies_truthfulness.py   # TruthfulQA benchmark
│   ├── benchmark_toxicity_hhrlhf.py               # Toxicity (Detoxify) on HH-RLHF
│   ├── benchmark_truthfulqa.py                    # TruthfulQA standalone
│   ├── experiment_3_ambiguity_comparison.py       # Ablation: high-ambiguity regime analysis
│   ├── parameter_search_optimized.py              # Bayesian hyper-parameter search (6-D)
│   ├── score_toxicity_detoxify_cpu.py             # CPU toxicity scoring utility
│   ├── logprob_utilities.py                       # compute_logprob / compute_logprobs_batched
│   ├── correct_plots/                             # Final Pareto-frontier figures
│   ├── all_observations.csv                       # Raw benchmark observations
│   └── stochastic_benchmark_results.json          # Stochastic strategy results
│
├── benchmarking/               # Comparative strategy benchmarks (fully preserved)
│   ├── run_benchmarks.py       # Master benchmark runner
│   ├── run_rrm_benchmark.py    # RRM benchmark
│   ├── sweep_swiss_mode_b.py   # Hyper-parameter sweep for Mode B
│   ├── sweep_deal.py           # DeAL sweep
│   ├── sweep_mod.py            # MOD sweep
│   ├── sweep_args.py           # ARGS sweep
│   ├── plot_pareto.py          # Pareto frontier plotting
│   └── strategies/             # Strategy implementations for comparative baselines
│       ├── bon.py              # Best-of-N
│       ├── deal.py             # DeAL (Decoding-time Alignment)
│       ├── mod.py              # MOD (Multi-Objective Decoding)
│       └── rrm.py              # Reward Reasoning Model
│
├── dpo training/               # DPO Blade training pipeline
│   ├── dpo_humour.py           # Humor Blade DPO training (Qwen2.5-7B + LoRA)
│   ├── dpo_train.py            # Core DPO training loop
│   ├── dpo_train_full.py       # Full-parameter DPO variant
│   ├── dpo_pku.py              # PKU SafeRLHF Blade training
│   ├── preprocess_humour.py    # Reddit r/Jokes + New Yorker data preprocessing
│   ├── preprocess_pku.py       # PKU dataset preprocessing
│   ├── preprocess_pku_v2.py    # PKU v2 preprocessing
│   ├── preprocess_v3.py        # Unified v3 preprocessing
│   ├── process_hh_rlhf.py      # HH-RLHF preprocessing
│   ├── find_duplicates.py      # Dataset deduplication
│   ├── analyse_overlap.py      # Dataset overlap analysis
│   ├── analyse_overlap_v2.py   # Dataset overlap analysis v2
│   ├── dpo_datasets/           # Preprocessed JSONL train/eval splits
│   └── README.md               # DPO training documentation
│
├── tribunal/                   # LLM-as-judge evaluation system
│   ├── serve_judge.py          # Judge server
│   ├── tribunal/               # Judge logic
│   ├── inputs/                 # Judge input files
│   └── eval_results/           # Judge evaluation outputs
│
├── tests/                      # Unit and integration tests
│   ├── test_elo_swiss.py       # Tests for EloSwissGenerator (Mode A)
│   ├── test_elo_system.py      # Tests for elo_bracket mechanics
│   ├── test_blade_rack.py      # Tests for BladeRack
│   └── verify_probabilistic_elo.py   # Probabilistic Elo correctness verification
│
├── AuthorKit27/                # AAAI 2027 paper submission
│   ├── CameraReady2027.tex     # Camera-ready paper source
│   ├── AnonymousSubmission2027.tex
│   ├── aaai2027.bib            # Bibliography
│   └── Figures/                # Paper figures
│
├── results/                    # Stored benchmark results
├── runs/                       # Benchmark run artifacts
├── docs/                       # Technical documentation
│   ├── SWISS_KNIFE_V2.pdf      # Paper draft
│   └── rrm_explanation.md
├── DeepEval/                   # DeepEval integration
├── mode_b_logit_mixing_explanation.md   # Background: logit mixing vs. tournament selection
└── huggingface.txt             # HuggingFace model references
```

---

## The Core Strategy: `elo_swiss_mode_b.py`

The full decoding pipeline for a single step:

```
1. Draft n candidate steps  ──→  Drafter (fast LM)
2. Score each step          ──→  Blade reward model  →  μ_i (and σ_i if sigma_mode != 'none')
3. Elo tournament           ──→  elo_bracket()  →  Thurstonian P(A>B) = Φ((μ_A−μ_B)/√(σ_A²+σ_B²))
4. Champion selection       ──→  softmax over  w_tournament·R_elo + w_blade·(μ − λσ)
5. Unconditional acceptance ──→  append winner tokens, no threshold, no rejection
```

### Key hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--probabilistic` | `False` | Use Thurstonian CDF (recommended for uncertainty-aware selection) |
| `--sigma-mode` | `none` | Uncertainty estimation: `none` / `log_ratio_proxy` / `mc_dropout` |
| `--w-tournament` | `1.0` | Weight of Elo rating in champion selection |
| `--w-blade` | `1.0` | Weight of blade UWO term in champion selection |
| `--uwo-lambda` | `0.5` | Uncertainty penalty λ: down-weights high-σ candidates at selection |
| `--elo-rounds` | `6` | Number of Elo rating rounds per step |
| `--elo-temperature` | `1.0` | Temperature for softmax over final Elo ratings |
| `--gsi-n` | `4` | Number of candidate steps drafted per decoding step |

---

## Baselines Compared

| Strategy | Description |
|---|---|
| **Swiss-Knife Mode B** (ours) | Thurstonian Elo tournament, unconditional acceptance |
| **Swiss-Knife Mode A** | Mode B + Verifier acceptance gate (threshold-based fallback) |
| **MOD** | Multi-Objective Decoding — token-level logit blending |
| **DeAL** | Decoding-time Alignment for Large Language Models (Confirm if we are comparing this ?) |
| **Best-of-N** | Sample N full responses, pick highest-reward |
| **RRM** | Reward Reasoning Model with CoT judge |

---

## Quickstart

```bash
# Install dependencies
pip install torch transformers peft datasets accelerate

# Run Mode B on a single prompt
python -c "
from Model_mechanics.config import SwissKnifeConfig
from Model_mechanics.elo_swiss_mode_b import EloSwissModeBGenerator

cfg = SwissKnifeConfig(
    drafter_model='Qwen/Qwen2.5-7B-Instruct',
    blade_model='path/to/dpo-blade-adapter',
    gsi_n=4,
    elo_rounds=6,
    probabilistic=True,
    sigma_mode='log_ratio_proxy',
)
gen = EloSwissModeBGenerator(cfg)
print(gen.generate('What is the capital of France?', max_new_tokens=128))
"

# Run harmlessness benchmark
python evaluation/benchmark_HH_harmlessness.py

# Run Bayesian hyperparameter search
python evaluation/parameter_search_optimized.py
```

---

## Ablation Study

`evaluation/experiment_3_ambiguity_comparison.py` isolates the effect of the tournament mechanism
in **high-ambiguity regimes** by comparing:

- `w_tournament=1.0, w_blade=0.0` — tournament-only selection
- Deterministic Elo baselines at temperatures T=28.6 and T=12.5
- Full Mode B with Thurstonian uncertainty weighting

---

## Citation

```bibtex
@inproceedings{swissknife2027,
  title     = {Swiss-Knife: Elo Tournament Selection for Decoding-Time Alignment},
  booktitle = {Proceedings of the 41st AAAI Conference on Artificial Intelligence},
  year      = {2027},
  note      = {Anonymous submission}
}
```
