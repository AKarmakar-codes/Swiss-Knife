# Swiss-Knife

# Swiss Knife: Complete Experimental Plan
### Pragya Lab, BITS Pilani Goa

---

> **Paper Abstract (Swiss Knife)**
>
> Alignment today is still checkpoint-bound: most pipelines implicitly hard-code a single behavioral compromise into a trained model, making it costly to revise, specialize, or hot-fix behavior after deployment. We introduce **Swiss Knife**, a protocol that repurposes speculative decoding into an alignment socket: a fast draft model generates candidates, while a separately trainable auditor model enforces a chosen alignment objective during generation. Swiss Knife's key algorithmic component is the **Tournament Sampling Auditor (TSA)** — either a pairwise knockout bracket or a Swiss-system schedule — to select the winner under an objective-specific score. Crucially, Swiss Knife supports objective-specific auditor "blades" (safety, helpfulness, harmlessness, informativeness, style) that can be tuned, swapped, and updated independently of the backbone.

---

## Table of Contents

1. [Background: What ECLIPTICA/CITA Built](#1-background-what-eclipticacita-built)
2. [Swiss Knife Architecture](#2-swiss-knife-architecture)
3. [Related Papers and Positioning](#3-related-papers-and-positioning)
4. [Experiment Plan](#4-experiment-plan)
   - [Phase 0 — Setup and Baselines](#phase-0--setup-and-baselines)
   - [Phase 1 — TSA Core Mechanism Ablations](#phase-1--tsa-core-mechanism-ablations)
   - [Phase 2 — Blade Training and Objective Coverage](#phase-2--blade-training-and-objective-coverage)
   - [Phase 3 — Switchability Experiments](#phase-3--switchability-experiments-core-contribution)
   - [Phase 4 — Robustness Experiments](#phase-4--robustness-experiments)
   - [Phase 5 — Systems Realism Experiments](#phase-5--systems-realism-experiments)
   - [Phase 6 — Comparison Against ECLIPTICA](#phase-6--comparison-against-ecliptica-ablation-table)
5. [Key Novelty Angles](#5-key-novelty-angles)
6. [Prioritized Start Order](#6-where-to-start-prioritized-order)

---

## 1. Background: What ECLIPTICA/CITA Built

ECLIPTICA is a training-time framework (Level 1) that teaches a **single LLaMA-3.1-8B backbone** to internalize multiple alignment regimes via natural-language instructions at inference time.

### The CITA Algorithm (Contrastive Instruction-Tuned Alignment)

Training uses quadruples `(I, X, Y+, Y-)` where the alignment instruction `I` defines preference relations relative to the same prompt `X`.

```
L_CITA = L_contrastive_preference(I, X, Y+, Y−) + λ · KL(π_θ(·|I,X) || π_ref)
```

- **L_contrastive_preference**: Logistic contrast on instruction-conditioned log-likelihood gaps. Features self-quenching: preference forces diminish once pairs separate.
- **Mandatory KL anchor**: Enforces a Riemannian trust region, ensuring instruction-conditioned policies remain stable during switching.

### ECLIPTICA Benchmarks and Results

| Benchmark | Description | Size |
|---|---|---|
| ECLIPTICA | 300 prompts × 10 instruction types | 3,000 cases |
| TruthfulQA | Epistemic calibration | 1,634 |
| Conditional Safety | Policy boundary testing | 1,000 |
| Length Control | Verbosity control | 1,000 |
| LITMUS | Alignment quality index | 2,800 |

| Method | Instruction-Alignment Efficiency | TruthfulQA Adaptation |
|---|---|---|
| PPO | 20.4% | baseline |
| GRPO | 36.1% | baseline |
| DPO | 56.1% | +0.001 |
| **CITA** | **86.7%** | **+0.054** |

### Core Limitation ECLIPTICA Could NOT Solve

- Backbone **internalizes** all regimes → they interfere with each other
- Cannot hot-swap alignment objective post-deployment without retraining
- New objective = full backbone retrain ($$$ cost)

**Swiss Knife solves this** by fully externalizing alignment into a small, swappable auditor module.

---

### Evolution Overview

```
LEVEL 0: STATIC ALIGNMENT
  (DPO / RLHF / GRPO)
  - One frozen checkpoint per policy
  - No runtime policy control
  - $$$$ per new policy

        ↓  solves: "separate checkpoint per policy"

LEVEL 1: ECLIPTICA + CITA
  (Instruction-Conditioned Switching)
  - One backbone, many policies via instruction I
  - π_θ(·|I,X) trained with CITA loss + KL anchor
  - ✅ Multi-policy per checkpoint
  - ❌ Regimes interfere inside backbone
  - ❌ Cannot hot-update post-deployment

        ↓  solves: "backbone must internalize ALL regimes"

LEVEL 2: SWISS KNIFE
  (Externalized Decode-Time Alignment)
  - Draft model generates K candidates/step
  - TSA tournament selects winner under chosen objective
  - Pluggable blades: Safety | Helpfulness | Harmlessness | Style
  - ✅ Alignment fully externalized
  - ✅ Hot-swap auditors post-deployment
  - ✅ Train small auditors (~100M) independently
  - $ cost per new objective
```

---

## 2. Swiss Knife Architecture

```
Prompt X
    │
    ▼
[Draft Model — fast, frozen]  ──→  K candidates per decoding step
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │  Tournament Sampling Auditor │
                            │           (TSA)              │
                            │                              │
                            │  Pairwise Knockout Bracket   │
                            │      OR                      │
                            │  Swiss-System Schedule       │
                            └─────────────────────────────┘
                                           │
                       ┌───────────────────┼──────────────────┐
                       ▼                   ▼                  ▼
                  [Safety Blade]  [Helpfulness Blade]  [Harmlessness Blade]
                       │                   │                  │
                       └───────────────────┴──────────────────┘
                                           │
                                           ▼
                                   Selected token/span
                                           │
                                           ▼
                                    Final Response Y
```

### Scoring Function at Each Step

```
score(y) = α · log π_draft(y | x) + (1 − α) · auditor_score(y, objective)
```

- `α = 1.0` → vanilla draft decoding (no alignment)
- `α = 0.0` → auditor-only (ignores fluency)
- `α ≈ 0.4–0.6` → balanced (target operating range)

### Auditor Blades

| Blade | Objective | Training Signal | Auditor Size |
|---|---|---|---|
| Safety | Prevent harmful outputs | PKU-SafeRLHF, HH-RLHF | ~100–300M |
| Helpfulness | Maximize task completion | UltraFeedback, MT-Bench | ~100–300M |
| Harmlessness | Reduce indirect harm (bias, stereotypes) | StereoSet, WinoBias | ~100–300M |
| Informativeness | Maximize factual accuracy | TruthfulQA pairs, FactScore | ~100–300M |
| Style | Formality, conciseness, format | Synthetic GPT-4o pairs | ~100–300M |

---

## 3. Related Papers and Positioning

### Direct Competitors at Decode-Time (must beat these)

| Paper | Method | Why Swiss Knife Differs |
|---|---|---|
| **SteerLM** (Dong et al., 2023) | Attribute-conditioned SFT, steer via labels at inference | Still a trained backbone; no modular auditor, no tournament |
| **Arithmetic Control** (Wang et al., 2024) | Directional preference alignment with multi-objective reward vectors | Linear combination of rewards — no tournament selection, no hot-swap |
| **Decoding-Time Multi-Obj** (Shi et al., 2024) | Runtime objective weighting without retraining | Weighted sum decoding, not tournament-based; no modular blades |
| **Inference-Time Value Guidance** (Liu et al., 2024) | Value function guides tokens at inference | Continuous value signal — no discrete tournament elimination |
| **PAD** (Chen et al., 2025) | Personalized alignment at decode time | Per-user preference, not objective-type switching; no speculative decoding |
| **CoSA** (Zhang et al., 2025) | Controllable safety at inference-time | Safety-only; no general multi-blade framework |
| **ECLIPTICA/CITA** (own Level 1) | Instruction-conditioned policy trained into backbone | Cannot hot-swap; regimes interfere inside backbone |

### Speculative Decoding Background

Swiss Knife reframes the **verifier** step in speculative decoding. Instead of the verifier checking distribution match (Leviathan et al., 2023), the TSA auditor enforces a chosen alignment objective. This is the core algorithmic novelty.

### Key Bibliography (119 refs in ECLIPTICA — most relevant for Swiss Knife)

**Decode-time / inference-time alignment:**
- dong-etal-2023-steerlm — SteerLM
- wang-etal-2024-arithmetic — Arithmetic Control
- shi2024decodingtime — Decoding-Time Multi-Objective
- liu-etal-2024-inference — Inference-Time Value Guidance
- chen2025pad — PAD: Personalized Alignment at Decoding-Time
- zhang2025cosa — Controllable Safety Alignment (CoSA)

**Safety / red-teaming benchmarks:**
- mazeika2024harmbench — HarmBench: standardized red-teaming eval
- ji2024pku — PKU-SafeRLHF
- bai2022hh — HH-RLHF (Helpful and Harmless)
- perez2022red / ganguli2022red — Red-teaming LLMs

**Alignment evaluation:**
- borah-etal-2025-alignment — Alignment Quality Index (AQI)
- lin2022truthfulqa — TruthfulQA
- zheng2023judging — MT-Bench / LLM-as-judge

**Preference optimization (prior work to position against):**
- rafailov2023direct — DPO
- ouyang2022training — RLHF/PPO
- meng2024simpo — SimPO
- hong2024orpo — ORPO

---

## 4. Experiment Plan

---

### Phase 0 — Setup and Baselines

**Goal**: Establish infrastructure and reference numbers before any novel experiments.

#### Model Selection

| Role | Model | Rationale |
|---|---|---|
| Draft model | Llama-3.2-1B (or GPT-2-XL) | Fast, small, well-studied |
| Generator/Backbone | Llama-3.1-8B | Same as ECLIPTICA — enables direct comparison |
| Auditor heads | ~100–300M param fine-tuned models | One per blade; cheap to train |

#### Baseline Systems to Implement

1. **Vanilla autoregressive** — no auditor, greedy/nucleus sampling
2. **Best-of-N sampling** — generate N outputs, pick best by reward model (naive decode-time baseline)
3. **SteerLM inference** — attribute-conditioned, same prompts
4. **ECLIPTICA/CITA** — Level 1 baseline from your own prior work

#### Dataset Reuse from ECLIPTICA

- ECLIPTICA benchmark: 300 prompts × 10 instruction types (3,000 cases)
- TruthfulQA for informativeness blade evaluation
- HarmBench for safety blade evaluation
- PKU-SafeRLHF + HH-RLHF for auditor training

---

### Phase 1 — TSA Core Mechanism Ablations

**Goal**: Prove the tournament mechanism works; find optimal hyperparameters.

---

#### Experiment 1.1 — Tournament Format Comparison

| Setting | Value |
|---|---|
| Independent variable | Knockout bracket vs. Swiss-system schedule |
| Fixed | K=8 candidates, Safety blade, 200 prompts |
| Metrics | Auditor score of selected token, harmlessness score, latency (ms/token) |

**Hypothesis**: Swiss-system is more stable under large K; knockout is faster but noisier at small K.

**Expected result table:**

| Format | Alignment Score | Latency (ms/tok) | Stability |
|---|---|---|---|
| Knockout | TBD | Lower | Lower |
| Swiss-system | TBD | Higher | Higher |

---

#### Experiment 1.2 — Candidate Set Size K

| Setting | Value |
|---|---|
| Independent variable | K ∈ {2, 4, 8, 16, 32} |
| Fixed | Safety blade, knockout format, 200 prompts |
| Metrics | Auditor alignment score, acceptance rate (%), tokens/sec |

**Hypothesis**: K=8–16 is the sweet spot before diminishing returns overwhelm throughput.

---

#### Experiment 1.3 — Token-Level vs. Span-Level Decoding

| Granularity | Description | Trade-off |
|---|---|---|
| Token-level | Tournament at every single token | Higher alignment precision, more auditor calls |
| Span-level (n=5) | Tournament over 5-token windows | Better coherence, lower resolution |

**Metrics**: BERTScore (coherence), auditor alignment score, auditor calls per output token.

---

#### Experiment 1.4 — Score Combination Weight α

```
score(y) = α · log π_draft(y|x) + (1−α) · auditor_score(y, objective)
```

| α | Behavior |
|---|---|
| 1.0 | Vanilla draft — no auditor effect |
| 0.8 | Slight auditor influence |
| 0.6 | Balanced (recommended starting point) |
| 0.4 | Auditor-heavy |
| 0.0 | Auditor-only — may sacrifice fluency |

**Metrics**: Perplexity (fluency proxy), auditor alignment score, degeneration rate (refuse-always %, boilerplate rate).

---

### Phase 2 — Blade Training and Objective Coverage

**Goal**: Train and validate each auditor blade independently. Each blade is a small (~100–300M) model fine-tuned on objective-specific preference data.

---

#### Experiment 2.1 — Safety Blade

| Item | Detail |
|---|---|
| Training data | PKU-SafeRLHF, HH-RLHF, HarmBench adversarial prompts |
| Architecture | DeBERTa-large fine-tuned as binary safe/unsafe scorer |
| Evaluation | HarmBench harmful compliance rate, refusal correctness |
| Critical test | `benign_refusal_rate` on 500 clearly benign prompts — must be <5% |

**Degeneration check**: The auditor must NOT refuse benign queries. Measure:
```
benign_refusal_rate = refused_benign_count / total_benign_prompts
```

---

#### Experiment 2.2 — Helpfulness Blade

| Item | Detail |
|---|---|
| Training data | UltraFeedback (cui2023ultrafeedback), OpenAssistant |
| Evaluation | MT-Bench helpfulness score, AlpacaEval win-rate vs. base model |
| Key test | Same prompt, swap Safety→Helpfulness blade → measure response length and information density increase |

---

#### Experiment 2.3 — Harmlessness Blade

Distinct from safety: targets *indirect* harms (stereotypes, misinformation, implicit bias).

| Item | Detail |
|---|---|
| Training data | StereoSet, WinoBias, PKU-SafeRLHF harmful preference pairs |
| Evaluation | CrowS-Pairs bias score, ToxiGen toxicity score |
| Distinction | Safety blade blocks explicit harm; Harmlessness blade reduces implicit harm |

---

#### Experiment 2.4 — Informativeness Blade

| Item | Detail |
|---|---|
| Training data | TruthfulQA preference pairs (truthful > hallucinated), FactScore-labeled outputs |
| Evaluation | TruthfulQA MC accuracy, FactScore on open-domain generation |
| Key insight | This blade should increase information density WITHOUT increasing harmfulness — tests decoupling |

---

#### Experiment 2.5 — Style Blade (Novel Contribution)

| Item | Detail |
|---|---|
| Training data | Synthetic GPT-4o generated pairs (formal vs. casual, verbose vs. concise, structured vs. prose) |
| Evaluation | Automated style classifiers, human preference on 100 examples |
| Variants | Formality sub-blade, Conciseness sub-blade, JSON/structured output sub-blade |

---

### Phase 3 — Switchability Experiments (Core Contribution)

**Goal**: Prove that swapping blades produces measurably different behaviors on the **same prompts** — directly extending the ECLIPTICA evaluation protocol to decode-time.

---

#### Experiment 3.1 — Objective Adherence Score (OA)

**Protocol**:
1. Take the 300 fixed prompts from ECLIPTICA benchmark
2. Run each prompt through Swiss Knife with each blade active
3. Score each output using all blade auditors as cross-evaluators

**Metric**:
```
OA(blade_b, prompt_x) = auditor_b(output generated with blade_b active on x)
```

**Expected result**: A 4×4 matrix where diagonal >> off-diagonal.

| | Safety Auditor | Helpfulness Auditor | Harmlessness Auditor | Informativeness Auditor |
|---|---|---|---|---|
| Safety Blade active | **HIGH** | medium | high | low |
| Helpfulness Blade active | low | **HIGH** | medium | high |
| Harmlessness Blade active | high | medium | **HIGH** | medium |
| Informativeness Blade active | medium | high | medium | **HIGH** |

---

#### Experiment 3.2 — Cross-Objective Separation

**Metric**:
```
Cross-Obj Separation = mean pairwise cosine distance(embedding(output_blade_i), embedding(output_blade_j))
                       averaged over all prompt pairs (i ≠ j)
```

**Compare against**:
- ECLIPTICA/CITA (same prompts, different instructions)
- SteerLM (same prompts, different attribute labels)
- Best-of-N with same reward model

**Swiss Knife should show higher separation** since objectives are fully decoupled in the auditor, not merged in backbone weights.

---

#### Experiment 3.3 — Live Blade Switching (Mid-Conversation)

**Setup**: Multi-turn conversation where blade changes between turns.

| Turn | Active Blade | Expected Behavior |
|---|---|---|
| Turn 1 | Safety | Conservative, cautious responses |
| Turn 2 | Helpfulness | Informative, task-complete responses |
| Turn 3 | Safety | Returns to conservative behavior |

**Metrics**:
- Time to switch (should be ~0ms — just a module pointer swap)
- Behavioral coherence: does Turn 3 behave same as Turn 1?
- Cross-turn consistency score

**Compare to ECLIPTICA/CITA**: instruction change requires backbone re-inference; Swiss Knife requires only auditor pointer change.

---

### Phase 4 — Robustness Experiments

**Goal**: Demonstrate that Swiss Knife resists degenerate solutions that plague naive reward-maximization decoding.

---

#### Experiment 4.1 — Refuse-Always Trap

**Motivation**: A poorly-trained safety auditor might learn to always return high scores for refusals, causing the TSA to always select refusal tokens.

**Test**:
- Feed 500 clearly benign prompts (cooking questions, math problems, factual queries)
- Measure `benign_refusal_rate = # refused / 500`

**Comparison**:
| Method | benign_refusal_rate (target) |
|---|---|
| Vanilla | ~1% |
| Greedy reward maximization (no tournament) | TBD — likely high |
| Swiss Knife TSA (knockout) | **Target: <3%** |
| Swiss Knife TSA (Swiss-system) | **Target: <3%** |

**Swiss Knife defense**: Tournament selection eliminates single-dominating solutions; a refuse-always token wins the tournament only if refusal truly scores highest AND draft distribution supports it.

---

#### Experiment 4.2 — Boilerplate / Templating Degeneration

**Motivation**: Best-of-N collapses to reward-maximizing templates ("I cannot assist with...").

**Metrics**:
```
Self-BLEU = BLEU(output_i, {output_j : j ≠ i})   # lower = more diverse = less boilerplate
Distinct-1 = |unique unigrams| / |total unigrams|
Distinct-2 = |unique bigrams| / |total bigrams|
```

Measure over 500 diverse prompts across all blades.

---

#### Experiment 4.3 — Distribution Shift Robustness

**Protocol**:
1. Train auditors on HH-RLHF distribution
2. Evaluate on OOD distributions: PKU-SafeRLHF, WildGuard, AdvBench

**Metric**: Alignment score drop = `score_in_distribution − score_OOD`

**Key claim**: Swiss Knife's modular auditor can be cheaply fine-tuned on new distribution, unlike ECLIPTICA which requires backbone retraining.

---

#### Experiment 4.4 — Adversarial Robustness (Jailbreaks)

**Attacks to run** (from HarmBench):
- GCG (Zou et al., 2023) — gradient-based adversarial suffix
- AutoDAN — automated discrete jailbreaks
- PAIR — prompt injection via iterative refinement

**Metric**: Attack Success Rate (ASR) — lower is better for Safety blade.

**Key insight**: Swiss Knife's auditor evaluates *output tokens* not *input prompts* → different attack surface than RLHF-trained models. Adversarial inputs that bypass the backbone's training are still filtered at decoding time.

---

### Phase 5 — Systems Realism Experiments

**Goal**: Prove Swiss Knife is deployable under real compute constraints.

---

#### Experiment 5.1 — Acceptance Rate

**Definition**: Fraction of draft model tokens that survive the TSA tournament.

```
acceptance_rate = (# draft tokens accepted by TSA) / (# total draft tokens generated)
```

**Target**: >60% acceptance rate (comparable to vanilla speculative decoding's ~70%).

Sweep K ∈ {4, 8, 16} × format ∈ {knockout, Swiss-system}.

---

#### Experiment 5.2 — Auditor Calls Per Output Token

Theoretical complexity:
```
Knockout bracket: O(log K) auditor calls per token
Swiss-system (K rounds): O(K log K) auditor calls per token
```

| K | Knockout calls/token | Swiss-system calls/token |
|---|---|---|
| 4 | 2 | 8 |
| 8 | 3 | 24 |
| 16 | 4 | 64 |

Measure **actual** forward pass count on A100 with batch size 1 and 8.

---

#### Experiment 5.3 — Latency vs. Quality Pareto Frontier

**X-axis**: tokens/second throughput (on A100 80GB)
**Y-axis**: objective-specific alignment quality score

**Systems to plot**:
1. Vanilla autoregressive (anchor point)
2. Best-of-N (K=8) — high quality, very slow
3. Swiss Knife TSA knockout (K=4)
4. Swiss Knife TSA knockout (K=8)
5. Swiss Knife TSA Swiss-system (K=8)
6. ECLIPTICA/CITA (instruction-conditioned, no auditor overhead)

**Expected**: Swiss Knife forms a better Pareto frontier than Best-of-N.

---

#### Experiment 5.4 — Auditor Size Scaling

Sweep auditor parameter count: 50M, 125M, 350M, 1.3B.

**Metrics**:
- Alignment quality score (per blade)
- Auditor inference latency (ms/token)
- Quality/compute ratio

**Expected**: 100–300M hits the sweet spot. Above 350M gives diminishing quality gains with large latency cost.

---

### Phase 6 — Comparison Against ECLIPTICA (Ablation Table)

**Goal**: The definitive paper table directly extending the evolution narrative.

| Metric | DPO (Level 0) | CITA/ECLIPTICA (Level 1) | Swiss Knife (Level 2) |
|---|---|---|---|
| Instruction-alignment efficiency | 56.1% | 86.7% | **Target: >90%** |
| TruthfulQA adaptation | +0.001 | +0.054 | **Target: +0.08+** |
| ECLIPTICA benchmark OA score | — | 86.7% | **Target: >88%** |
| Post-deployment hot-swap | ❌ | ❌ | ✅ |
| Regime interference | N/A | Partial (KL mitigates) | ✅ None (decoupled) |
| Refuse-always rate (benign prompts) | ~2% | ~5% | **Target: <3%** |
| Self-BLEU (diversity) | high | medium | **Target: low** |
| Update cost (new objective) | Full retrain ($$$) | Backbone retrain ($$) | **Small auditor ($)** |
| Throughput vs. vanilla | 1.0× | 1.0× | **Target: >0.7×** |
| Cross-objective separation | low | medium | **Target: high** |

---

## 5. Key Novelty Angles

### Novelty 1 — Speculative Decoding as Alignment Socket
Nobody has reframed the **verifier** in speculative decoding as an alignment enforcer. Standard speculative decoding uses the verifier only to preserve the backbone's distribution. Swiss Knife repurposes this step to *change* the distribution toward a chosen objective. This is a fundamental reframing of what speculative decoding can do.

### Novelty 2 — Tournament Sampling vs. Greedy Reward
Standard decode-time alignment methods use greedy reward maximization: `argmax_y auditor(y)`. This collapses to degenerate solutions (refuse-always, boilerplate) because a single token can dominate. Tournament sampling via brackets:
- Prevents single-token domination through elimination rounds
- Combines draft likelihood signal naturally (losing bracket members have low draft score)
- Formal analysis possible via multi-armed bandit / Condorcet winner theory

### Novelty 3 — Objective-Specific Blade Isolation
ECLIPTICA/CITA relies on KL regularization to *reduce* cross-regime interference, but cannot eliminate it (all objectives share backbone weights). Swiss Knife **architecturally decouples** objectives into separate auditor modules. This means:
- Safety updates cannot interfere with Helpfulness blade
- Style blade can be added without any retraining of existing blades
- Each blade can be versioned, audited, and rolled back independently

### Novelty 4 — Post-Deployment Updatability
The paper makes a practical deployment claim: when a new alignment requirement emerges (e.g., stricter refusal posture for a new market), only the relevant blade needs to be retrained (~100–300M params, cheap) and hot-swapped into production. No backbone downtime, no full retraining. This is qualitatively different from anything in the current literature.

---

## 6. Where to Start (Prioritized Order)

### Step 1 — Pipeline Validation (Week 1–2)
Implement TSA on top of HuggingFace's speculative decoding API.
```python
# Pseudocode sketch
draft_model = LlamaForCausalLM("meta-llama/Llama-3.2-1B")
backbone = LlamaForCausalLM("meta-llama/Llama-3.1-8B")
auditor = SafetyBlade("deberta-large-safety-finetuned")

for step in range(max_steps):
    candidates = draft_model.generate(prompt, num_return=K)         # K draft tokens
    scores = auditor.score(candidates)                               # blade scores
    winner = tournament_knockout(candidates, scores, draft_scores)   # TSA selection
    output.append(winner)
```
**Validates**: The pipeline exists and produces coherent output.

### Step 2 — Safety Blade Training (Week 2–3)
Fine-tune DeBERTa-large on HH-RLHF preference pairs as a binary classifier.
- Easiest blade to train (clearest signal)
- Most benchmarked (HarmBench, PKU-SafeRLHF)
- Run Experiment 4.1 (refuse-always test) immediately after training

### Step 3 — Switchability on ECLIPTICA Benchmark (Week 3–4)
Run Phase 3, Experiments 3.1 and 3.2 using ECLIPTICA's 300-prompt held-out set.
- This gives you a direct comparison with CITA's published 86.7% number
- Uses existing evaluation infrastructure from ECLIPTICA

### Step 4 — Tournament Degeneration Test (Week 4)
Run Experiment 4.1 (refuse-always trap) formally.
- This validates the central mechanical claim of TSA
- Compare tournament vs. greedy reward selection directly

### Step 5 — Pareto Curve (Week 5–6)
Run Phase 5, Experiment 5.3 (latency vs. quality Pareto frontier).
- This is the systems realism centerpiece figure for the paper

### Step 6 — Full Blade Suite + Final Ablation Table (Week 6–8)
Train remaining blades, run all benchmarks, compile Phase 6 comparison table.

---

## Appendix: Relevant Papers Quick Reference

| Citation Key | Paper Title | Relevance |
|---|---|---|
| dong-etal-2023-steerlm | SteerLM: Attribute Conditioned SFT | Direct baseline |
| wang-etal-2024-arithmetic | Arithmetic Control of LLMs | Direct baseline |
| shi2024decodingtime | Decoding-Time Multi-Objective Alignment | Direct baseline |
| liu-etal-2024-inference | Inference-Time Value Guidance | Direct baseline |
| chen2025pad | PAD: Personalized Alignment at Decoding-Time | Direct baseline |
| zhang2025cosa | Controllable Safety Alignment (CoSA) | Direct baseline |
| mazeika2024harmbench | HarmBench | Safety eval benchmark |
| ji2024pku | PKU-SafeRLHF | Safety auditor training data |
| bai2022hh | HH-RLHF | Safety + helpfulness training data |
| cui2023ultrafeedback | UltraFeedback | Helpfulness training data |
| lin2022truthfulqa | TruthfulQA | Informativeness eval |
| borah-etal-2025-alignment | Alignment Quality Index (AQI) | Alignment eval metric |
| rafailov2023direct | DPO | Level 0 baseline |
| ouyang2022training | RLHF/PPO | Level 0 baseline |
| dubey2024llama3 | Llama 3 Herd | Backbone model |
| zou2023universal | GCG Adversarial Attacks | Robustness eval |
| perez2022red | Red Teaming LLMs | Adversarial eval |
| zheng2023judging | MT-Bench / LLM-as-Judge | Helpfulness eval |
| meng2024simpo | SimPO | Preference optimization baseline |
| hong2024orpo | ORPO | Preference optimization baseline |

---

*Document generated: 2026-03-24*
*Pragya Lab, BITS Pilani Goa*
*Swiss Knife — Experiment Planning Document v1.0*
