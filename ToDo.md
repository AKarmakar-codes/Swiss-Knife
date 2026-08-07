# Swiss Knife — Paper ToDo

##  Ablation Studies — Teammate Instructions

> All commands are run from the **project root** (`Swiss-Knife/`).
> **Single entry point for both tests:** `evaluation/run_hh_experiments.py` — passing `--test all` (or omitting `--test`) runs both ablation experiments sequentially.

### Executive Quick Start — Run Both Tests at Once

If you want to execute **both ablation tests together**, use the commands below:

#### Phase 1 — Combined Generation (requires GPU)
```bash
python evaluation/run_hh_experiments.py \
    --test all \
    --mode generate \
    --num_samples 50 \
    --max_new_tokens 768
```

#### Phase 2 — Tribunal LLM-as-Judge Evaluation (requires GPU)
```bash
# Terminal 1 — start judge server
cd tribunal
python serve_judge.py

# Terminal 2 — score both test outputs
cd tribunal
python -m tribunal.run_eval --input inputs/sigma_validity --output outputs/sigma_validity
python -m tribunal.run_eval --input inputs/tournament_value --output outputs/tournament_value
```

#### Phase 3 — Combined Offline Analysis (no GPU needed)
```bash
python evaluation/run_hh_experiments.py \
    --test all \
    --mode analyze
```

---

### Test 1 — `evaluation/test_sigma_validity.py`
**Goal:** Determine whether the `log_ratio_proxy` sigma (σ) measures genuine candidate uncertainty or is just random noise.
It compares three σ conditions: **real σ** (actual log-ratio proxy), **shuffled σ** (same values, randomly reassigned to candidates), and **zero σ** (σ=0, collapses to plain Bradley-Terry Elo).
If real σ beats shuffled and zero, it is doing real work.



#### Phase 1 — Generate (requires GPU)
```bash
python evaluation/run_hh_experiments.py \
    --test sigma_validity \
    --mode generate \
    --num_samples 50 \
    --max_new_tokens 768
```
- Loads 50 prompts from the Anthropic HH-RLHF harmlessness test split (seed 42).
- Runs three generators (real σ / shuffled σ / zero σ) on all prompts.
- Writes three Tribunal input JSONL files to `tribunal/inputs/sigma_validity/`:
  - `elo_real_sigma.jsonl`
  - `elo_shuffled_sigma.jsonl`
  - `elo_zero_sigma.jsonl`
- Also writes per-step stats to `runs/sigma_validity/step_sigma_stats.json`.
- Live logs are persisted to `runs/logs/experiment_run.log`.

> **Sanity check before proceeding:** confirm those three `.jsonl` files exist and are non-empty.







#### Phase 2 — Tribunal (LLM-as-Judge, requires GPU)
Open two terminals. In terminal 1, start the judge server and wait until it prints "Model loaded":
```bash
cd tribunal
python serve_judge.py
```
In terminal 2, point Tribunal at the sigma validity inputs:
```bash
cd tribunal
python -m tribunal.run_eval \
    --input inputs/sigma_validity \
    --output outputs/sigma_validity
```
Wait until scoring completes. You should see three `*_eval.csv` files appear in `tribunal/outputs/sigma_validity/`.

#### Phase 3 — Analyze (no GPU needed)
```bash
python evaluation/run_hh_experiments.py \
    --test sigma_validity \
    --mode analyze
```
- Reads `tribunal/outputs/sigma_validity/` CSVs and `runs/sigma_validity/step_sigma_stats.json`.
- Computes composite **Scalar Objective Score** (matching `scalar_objective` in `parameter_search_optimized.py`: harmonic mean of Quality & Safety) and reports all 6 individual Tribunal rubrics: `response_quality`, `relevance`, `helpfulness`, `toxicity`, `harmfulness`, `refusal`.
- Stratifies prompts into **Low / Medium / High σ tiers** and prints win rates + objective deltas for real σ vs. shuffled and vs. zero.
- Saves summary tables and plots to `runs/tribunal_plots/sigma_validity/`.

> **Key numbers to record for the paper:**
>
> *Tribunal outcome stats & Objective Function (primary ground-truth claims):*
> - Composite ΔScalar Objective & ΔQuality (real σ − shuffled σ, real σ − zero σ) overall and per σ-tier
> - All individual Tribunal rubric scores (`response_quality`, `relevance`, `helpfulness`, `toxicity`, `harmfulness`, `refusal`) reported separately for real σ, shuffled σ, and zero σ.
> - ΔSafety = Δ(1 − mean(toxicity, harmfulness)) — is the gain quality-driven or safety-driven?
> - `refusal_delta` (Tribunal refusal score: real σ − shuffled σ) — are we winning by refusing less, or by being genuinely better?
>
> *Behavioural bridge stats (pre-outcome; explain when σ actually activated):*
> - `upset_rate` — fraction of steps where real-σ champion ≠ argmax(μ). **If near zero, UWO never fired and ΔObjective is noise.**
> - `mean_champion_sigma_rank` — σ-rank (0=lowest σ) of the champion chosen by real-σ. If UWO works, this should be below pool average.
> - `mean_sigma_spread` — std(σ) across candidates per step. Low spread = UWO ranking is near-random regardless of real vs. shuffled.
>
> *Compositional/cascade stats (explain why final response quality differs):*
> - `first_divergence_step` — which step real-σ and zero-σ first diverge. Early divergence cascades into very different responses.
> - `total_divergence_steps` — absolute count of diverging steps (not just rate).
> - `response_length_delta` — token length difference. Must control for length bias.
>
> *Subset-defining conditional checks (pre-outcome, for Opus 5 hypothesis):*
> - Composite ΔObjective conditional on `upset_rate > 0` vs. `== 0` — does σ only help when UWO actually changed the pick?
> - Composite ΔObjective conditional on `first_divergence_step < 5` vs. late — does early cascade matter?
> - Composite ΔObjective conditional on `mean_sigma_spread > threshold` — is σ discriminative enough to matter?

---

### Test 2 — `evaluation/test_tournament_value.py`
**Goal:** Determine whether the **Thurstonian probabilistic tournament** (`probabilistic=True`) outperforms a deterministic Elo baseline (`elo_baseline`, `w_blade=0`, `probabilistic=False`) and a plain softmax-over-rewards baseline.
Three strategies are compared: **Thurstonian** (full Mode B), **Elo Baseline** (`elo_baseline`, deterministic tournament-only, no UWO), **Softmax Blade** (no tournament at all).

#### Phase 1 — Generate (requires GPU)
```bash
python evaluation/run_hh_experiments.py \
    --test tournament_value \
    --mode generate \
    --num_samples 50 \
    --max_new_tokens 768
```
- Loads the same 50 HH-RLHF prompts (seed 42) used in Test 1.
- Runs all three strategy generators (Thurstonian / Elo Baseline / Softmax Blade).
- Writes three Tribunal input JSONL files to `tribunal/inputs/tournament_value/`:
  - `elo_thurstonian.jsonl`
  - `elo_baseline.jsonl`
  - `elo_softmax_blade.jsonl`
- Also writes `runs/tournament_value/step_tournament_stats.json`.
- Live logs appended to `runs/logs/experiment_run.log`.

> **Sanity check:** confirm those three `.jsonl` files exist and are non-empty.

#### Phase 2 — Tribunal (LLM-as-Judge, requires GPU)
```bash
# Terminal 1 — start judge (skip if already running from Test 1)
cd tribunal
python serve_judge.py

# Terminal 2 — run evaluation
cd tribunal
python -m tribunal.run_eval \
    --input inputs/tournament_value \
    --output outputs/tournament_value
```
Wait for three `*_eval.csv` files in `tribunal/outputs/tournament_value/`.



















#### Phase 3 — Analyze (no GPU needed)
```bash
python evaluation/run_hh_experiments.py \
    --test tournament_value \
    --mode analyze
```
- Reads `tribunal/outputs/tournament_value/` CSVs and `runs/tournament_value/step_tournament_stats.json`.
- Computes composite **Scalar Objective Score** (matching `scalar_objective` in `parameter_search_optimized.py`: harmonic mean of Quality & Safety) and reports all 6 individual Tribunal rubrics: `response_quality`, `relevance`, `helpfulness`, `toxicity`, `harmfulness`, `refusal`.
- Stratifies prompts into **High / Medium / Low Ambiguity tiers** (by mean Δμ across the candidate pool).
- Computes data-driven Spearman feature correlations across all pre-outcome step statistics and composite ΔObjective.
- Prints the **Divergence-Conditioned Table** (comparing prompts where Thurstonian intervened vs. where it remained idle).
- Discovers and exports the **Top-Quartile Confident Subset** (`confident_subset.csv`) where Thurstonian gains are largest.
- Saves bar charts (`tournament_value_gap.png`) and scatter plot (`delta_quality_vs_delta_mu_scatter.png`).

> **Key numbers to record for the paper:**
>
> *Tribunal outcome stats & Objective Function (primary ground-truth claims):*
> - Composite ΔScalar Objective & ΔQuality (Thurstonian − Elo Baseline) overall and per ambiguity tier
> - All individual Tribunal rubric scores (`response_quality`, `relevance`, `helpfulness`, `toxicity`, `harmfulness`, `refusal`) reported separately for Thurstonian, Elo Baseline, and Softmax Blade.
> - ΔSafety = Δ(1 − mean(toxicity, harmfulness)) — verifies safety retention.
> - `refusal_delta` (Tribunal refusal score: Thurstonian − Elo Baseline). **Critical:** if Thurstonian wins only by refusing more, the tournament mechanism is not the cause — the harmlessness blade is.
>
> *Behavioural bridge stats (pre-outcome; explain when tournament activated):*
> - `t_base_disagreement_rate` / `t_bt_disagreement_rate` — fraction of steps where Thurstonian chose a different champion than `elo_baseline`. **Primary activation signal: if disagreement_rate ≈ 0, Thurstonian and Baseline are identical and any ΔObjective is noise.**
> - `mean_champion_sigma_rank` — σ-rank of Thurstonian’s champion (0=least uncertain). If UWO works, should be significantly below pool average (N-1)/2.
> - `mean_sigma_spread` — std(σ) across candidates. Low spread = UWO ranking is near-random; high spread = σ genuinely discriminates.
> - `base_greedy_agreement_rate` — fraction of steps where `elo_baseline` == argmax(μ). Confirms `elo_baseline` is a clean greedy-sort baseline.
>
> *Compositional/cascade stats (explain why final response quality differs):*
> - `first_divergence_step` — step index of first T vs. Baseline champion disagreement. Step 0 divergence → almost entirely different responses.
> - `total_divergence_steps` — absolute count of diverging steps (the “effective dose” of the tournament’s intervention on the response).
> - `response_length_delta` — Thurstonian response − Baseline response in tokens. Must control for length bias.
>
> *Subset-defining conditional checks (pre-outcome, for Opus 5 hypothesis):*
> - Composite ΔObjective and rubric deltas conditional on `t_base_disagreement_rate > 0.3` vs. `≤ 0.3`
> - Composite ΔObjective conditional on `first_divergence_step < 5` vs. later
> - Composite ΔObjective conditional on `mean_sigma_spread > 0.05`
> - `refusal_delta` sign — if Thurstonian wins AND refusal_delta < 0 (less refusal), the win is genuine quality; if refusal_delta > 0, it’s a safety-collapse avoidance story

---


















### Phase 4 — Opus 5 Hypothesis: Finding the Prompt Subset Where Thurstonian Wins

After both Analyze steps are done, feed the following to **Claude Opus** (or equivalent frontier model):

> **Prompt template for Opus 5:**
>
> *"You are helping us write the Analysis section of an AAAI 2027 paper on Swiss Knife, a decode-time alignment system. The core novelty is `elo_swiss_mode_b`: a Thurstonian probabilistic Elo tournament that selects a decoding-step champion unconditionally, using an uncertainty-weighted objective (UWO, μ − λσ) to penalise high-σ candidates.*
>
> *We ran two ablation experiments on the Anthropic HH-RLHF harmlessness test split. The Tribunal metrics (response_quality, relevance, helpfulness, toxicity, harmfulness, refusal) and the composite Scalar Objective function (harmonic mean of Quality and Safety) are the ground-truth outcome variables. μ and σ per step are mechanism-level diagnostics. Behavioural stats (t_base_disagreement_rate, champion_sigma_rank, first_divergence_step, total_divergence_steps, etc.) are the bridge — they capture whether and how the tournament’s different selection actually propagated into a different final response.*
>
> *State a single, falsifiable subset hypothesis of this exact form:*
> *"Thurstonian significantly outperforms Elo Baseline (ΔObjective > X) if and only if [condition using only pre-outcome step stats], because [causal mechanism linking step-level selection to final response quality]. Outside this regime, ΔObjective ≈ 0 because [explanation]."*
>
> *Requirements for the hypothesis:*
> *1. Use ONLY pre-outcome features: t_base_disagreement_rate, mean_sigma_spread, first_divergence_step, total_divergence_steps, mean_delta_mu, mean_champion_sigma_rank. Do NOT use Tribunal scores as a condition.*
> *2. Name a specific numeric threshold for every feature used.*
> *3. Explain the mechanism — why does the Thurstonian CDF resolution of ambiguous matches produce a better final response in that regime?*
> *4. State the null prediction: what should ΔObjective look like on prompts that fail the condition?*
> *5. Report all individual Tribunal rubrics (quality, relevance, helpfulness, toxicity, harmfulness, refusal) alongside ΔObjective, and note whether refusal_delta is positive or negative — this separates genuine quality improvement from safety-collapse avoidance.*
>
> *Attached data:*
> *- `runs/sigma_validity/sigma_validity_summary.csv` — Composite ΔObjective, all 6 Tribunal rubrics, ΔSafety, refusal_delta, win rates per σ-tier, upset_rate, champion_sigma_rank, sigma_spread, first_divergence_step*
> *- `runs/tournament_value/tournament_value_summary.csv` — same metrics per ambiguity tier (mean Δμ), plus total_divergence_steps, base_greedy_agreement_rate, response_length_delta*
> *- `runs/tournament_value/confident_subset.csv` — full per-prompt feature table for the top-quartile ΔObjective prompts (including all 6 Tribunal rubrics)*
> *- Scatter plot: `delta_quality_vs_delta_mu_scatter.png` (x=mean Δμ, y=ΔObjective T−Baseline, colour=mean σ)"*

- [ ] Record Opus 5’s hypothesis, the specific feature thresholds, and the null prediction in the paper’s Analysis section.
- [ ] Verify each threshold: split `confident_subset.csv` by the proposed condition and confirm within-condition win rate is materially higher than out-of-condition win rate.
- [ ] Check refusal_delta sign in the subset to confirm the win story (genuine quality vs. safety-collapse avoidance).

---
