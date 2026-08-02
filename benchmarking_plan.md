# Swiss Knife: Benchmarking and Evaluation Plan
## Target: AAAI-27, AI Alignment Track — One Week Deadline

---

## 1. Baseline Strategy Map

All baselines in the decode-time alignment space cluster into four **family types**, which is how reviewers will think about them. Your claim is: **tournament selection over DPO blade candidates beats all four families on helpfulness/harmlessness without retraining**.

| Family | Core Mechanism | Representative |
|---|---|---|
| **Greedy reranking** | Score N full responses, pick best | Best-of-N (BoN) |
| **Token-level logit steering** | Adjust per-token distribution at each step | ARGS, MOD |
| **Heuristic search** | A* or beam search guided by reward | DeAL |
| **Speculative alignment** | Draft-then-verify with reward tilt | GSI (Geuter et al., ICLR 2026) |

Swiss Knife sits at the boundary of families 3 and 4, and its distinguishing property is: **tournament-based, uncertainty-aware, calibration-invariant selection with O(1) objective switching**.

---

## 2. Baselines — Tiered by Priority

### TIER 1: Absolutely Required (Reviewers will expect these)
*These are the minimum necessary for acceptance. Missing any of these is a desk-rejection risk.*

---

#### T1-A: Best-of-N (BoN) sampling
**Paper:** Nakano et al., 2021 (WebGPT). Well-established standard.
**Why mandatory:** The universal baseline for inference-time alignment. Every paper in this space compares against it. N=5 or N=8 (same candidate budget as Swiss Knife `gsi_n`).
**Implementation difficulty:** 1/5 — trivially simple. Generate N responses, pick argmax reward.
**Your reward signal:** Use the harmlessness or helpfulness blade `r_blade` score on full responses. No tournament needed.
**Notes:** Use same N as your `gsi_n` to make the comparison fair (same compute budget).

---

#### T1-B: Greedy Argmax (Unaligned Baseline)
**Why mandatory:** This is your "floor" — the Qwen 2.5 7B SFT model with no alignment at all. Without this, you cannot claim improvement.
**Implementation difficulty:** 1/5 — already done (you have `baseline_argmax` in your harness).
**Notes:** Already implemented as `baseline` in your benchmark scripts.

---

#### T1-C: ARGS (Alignment as Reward-Guided Search)
**Paper:** Khanov et al., ICLR 2024. Available at: `github.com/deeplearning-wisc/args`.
**Why mandatory:** ARGS is the most-cited token-level reward steering baseline. DiffPO, PAD, and DeAL all compare against it. If you skip it, reviewers will notice.
**Mechanism:** At each decoding step, adjusts the next-token probability distribution by adding `alpha * r(x, y_prefix)` to the logits, where `r` is an external reward signal.
**Implementation difficulty:** 2/5 — You already have `r_blade` as your reward. The key modification is: instead of sampling step-level candidates and running a tournament, you run a single forward pass and add `beta * r_blade_logprob` to the base logits at each token step. This is a token-level steering loop, not a step-level tournament.
**HH-RLHF dataset compatible:** Yes, they used it.

---

#### T1-D: MOD (Multi-Objective Decoding)
**Paper:** Shi et al., NeurIPS 2024. `arxiv.org/abs/2408.15267`.
**Why mandatory:** MOD is the direct linear-mixture competitor. Your calibration-invariance argument only lands if you show that MOD (which IS vulnerable to reward miscalibration via additive constants) performs worse in misaligned conditions. MOD is also training-free like Swiss Knife.
**Mechanism:** Generates next token as a linear combination of logit distributions from multiple specialized models: `log p_combined = w1 * log p_helpful + w2 * log p_harmless`.
**Implementation difficulty:** 3/5 — You need to run both your helpfulness and harmlessness blades simultaneously at each token step, combine their log-probability distributions linearly, and argmax. The blades' LoRA adapters need to be loaded for both simultaneously.
**Critical note:** This is the most important comparison theoretically. Your Thurstonian tournament vs. MOD's linear mixture is the heart of the paper's claim.

---

#### T1-E: DeAL (Decoding-time ALignment)
**Paper:** Huang et al. (USC/Amazon), arXiv 2402.06147v3.
**Why mandatory:** DeAL also targets HH-RLHF (helpfulness + harmlessness) with parametric reward heuristics. It's directly on your benchmark dataset.
**Mechanism:** A* search at token level with greedy lookahead. At each step, scores top-k token extensions using a reward model and selects the best.
**Implementation difficulty:** 3/5 — The core loop is: sample top-k candidate tokens, run `r_blade` forward on each k-token extension, pick argmax. The lookahead mechanism is optional for a basic comparison (you can use lookahead=0 which reduces to token-level reranking).
**Notes:** Even a simplified DeAL (no lookahead, top-k=n like your gsi_n) is publishable as a comparison. Cite the paper correctly and note that you use the parametric reward variant with your DPO blade.

---

### TIER 2: Strongly Recommended (Adds robustness and breadth)
*Add these if time permits. Each takes 1-2 days to implement. A paper with Tier 1 + 2 is significantly stronger.*

---

#### T2-A: DiffPO
**Paper:** Chen et al., ACL 2025. You have the full paper and txt.
**Relevance:** DiffPO also uses HH-RLHF and reports Helpful/Harmless scores. You can pull their published numbers directly without reimplementing (since they use Llama-3 and Mistral, not Qwen, the comparison is cross-model).
**Implementation difficulty:** 4/5 to reimplement yourself. However, 1/5 if you just cite their published HH-RLHF scores in a comparison table with a footnote: "DiffPO numbers from Chen et al. (2025), evaluated on Llama-3-8B-SFT."
**Recommended approach:** Cite their numbers. The different base model is a valid caveat, but it gives reviewers a useful reference point.

---

#### T2-B: PAD (Personalized Alignment at Decoding-time)
**Paper:** Chen et al., arXiv 2410.04070v7. You have the full paper and txt.
**Relevance:** PAD also compares on harmless/helpful dimensions on HH-RLHF and P-Soups. Cross-model comparison is fine to cite.
**Implementation difficulty:** 4/5 to reimplement. 1/5 to cite.
**Recommended approach:** Cite their published numbers from Table 2. Note the caveat (Llama-3-8B backbone vs. your Qwen 2.5 7B SFT).

---

#### T2-C: Ablations of Swiss Knife itself
**These are internal comparisons, not external baselines, but AAAI reviewers will demand them.**

| Ablation | What it tests | Implementation |
|---|---|---|
| `swiss_mode_b` vs `elo_swiss_mode_b` | Tournament alone vs. tournament + Elo rating system | Already implemented |
| `gsi_softmax` vs `elo_swiss_mode_b` | Softmax selection vs. Thurstonian tournament | Already implemented |
| Mode A (`elo_swiss`) vs Mode B (`elo_swiss_mode_b`) | With vs. without verifier fallback | Already implemented |
| `--probabilistic` ON vs OFF | Thurstonian vs. Bradley-Terry match outcomes | Already implemented |
| `lambda=0` vs `lambda=0.3` | UWO uncertainty penalty ablation | Config param |
| `n=3` vs `n=5` vs `n=8` | Candidate count ablation | Config param |

**Implementation difficulty:** 1/5 — all already in your codebase. Just run them.

---

#### T2-D: Reward-Shifted Speculative Sampling (RSSS)
**Paper:** You have the PDF (`Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner.pdf`).
**Relevance:** This is close to GSI and to your Mode A. It applies reward shifts during speculative sampling.
**Implementation difficulty:** 3/5
**Recommended approach:** Read the paper, check if their benchmark is HH-RLHF or reasoning. If not HH-RLHF, deprioritize.

---

#### T2-E: GSI (Guided Speculative Inference, Geuter et al., ICLR 2026)
**Paper:** You have the full paper (`gsi_paper.txt`). This is the theoretical parent of your method.
**Relevance:** You extend GSI by replacing its softmax selection with a Thurstonian Elo tournament. You MUST cite this paper and ideally show you outperform it.
**Implementation difficulty:** 2/5 — your existing `gsi_softmax` generator IS essentially GSI Mode A (with tilted rewards). Run it on the same HH-RLHF benchmark and label it "GSI baseline" in the table.
**Important:** GSI was designed for reasoning benchmarks (MATH500, OlympiadBench), NOT for HH-RLHF. Your contribution is adapting the speculative inference paradigm to alignment objectives with DPO blades instead of PRMs. Make this distinction clearly in the paper.

---

### TIER 3: Optional / Future Work
*Do not attempt within a one-week deadline. Reference them in related work.*

---

| Baseline | Reason to skip | Reference |
|---|---|---|
| **Collab** (collaborative decoding) | Targets factual tasks, not HH-RLHF | Shi et al., 2024 |
| **GenARM** (generative reward modeling) | Requires a generative reward model, not a DPO blade | arXiv 2410.08193 |
| **Controlled Decoding (CD)** | Requires training a prefix-scorer; very different setup | Mudgal et al., 2024 |
| **RLHF-PPO baseline** | Requires full training run on your SFT model | Too slow |
| **SimPO** | Training-time method, not inference-time | Meng et al., 2024 |
| **MetaAligner** | Requires training an additional corrective model | Yang et al., 2024 |

---

## 3. Implementation Order for a One-Week Deadline

```
Day 1:   Run all existing Swiss Knife strategies (T1-E-equivalent via gsi_softmax,
         swiss, elo_swiss, elo_swiss_mode_b) on HH-RLHF + helpfulness benchmarks.
         → Produces your main ablation table.

Day 2:   Implement BoN (T1-A). 5 lines of code. Run on same prompts.
         Verify baseline_argmax already works (T1-B).

Day 3:   Implement ARGS (T1-C). Token-level `beta * r_blade` steering loop.
         Run on same 15-50 prompts.

Day 4:   Implement simplified DeAL (T1-E). Top-k token candidates + r_blade scoring.
         No lookahead required for a pilot comparison.

Day 5:   Implement MOD (T1-D). Linear combination of helpfulness + harmlessness
         blade logits at each token step.

Day 6:   Scale up to 50+ prompts. Run tribunal evaluation on all strategies.
         Collect results for the main table.

Day 7:   Write results section. Pull DiffPO/PAD published numbers for reference table.
         Finalize the paper.
```

---

## 4. Metric Sufficiency Analysis

### Your Current 6 Tribunal Metrics
`response_quality`, `relevance`, `helpfulness`, `toxicity`, `harmfulness`, `refusal`

### Are They Sufficient for AAAI?

**Short answer: Mostly yes, but add 2 more.**

| Current Metric | Alignment Track Sufficiency | Notes |
|---|---|---|
| `response_quality` | Good | Maps to "overall alignment quality" |
| `relevance` | Good | Measures prompt-adherence |
| `helpfulness` | Good | Required for HH-RLHF |
| `toxicity` | Good | Standard safety metric |
| `harmfulness` | Good | Core claim metric |
| `refusal` | Good | Critical to distinguish over-refusal from safety |

### 2 Metrics You Should Add

**Add-1: ArmoRM Score (Reward Model Score)**
- **What:** Run `RLHFlow/ArmoRM-Llama3-8B-v0.1` (or `internlm/internlm2-7b-reward`) on all responses. Report the mean reward model score per strategy.
- **Why:** ArmoRM is currently the dominant automatic alignment evaluation tool in the field (DiffPO, PAD, and most 2024-2025 papers report it). AAAI reviewers will expect it as an objective, reproducible score alongside LLM-as-judge.
- **Difficulty:** 2/5. One inference pass over all collected responses. No judge API needed.

**Add-2: Calibration Invariance Test (Swiss Knife-specific)**
- **What:** Report whether adding a constant (+1000) to all blade scores changes tournament outcomes. This validates your theoretical claim.
- **Why:** This is your unique claim and no other baseline can pass this test. It becomes a structural differentiator in the paper.
- **Difficulty:** 1/5. Already done empirically in your tests. Just format it for the paper.

### Metrics You Do NOT Need

| Metric | Reason to Skip |
|---|---|
| BLEU / ROUGE | Not relevant for open-ended alignment |
| Perplexity | Does not measure alignment |
| MT-Bench | Requires GPT-4 judge; too broad for your scope |
| AlpacaEval 2 | Win-rate against GPT-4; cross-model, hard to compare |

---

## 5. Evaluation Summary Table (Template)

Fill in your results:

| Strategy | Helpfulness (ArmoRM) | Harmlessness (ArmoRM) | Refusal Rate | Toxicity | Quality (Judge) | Tok/s |
|---|---|---|---|---|---|---|
| Baseline (Argmax) | --- | --- | --- | --- | --- | --- |
| Best-of-N (N=5) | --- | --- | --- | --- | --- | --- |
| ARGS | --- | --- | --- | --- | --- | --- |
| MOD | --- | --- | --- | --- | --- | --- |
| DeAL (simplified) | --- | --- | --- | --- | --- | --- |
| GSI Softmax | --- | --- | --- | --- | --- | --- |
| Swiss-mode-B | --- | --- | --- | --- | --- | --- |
| **Elo-Swiss-mode-B (ours)** | --- | --- | --- | --- | --- | --- |

> [!IMPORTANT]
> The `Tok/s` column matters for the AAAI Alignment track because it demonstrates practical deployability. Your ~14 tok/s number shows Swiss Knife is competitive with real-time constraints.

---

## 6. Key Differentiators to Highlight in the Results

These are the claims your benchmark MUST support:

1. **Swiss Knife > BoN** on helpfulness at the same candidate budget N — shows tournament beats greedy argmax
2. **Swiss Knife > MOD** on harmlessness — shows calibration-invariant tournament beats linear mixture
3. **Swiss Knife > ARGS** on refusal rate — shows step-level tournament avoids over-refusal better than token-level steering
4. **Swiss Knife 0.00 toxicity** across all test prompts — replicated from your tribunal report
5. **Mode B > Softmax (gsi_softmax)** on helpfulness — shows Elo tournament beats naive softmax selection
6. **Calibration invariance test: PASS** — shows +1000 additive shift does not change tournament outcomes (unique structural claim)
