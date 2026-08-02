# Swiss-Knife Mode-B Logit Mixing Specification

### 1. Conceptual Paradigm: MOD vs. Swiss-Knife Mode-B

| Feature | Multi-Objective Decoding (MOD) | Swiss-Knife Mode-B (Elo-Swiss) |
| :--- | :--- | :--- |
| **Mixing Granularity** | **Token-level** linear log-probability sum | **Step/Span-level** candidate scoring & tournament |
| **Logit Formula** | `log P_MOD(y_t) = w_1 * log P_help(y_t) + w_2 * log P_harm(y_t)` | `logit_i = w_tournament * Z-norm((R_i - 1500) / T) + w_blade * Z-norm(mu_composite_i - lambda * sigma_composite_i)` |
| **Robustness to Spikes** | **Low**: A single rogue logit spike in one model degrades generation | **High**: Thurstonian Elo ranking flattens extreme outliers via variance normalization |
| **Uncertainty Awareness** | None (assumes calibrated log probabilities) | **Explicit UWO**: Penalizes candidate uncertainty (`sigma_i`) via Log-Ratio Proxy |
| **Acceptance Policy** | Greedy/Nucleus token sampling | **Unconditional Acceptance** of Elo Champion (low latency, zero gating overhead) |

---

### 2. Multi-Blade Logit Mixing Mechanics in Mode B

The logit mixing in Swiss-Knife Mode B occurs at the **candidate step selection stage**. Given a prompt prefix `x`:

#### Step A: Draft Candidate Generation
The base model `pi_draft` samples `N` candidate reasoning steps: `S_1, S_2, ..., S_N`.

#### Step B: Multi-Blade Scoring & Uncertainty Estimation
For each candidate step `S_i`, scores are computed using both the **Helpfulness Blade** and **Harmlessness Blade**:

1. **Helpfulness Blade**:
   - Reward: `mu_help_i = r_help(x, S_i)`
   - Uncertainty: `sigma_help_i = abs(mu_help_i - (1 / beta) * (log pi_help(S_i) - log pi_draft(S_i)))`

2. **Harmlessness Blade**:
   - Reward: `mu_harm_i = r_harm(x, S_i)`
   - Uncertainty: `sigma_harm_i = abs(mu_harm_i - (1 / beta) * (log pi_harm(S_i) - log pi_draft(S_i)))`

3. **Composite Multi-Blade Reward & Uncertainty**:
   Given objective Pareto weight `gamma_help` in range `[0, 1]` and `gamma_harm = 1.0 - gamma_help`:
   - Composite Reward: `mu_composite_i = gamma_help * mu_help_i + gamma_harm * mu_harm_i`
   - Composite Uncertainty: `sigma_composite_i = gamma_help * sigma_help_i + gamma_harm * sigma_harm_i`

---

#### Step C: Thurstonian Elo Tournament
Candidates enter a `K`-round Elo tournament with initial rating `R_i = 1500.0` for all `i`. Pairwise matches between candidate `A` and `B` are resolved using the **Thurstonian Case-V Model**:

- `P(A beats B) = Normal_CDF( (mu_composite_A - mu_composite_B) / sqrt(sigma_composite_A^2 + sigma_composite_B^2 + epsilon) )`

- **Elo Updates**: Following match outcomes, Elo ratings `R_i` are updated across rounds (using decaying K-factors: `40, 32, 24, 16, 12, 10`):
  - `R_A_new = R_A + K_round * ( P(A beats B) - Sigmoid((R_A - R_B) * ln(10) / 400) )`

---

#### Step D: Final Z-Normalized Champion Logit Synthesis

To compute final selection probabilities across candidates, Mode B synthesizes the **Tournament Rating** logit and the **Uncertainty-Weighted Objective (UWO) Blade** logit:

- `logit_i = w_tournament * Z-norm( (R_i - 1500) / T_elo ) + w_blade * Z-norm( mu_composite_i - lambda_uwo * sigma_composite_i )`

where Z-normalization is defined as:
- `Z-norm(X_i) = (X_i - mean(X)) / (std(X) + 1e-6)`

> **Why Z-Normalization is Essential**: Raw Elo ratings operate in the scale of `[1400, 1600]` (divided by temperature `T`, giving raw values around `+/- 50`), whereas DPO rewards `mu` operate in `[-1.0, 1.0]`. Without Z-normalization, the tournament term completely overwhelms the blade term regardless of your settings. Z-normalization maps both components to zero mean and unit variance across candidate options, making `w_tournament` and `w_blade` true controllable hyperparameters.

Finally, the champion step index is drawn via a softmax draw over the logits:

- `P(Select Step i) = exp(logit_i) / sum_j(exp(logit_j))`

The selected step is **unconditionally accepted** (Mode B hallmark), eliminating verifier fallback latency.

---

### 3. Implementation Code: Multi-Blade Mode B Generator

Below is how to extend `elo_swiss_mode_b.py` and `elo_system.py` to evaluate dual blades (Helpfulness + Harmlessness):

```python
import torch
import torch.nn.functional as F
from Model_mechanics.elo_system import elo_bracket
from Model_mechanics.sigma_estimator import estimate_mu_sigma

def compute_dual_blade_mode_b_step(
    prefix_ids: torch.Tensor,
    candidate_step_ids: list,
    draft_logprobs: torch.Tensor,
    helpfulness_blade,
    harmlessness_blade,
    gamma_help: float = 0.5,
    beta: float = 0.1,
    cfg = None
) -> int:
    """
    Computes step-level candidate selection for Mode B using dual DPO blades.
    """
    gamma_harm = 1.0 - gamma_help
    
    # 1. Score Helpfulness Blade & Uncertainty
    mu_help, sigma_help = estimate_mu_sigma(
        prefix_ids=prefix_ids,
        step_token_ids_list=candidate_step_ids,
        blade=helpfulness_blade,
        sigma_mode=cfg.sigma_mode,
        draft_logprobs=draft_logprobs,
        beta=beta,
    )

    # 2. Score Harmlessness Blade & Uncertainty
    mu_harm, sigma_harm = estimate_mu_sigma(
        prefix_ids=prefix_ids,
        step_token_ids_list=candidate_step_ids,
        blade=harmlessness_blade,
        sigma_mode=cfg.sigma_mode,
        draft_logprobs=draft_logprobs,
        beta=beta,
    )

    # 3. Composite Multi-Blade Reward & Uncertainty
    mu_composite = gamma_help * mu_help + gamma_harm * mu_harm
    sigma_composite = gamma_help * sigma_help + gamma_harm * sigma_harm

    # 4. Perform Thurstonian Elo Tournament & Z-Norm Champion Selection
    champion_idx = elo_bracket(
        target_scores=draft_logprobs,
        blade_scores=mu_composite,
        alpha=cfg.alpha,
        normalize=True,
        temperature=cfg.elo_temperature,
        rounds=cfg.elo_rounds,
        beta=beta,
        sigmas=sigma_composite if cfg.sigma_mode != "none" else None,
        hard_draw=cfg.hard_draw,
        w_tournament=cfg.w_tournament,
        w_blade=cfg.w_blade,
        uwo_lambda=cfg.uwo_lambda,
        probabilistic=True,
    )

    return champion_idx
```

---

### 4. Key Advantages for AAAI Pareto Benchmarks vs. MOD

1. **Non-Convex Pareto Control**: Adjusting `gamma_help` in range `[0, 1]` smoothly traces out superior Pareto frontiers on Helpfulness vs. Harmlessness compared to MOD.
2. **Elimination of Syntax Corruption**: Generating multi-token candidate steps from `pi_draft` guarantees syntactic coherence, whereas MOD's token-level logit addition often mixes contradictory token probabilities.
3. **Robustness via Thurstonian CDF**: In uncertainty-heavy regions (where blades disagree or predict high variance), the Thurstonian match outcome probability flattens toward `0.5`, protecting generation from over-exploiting noisy reward estimates.
