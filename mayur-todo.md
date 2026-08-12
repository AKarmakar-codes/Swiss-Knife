# Pending Benchmarking Tasks

Please pick up one of the following benchmarking tasks for our AAAI paper submission:

### Option 1: Benchmark Mode B on HarmfulQ
We need to benchmark our primary strategy (`Model_mechanics/elo_swiss_mode_b.py`) on the `harmfulQ` dataset. 

Please use the hyperparameters from **Round 0 Config 22** (`r0config22`), which were found during our Bayesian search for **Harmlessness** (`all_observations-harmlessness.csv`). 

To save you time pulling them from the script, here are the exact parameters to use:
- `--elo-temperature`: `28.57587`
- `--w-tournament`: `0.74063`
- `--w-blade`: `2.00907`
- `--uwo-lambda`: `0.82332`
- `--elo-rounds`: `9`
- `--gsi-n`: `11`

*(Note: `--beta` remains fixed at `0.1` as per the evaluation scripts.)*

### Option 2: Run the Dual Blade Logit Mixing Experiment
Alternatively, run the logit mixing experiment (dual blade) to evaluate our Pareto frontier control against Multi-Objective Decoding (MOD). 

You can use the new generator built in `Model_mechanics/elo_swiss_dual_blade_mode_b.py`, which implements the candidate-batch normalization logic for properly mixing the Helpfulness and Harmlessness blades.

For the **Helpfulness** blade configuration, use **Round 1 Config 8** (`r1config8`) from `all_observations-helpfulness.csv`:
- `--elo-temperature`: `9.55040`
- `--w-tournament`: `1.63081`
- `--w-blade`: `0.24521`
- `--uwo-lambda`: `0.00285`
- `--elo-rounds`: `7`
- `--gsi-n`: `13`

use bfloat16 for everything

Daywise Plan:
Sunday :
    1. Mayur: Complete training Honesty blade
    2. Agnibh: Logit mixing experiment with helpfulness and harmfulness
    3. Shreyash : Start Bayesian Search on Honesty 
Monday :
    1. Agnibh: Benchmark Mode B on HarmfulQ
    2. 
    3. 