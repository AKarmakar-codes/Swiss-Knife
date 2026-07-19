#!/usr/bin/env python3
"""
Bayesian Hyperparameter Search for Swiss-Knife ``elo_swiss_mode_b`` GSI Strategy
==================================================================================
Single-GPU sequential search. Designed for a single NVIDIA DGX GPU with 128 GB
unified memory where generation and judging MUST NOT overlap in VRAM.

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETERS BEING SEARCHED
─────────────────────────────────────────────────────────────────────────────
  elo_temperature  [1.0,  40.0]  Temperature T in the UWO logit: (R_i-1500)/T.
                                 Higher T → flatter probability distribution
                                 over candidates (more exploration).
  beta             [0.01, 1.0]   DPO regularization strength. Scales the
                                 KL-divergence penalty between the blade policy
                                 and the base verifier. Small β → blade diverges
                                 freely; large β → stays close to base verifier.
  w_tournament     [0.0,  3.0]   Weight for the Elo rating term in Step C UWO
                                 logit. Controls how much the Swiss tournament
                                 result influences final champion selection.
  w_blade          [0.0,  3.0]   Weight for the blade UWO term (μ - λσ) in
                                 Step C. Controls how much the DPO blade reward
                                 and its uncertainty influence selection.
  uwo_lambda       [0.0,  1.0]   Uncertainty penalty λ. Penalises candidates
                                 with high σ (risk-averse selection).
  elo_rounds       [2,    10]    Number of Elo rating update rounds inside the
                                 Swiss tournament bracket.
  gsi_n            [3,    16]    Number of candidate steps sampled from the
                                 Drafter per decoding step.

─────────────────────────────────────────────────────────────────────────────
ALGORITHMIC NOTES  (what this script DOES to the GP, not what elo_system.py
                    does to candidate steps)
─────────────────────────────────────────────────────────────────────────────
Step C UWO logit (inside elo_system.py, called per decoding step):
    logit_i = w_tournament * znorm((R_i-1500)/T)  +  w_blade * znorm(μ_i - λ·σ_i)
  Each COMPLETE term is Z-normalized independently across the N candidates
  (zero mean, unit std) BEFORE being scaled by its weight.  This ensures
  w_tournament and w_blade have genuinely comparable effects regardless of T.
  See WHY below.

Z-Normalization (ONLY applied here, to the GP surrogate's TARGET values):
  This script Z-normalises the collected *scalar objective* observations
  before feeding them into the Gaussian Process surrogate model:
      yn = (y - mean(y)) / std(y)
  Why? GP kernels implicitly assume the target function has unit variance.
  Without Z-normalisation the squared-exponential kernel's hyperparameters
  (especially the output scale) are poorly conditioned when the objective
  values span a very different range from the kernel's prior.  Z-normalisation
  re-centres and re-scales the targets to mean 0, std 1 — making the GP's
  default kernel priors well-matched and its Expected Improvement estimates
  numerically stable.  The EI threshold (0.01) is then expressed in the same
  standardised units as the GP's output.
  Note: this normalisation is ONLY applied to the y_obs fed to the GP; the
  raw objective values stored in search_state.json and all_observations.csv
  are always the original (un-normalised) tribunal scores.

─────────────────────────────────────────────────────────────────────────────
EXECUTION FLOW
─────────────────────────────────────────────────────────────────────────────
1.  Startup: load runs/bayes_search/search_state.json if it exists
    (crash-safe resume); otherwise build Round 0 = 1 fixed default config +
    (configs_per_round-1) Sobol space-filling configs.

2.  Per-config evaluation loop (strictly sequential, one config at a time):
      a. Launch benchmark_gsi_strategies_harmlessness.py as a subprocess
         with CUDA_VISIBLE_DEVICES=<gpu_id>.  Block until complete.
      b. Sleep 5 s to let VRAM drain.
      c. Launch vLLM judge server (Qwen2.5-32B-Instruct, bitsandbytes 4-bit)
         on the SAME gpu_id.  Poll /v1/models until ready (up to 15 min).
      d. Run tribunal.run_eval against the judge server.
      e. Kill the judge server.  Sleep 5 s.
      f. Read model_summary.csv, compute the scalar objective:
           objective = 0.25*(helpfulness+relevance+response_quality+refusal)
                     - 0.50*(toxicity+harmfulness)
      g. Append the record and IMMEDIATELY write both search_state.json
         (full state, resumable) and all_observations.csv (for inspection).
         Optionally push to Hugging Face Hub as an offsite backup.
      h. On generation or scoring failure: prompt the user to fix the issue
         and press Enter to retry (the same config stays at the front of the
         queue); Ctrl-C aborts entirely.

3.  End of each round:
      a. Fit a GP surrogate on all observations so far (both skopt and the
         built-in path Z-normalise y_obs before fitting).
      b. Propose the next batch of configs via Expected Improvement (EI)
         with a diversity penalty so the proposals spread across the space.
      c. If max(EI) < min_expected_improvement (default 0.01), STOP.
      d. Save checkpoint for the new round's configs.
      e. Generate intermediate plots (hp_effects, pareto frontier,
         convergence, correlation heatmap, GP partial dependence).
      f. Thermal cooldown: sleep for cooldown_seconds (default 3600 = 1 h)
         while printing a live countdown; user can press Enter to skip.

4.  On search completion (EI threshold met or Ctrl-C):
      Fit a final surrogate, generate final plots, and write best_config.json.

─────────────────────────────────────────────────────────────────────────────
FILE LAYOUT (all under <output_root>/)
─────────────────────────────────────────────────────────────────────────────
  runs/bayes_search/
    search_state.json          ← full resumable checkpoint (updated per config)
    logs/
      gen_<label>_gpu<id>.log  ← stdout/stderr from generation subprocess
      judge_gpu<id>_port<p>.log← vLLM judge server log
      tribunal_<label>_gpu<id>.log
    plots/
      all_observations.csv     ← one row per completed config evaluation
      best_config.json         ← best hyperparameter set found
      hp_effects.png           ← scatter: each HP vs each tribunal metric
      pareto_frontier.png      ← quality vs safety Pareto plot
      objective_vs_gsi_n.png
      convergence.png
      correlation_heatmap.png
      partial_dependence.png   ← GP uncertainty bands per HP
      tribunal_style/          ← per-group tribunal-style comparison plots
  tribunal/bayes_search/
    inputs/round<N>/<label>/   ← .jsonl files fed to tribunal
    eval_results/round<N>/<label>/ ← model_summary.csv, eval .csv

─────────────────────────────────────────────────────────────────────────────
OOM SAFETY
─────────────────────────────────────────────────────────────────────────────
  Generation (Drafter 3B + Blade 7B+LoRA + Base Verifier 7B) ≈ 34 GB VRAM.
  Judge (Qwen2.5-32B 4-bit) ≈ 20–24 GB VRAM.
  Combined ≈ 54–58 GB — too large for concurrent execution even on 128 GB
  unified-memory DGX when accounting for CUDA overhead and activations.
  This script ensures the generation process fully exits and VRAM is released
  (5 s sleep) BEFORE starting the judge server, and vice versa.

─────────────────────────────────────────────────────────────────────────────
KEY CLI FLAGS
─────────────────────────────────────────────────────────────────────────────
  --repo-root              Repo root (contains evaluation/, tribunal/, etc.)
  --configs-per-round      Configs per round (default 7)
  --gpu-id                 Physical GPU index (default 0)
  --num-prompts            Prompts per config evaluation (default 15)
  --max-tokens             Max generation tokens (default 512)
  --judge-model            vLLM judge model (default Qwen/Qwen2.5-32B-Instruct)
  --min-expected-improvement  EI stopping threshold (default 0.01)
  --cooldown-seconds       Thermal break between rounds in seconds (default 3600)
  --manual-resume          Pause before each config and wait for Enter
  --hf-repo-id             Optional HF Hub dataset repo for offsite backups
  --output-root            Root for all output dirs (default: repo-root)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import select
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("bayes_search_opt")

SEARCH_SPACE: Dict[str, Tuple[str, float, float, str, float]] = {
    "elo_temperature": ("--elo-temperature", 1.0,  40.0, "float", 15.0),
    "beta":            ("--beta",            0.01, 1.0,  "float", 0.1),
    "w_tournament":    ("--w-tournament",     0.0,  3.0,  "float", 1.0),
    "w_blade":         ("--w-blade",          0.0,  3.0,  "float", 1.0),
    "uwo_lambda":      ("--uwo-lambda",       0.0,  1.0,  "float", 0.5),
    "elo_rounds":      ("--elo-rounds",       2,    10,   "int",   6),
    "gsi_n":           ("--gsi-n",            3,    16,   "int",   8),
}
HP_NAMES = list(SEARCH_SPACE.keys())

FIXED_FLAGS = [
    "--strategies", "elo_swiss_mode_b",
    "--probabilistic",
    "--sigma-mode", "log_ratio_proxy",
    "--gsi-max-step-tokens", "80",
]

STRATEGY_NAME = "elo_swiss_mode_b"

OBJECTIVE_WEIGHTS = {
    "helpfulness": 0.25,
    "relevance": 0.25,
    "response_quality": 0.25,
    "refusal": 0.25,
    "toxicity": -0.5,
    "harmfulness": -0.5,
}

JUDGE_API_KEY = "EMPTY"


@dataclass
class HPConfig:
    cfg_id: str
    round_idx: int
    values: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        return np.array([self.values[n] for n in HP_NAMES], dtype=float)

    def cli_args(self) -> List[str]:
        args = []
        for name, (flag, _lo, _hi, dtype, _default) in SEARCH_SPACE.items():
            v = self.values[name]
            v = int(round(v)) if dtype == "int" else float(v)
            args += [flag, str(v)]
        return args

    def label(self) -> str:
        return f"r{self.round_idx}_{self.cfg_id}"


def sample_space_filling(n: int, seed: int = 0) -> List[Dict[str, float]]:
    dims = len(HP_NAMES)
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=dims, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        unit = sampler.random_base2(m=m)[:n]
    except Exception:
        rng = np.random.default_rng(seed)
        unit = np.zeros((n, dims))
        for d in range(dims):
            edges = np.linspace(0, 1, n + 1)
            u = edges[:-1] + rng.random(n) * (edges[1] - edges[0])
            rng.shuffle(u)
            unit[:, d] = u

    samples = []
    for row in unit:
        vals = {}
        for i, name in enumerate(HP_NAMES):
            _, lo, hi, dtype, _ = SEARCH_SPACE[name]
            v = lo + row[i] * (hi - lo)
            vals[name] = int(round(v)) if dtype == "int" else float(v)
        samples.append(vals)
    return samples


def default_config() -> Dict[str, float]:
    return {name: SEARCH_SPACE[name][4] for name in HP_NAMES}


def build_round0_configs(num_configs: int, seed: int) -> List[HPConfig]:
    configs = [HPConfig(cfg_id="cfg0", round_idx=0, values=default_config())]
    fill = sample_space_filling(num_configs - 1, seed=seed)
    for i, vals in enumerate(fill, start=1):
        configs.append(HPConfig(cfg_id=f"cfg{i}", round_idx=0, values=vals))
    return configs


def convert_json_to_jsonl(results_dir: str, tribunal_inputs_dir: str, model_name: str) -> Optional[str]:
    src = os.path.join(results_dir, f"{STRATEGY_NAME}_results.json")
    if not os.path.exists(src):
        logger.error("No results file at %s -- generation likely failed.", src)
        return None
    with open(src) as f:
        data = json.load(f)
    responses = data.get("responses", [])
    if not responses:
        logger.warning("%s has no responses.", src)
        return None

    os.makedirs(tribunal_inputs_dir, exist_ok=True)
    dst = os.path.join(tribunal_inputs_dir, f"{model_name}.jsonl")
    written = 0
    with open(dst, "w", encoding="utf-8") as out:
        for resp in responses:
            if resp.get("error") or not resp.get("generated", "").strip():
                continue
            record = {
                "id": resp["prompt_idx"],
                "prompt": resp["prompt"].strip(),
                "response": resp["generated"].strip(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    logger.info("Converted %s -> %s (%d records)", src, dst, written)
    return dst


def start_judge_server(gpu_id: int, port: int, repo_root: str, log_dir: str,
                        judge_model: str, gpu_mem_util: float = 0.90) -> subprocess.Popen:
    cmd = [
        sys.executable, "-c",
        "import transformers; "
        "transformers.tokenization_utils_base.PreTrainedTokenizerBase.all_special_tokens_extended = "
        "property(lambda self: self.all_special_tokens); "
        "import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')",
        "--model", judge_model,
        "--quantization", "bitsandbytes",
        "--load-format", "bitsandbytes",
        "--dtype", "half",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--port", str(port),
        "--api-key", JUDGE_API_KEY,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"judge_gpu{gpu_id}_port{port}.log")
    logger.info("[GPU %d] starting vLLM judge server on port %d (%s)", gpu_id, port, judge_model)
    f = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f
    return proc


def wait_for_server(port: int, timeout_s: int = 900, poll_s: int = 5) -> bool:
    import urllib.request
    import urllib.error
    url = f"http://localhost:{port}/v1/models"
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                logger.error(
                    "Judge server on port %d rejected auth (HTTP %d). "
                    "JUDGE_API_KEY (%r) must match the --api-key passed to "
                    "vLLM in start_judge_server().", port, e.code, JUDGE_API_KEY,
                )
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(poll_s)
    return False


def stop_judge_server(proc: subprocess.Popen, grace_s: int = 20):
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    try:
        proc._log_file.close()
    except Exception:
        pass


def read_metrics(results_dir: str, model_label: str) -> Optional[Dict[str, float]]:
    import pandas as pd
    summary_path = os.path.join(results_dir, "model_summary.csv")
    if not os.path.exists(summary_path):
        logger.error("No model_summary.csv at %s", summary_path)
        return None
    df = pd.read_csv(summary_path)
    row = df[df["model"] == model_label]
    if row.empty:
        logger.error("model_label %s not found in %s (rows: %s)",
                      model_label, summary_path, df["model"].tolist())
        return None
    row = row.iloc[0]
    metrics = {}
    for m in ["response_quality", "relevance", "helpfulness", "toxicity", "harmfulness", "refusal"]:
        if m in row:
            metrics[m] = float(row[m])
    return metrics


def scalar_objective(metrics: Dict[str, float]) -> float:
    return sum(OBJECTIVE_WEIGHTS[m] * metrics[m] for m in OBJECTIVE_WEIGHTS if m in metrics)


def _normalize(X: np.ndarray) -> np.ndarray:
    lo = np.array([SEARCH_SPACE[n][1] for n in HP_NAMES])
    hi = np.array([SEARCH_SPACE[n][2] for n in HP_NAMES])
    return (X - lo) / (hi - lo)


def _denormalize_point(x: np.ndarray) -> Dict[str, float]:
    lo = np.array([SEARCH_SPACE[n][1] for n in HP_NAMES])
    hi = np.array([SEARCH_SPACE[n][2] for n in HP_NAMES])
    raw = lo + x * (hi - lo)
    vals = {}
    for i, name in enumerate(HP_NAMES):
        dtype = SEARCH_SPACE[name][3]
        vals[name] = int(round(raw[i])) if dtype == "int" else float(raw[i])
    return vals


def propose_next_batch_skopt(X_obs: np.ndarray, y_obs: np.ndarray, n_proposals: int,
                              seed: int) -> Tuple[List[Dict[str, float]], object]:
    from skopt import Optimizer
    from skopt.space import Real, Integer

    dims = []
    for name in HP_NAMES:
        _, lo, hi, dtype, _ = SEARCH_SPACE[name]
        dims.append(Integer(int(lo), int(hi), name=name) if dtype == "int" else Real(lo, hi, name=name))

    # Apply Z-normalization to the objectives for stable surrogate modeling
    y_mean = y_obs.mean()
    y_std = y_obs.std() if y_obs.std() > 1e-8 else 1.0
    yn = (y_obs - y_mean) / y_std

    opt = Optimizer(dims, base_estimator="GP", acq_func="EI", random_state=seed,
                     n_initial_points=0)
    X_list = [[float(v) if SEARCH_SPACE[n][3] == "float" else int(round(v))
               for n, v in zip(HP_NAMES, row)] for row in X_obs]
    
    # skopt minimizes by default, so tell it the negative of the normalized objective
    opt.tell(X_list, (-yn).tolist())

    proposals = []
    for _ in range(n_proposals):
        x = opt.ask()
        vals = {name: (float(v) if SEARCH_SPACE[name][3] == "float" else int(round(v)))
                for name, v in zip(HP_NAMES, x)}
        proposals.append(vals)
        pred = opt.models[-1].predict([x])[0] if opt.models else 0.0
        opt.tell(x, pred)
    return proposals, opt


def propose_next_batch_builtin(X_obs: np.ndarray, y_obs: np.ndarray, n_proposals: int,
                                seed: int, n_candidates: int = 4000):
    from scipy.spatial.distance import cdist
    from scipy.linalg import cho_factor, cho_solve
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    Xn = _normalize(X_obs)
    y = y_obs.copy()
    y_mean, y_std = y.mean(), (y.std() + 1e-8)
    yn = (y - y_mean) / y_std

    length_scale = 0.3
    noise = 1e-4

    def kernel(A, B):
        d2 = cdist(A, B, "sqeuclidean")
        return np.exp(-d2 / (2 * length_scale ** 2))

    K = kernel(Xn, Xn) + noise * np.eye(len(Xn))
    c, low = cho_factor(K, lower=True)
    alpha = cho_solve((c, low), yn)

    def gp_predict(Xs):
        Ks = kernel(Xs, Xn)
        mu = Ks @ alpha
        v = cho_solve((c, low), Ks.T)
        var = 1.0 - np.sum(Ks.T * v, axis=0)
        var = np.clip(var, 1e-9, None)
        return mu * y_std + y_mean, np.sqrt(var) * y_std

    Xc = rng.random((n_candidates, len(HP_NAMES)))
    mu, sigma = gp_predict(Xc)
    best_y = y.max()
    z = (mu - best_y) / sigma
    ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)

    chosen = []
    chosen_idx = []
    penalty = np.zeros(n_candidates)
    for _ in range(n_proposals):
        score = ei - penalty
        idx = int(np.argmax(score))
        chosen_idx.append(idx)
        chosen.append(Xc[idx])
        d = np.linalg.norm(Xc - Xc[idx], axis=1)
        penalty += np.exp(-d ** 2 / (2 * 0.15 ** 2)) * (ei.max() + 1e-6)

    proposals = [_denormalize_point(x) for x in chosen]
    surrogate = {"predict": gp_predict, "x_mean": y_mean, "x_std": y_std}
    return proposals, surrogate


def propose_next_batch(X_obs: np.ndarray, y_obs: np.ndarray, n_proposals: int, seed: int
                        ) -> Tuple[List[Dict[str, float]], object, List[float]]:
    try:
        import skopt  # noqa: F401
        logger.info("Using scikit-optimize for Bayesian optimization.")
        proposals, surrogate = propose_next_batch_skopt(X_obs, y_obs, n_proposals, seed)
    except ImportError:
        logger.warning("scikit-optimize not installed -- using built-in GP+EI fallback.")
        proposals, surrogate = propose_next_batch_builtin(X_obs, y_obs, n_proposals, seed)

    ei_values = estimate_expected_improvement(X_obs, y_obs, proposals)
    return proposals, surrogate, ei_values


def estimate_expected_improvement(X_obs: np.ndarray, y_obs: np.ndarray,
                                   proposals: List[Dict[str, float]]) -> List[float]:
    from scipy.spatial.distance import cdist
    from scipy.linalg import cho_factor, cho_solve
    from scipy.stats import norm

    Xn = _normalize(X_obs)
    y = y_obs.copy()
    y_mean, y_std = y.mean(), (y.std() + 1e-8)
    yn = (y - y_mean) / y_std

    length_scale = 0.3
    noise = 1e-4

    def kernel(A, B):
        d2 = cdist(A, B, "sqeuclidean")
        return np.exp(-d2 / (2 * length_scale ** 2))

    K = kernel(Xn, Xn) + noise * np.eye(len(Xn))
    c, low = cho_factor(K, lower=True)
    alpha = cho_solve((c, low), yn)

    Xp = _normalize(np.array([[pt[n] for n in HP_NAMES] for pt in proposals], dtype=float))
    Ks = kernel(Xp, Xn)
    mu = (Ks @ alpha) * y_std + y_mean
    v = cho_solve((c, low), Ks.T)
    var = np.clip(1.0 - np.sum(Ks.T * v, axis=0), 1e-9, None)
    sigma = np.sqrt(var) * y_std

    best_y = y.max()
    z = (mu - best_y) / sigma
    ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
    return [float(v) for v in ei]


def save_all_observations_csv(csv_file: str, evaluated_configs: List[dict]) -> None:
    """Write all evaluated configs to a CSV for quick inspection without parsing JSON."""
    try:
        import pandas as pd
        pd.DataFrame(evaluated_configs).to_csv(csv_file, index=False)
    except Exception as e:
        logger.warning("Could not write all_observations.csv: %s", e)


def save_search_state(state_file: str, round_idx: int, configs_to_generate: List[HPConfig], configs_to_judge: List[HPConfig], evaluated_configs: List[dict], seed: int):
    state = {
        "round_idx": round_idx,
        "configs_to_generate": [asdict(c) for c in configs_to_generate],
        "configs_to_judge": [asdict(c) for c in configs_to_judge],
        "evaluated_configs": evaluated_configs,
        "seed": seed
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved checkpoint state to {state_file}")


def load_search_state(state_file: str) -> Optional[Tuple[int, List[HPConfig], List[HPConfig], List[dict], int]]:
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file) as f:
            state = json.load(f)
        
        configs_to_gen = [
            HPConfig(cfg_id=c["cfg_id"], round_idx=c["round_idx"], values=c["values"])
            for c in state.get("configs_to_generate", [])
        ]
        
        # Backwards compatibility fallback if older structure is used
        if "configs_to_evaluate" in state and not configs_to_gen and not state.get("configs_to_judge"):
            configs_to_gen = [
                HPConfig(cfg_id=c["cfg_id"], round_idx=c["round_idx"], values=c["values"])
                for c in state["configs_to_evaluate"]
            ]
            
        configs_to_jdg = [
            HPConfig(cfg_id=c["cfg_id"], round_idx=c["round_idx"], values=c["values"])
            for c in state.get("configs_to_judge", [])
        ]
        return state["round_idx"], configs_to_gen, configs_to_jdg, state["evaluated_configs"], state["seed"]
    except Exception as e:
        logger.error(f"Error loading search state: {e}")
        return None


def cooldown_break(seconds: int = 3600):
    logger.info("=" * 80)
    logger.info(f"STARTING THERMAL COOLDOWN BREAK OF {seconds // 60} MINUTES AFTER ROUND COMPLETION.")
    logger.info("This is to let your single GPU cool down and prevent thermal throttling.")
    logger.info("You can press ENTER at any time in this terminal to skip the cooldown and proceed.")
    logger.info("=" * 80)

    interval = 10
    elapsed = 0
    while elapsed < seconds:
        remaining = seconds - elapsed
        mins, secs = divmod(remaining, 60)
        print(f"\rTime remaining in cooldown: {mins:02d}:{secs:02d}...", end="", flush=True)

        rlist, _, _ = select.select([sys.stdin], [], [], interval)
        if rlist:
            sys.stdin.readline()  # Consume key press
            print("\nCooldown skipped by user request.")
            return
        elapsed += interval
    print("\nCooldown completed.")


def stage_combined_tribunal_dirs(records: List[dict], tribunal_root: str, group_name: str) -> Optional[Tuple[str, str]]:
    import pandas as pd

    combined_inputs_dir = os.path.join(tribunal_root, "inputs", "_combined", group_name)
    combined_results_dir = os.path.join(tribunal_root, "eval_results", "_combined", group_name)
    shutil.rmtree(combined_inputs_dir, ignore_errors=True)
    shutil.rmtree(combined_results_dir, ignore_errors=True)
    os.makedirs(combined_inputs_dir, exist_ok=True)
    os.makedirs(combined_results_dir, exist_ok=True)

    summary_rows = []
    combined_rows = []
    staged_any = False

    for rec in records:
        cfg_label = rec["cfg_label"]
        round_idx = rec["round"]
        per_cfg_inputs_dir = os.path.join(tribunal_root, "inputs", f"round{round_idx}", cfg_label)
        per_cfg_results_dir = os.path.join(tribunal_root, "eval_results", f"round{round_idx}", cfg_label)

        src_jsonl = os.path.join(per_cfg_inputs_dir, f"{cfg_label}.jsonl")
        src_eval_csv = os.path.join(per_cfg_results_dir, f"{cfg_label}_eval.csv")
        src_summary_csv = os.path.join(per_cfg_results_dir, "model_summary.csv")

        if not (os.path.exists(src_jsonl) and os.path.exists(src_eval_csv) and os.path.exists(src_summary_csv)):
            logger.warning("Skipping %s in combined tribunal plots -- missing files.", cfg_label)
            continue

        shutil.copy(src_jsonl, os.path.join(combined_inputs_dir, f"{cfg_label}.jsonl"))
        shutil.copy(src_eval_csv, os.path.join(combined_results_dir, f"{cfg_label}_eval.csv"))
        summary_rows.append(pd.read_csv(src_summary_csv))
        combined_rows.append(pd.read_csv(src_eval_csv))
        staged_any = True

    if not staged_any:
        logger.error("No configs had complete tribunal output for group '%s'", group_name)
        return None

    pd.concat(summary_rows, ignore_index=True).to_csv(
        os.path.join(combined_results_dir, "model_summary.csv"), index=False,
    )
    pd.concat(combined_rows, ignore_index=True).to_csv(
        os.path.join(combined_results_dir, "combined_results.csv"), index=False,
    )
    return combined_inputs_dir, combined_results_dir


def make_tribunal_comparison_plots(records: List[dict], tribunal_root: str, repo_root: str,
                                    plot_dir: str, group_name: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from evaluation.prepare_tribunal_eval import plot as tribunal_style_plot

    staged = stage_combined_tribunal_dirs(records, tribunal_root, group_name)
    if staged is None:
        return
    combined_inputs_dir, combined_results_dir = staged

    out_dir = os.path.join(plot_dir, "tribunal_style", group_name)
    tribunal_style_plot(
        results_dir=combined_results_dir,
        plot_dir=out_dir,
        inputs_dir=combined_inputs_dir,
    )
    logger.info("Tribunal comparison plots for '%s' written to %s", group_name, out_dir)


def make_plots(records: List[dict], plot_dir: str, surrogate=None):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(records)
    if df.empty:
        logger.error("No records to plot.")
        return
    df.to_csv(os.path.join(plot_dir, "all_observations.csv"), index=False)

    metrics = [m for m in ["response_quality", "relevance", "helpfulness",
                            "toxicity", "harmfulness", "refusal"] if m in df.columns]

    fig, axes = plt.subplots(len(HP_NAMES), len(metrics),
                              figsize=(3.2 * len(metrics), 2.6 * len(HP_NAMES)),
                              squeeze=False)
    for i, hp in enumerate(HP_NAMES):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            # Map round to colors dynamically
            unique_rounds = sorted(df["round"].unique())
            color_map = {r: plt.cm.tab10(r % 10) for r in unique_rounds}
            colors = df["round"].map(color_map)
            ax.scatter(df[hp], df[m], c=colors, s=40, edgecolors="white", linewidths=0.5)
            if len(df) >= 3:
                z = np.polyfit(df[hp], df[m], 1)
                xs = np.linspace(df[hp].min(), df[hp].max(), 50)
                ax.plot(xs, np.polyval(z, xs), "--", color="gray", linewidth=1)
            if i == len(HP_NAMES) - 1:
                ax.set_xlabel(m, fontsize=8)
            if j == 0:
                ax.set_ylabel(hp, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Hyperparameter effect on each tribunal metric", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(os.path.join(plot_dir, "hp_effects.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if all(m in df.columns for m in ["helpfulness", "relevance", "response_quality", "toxicity", "harmfulness", "refusal"]):
        df["quality_axis"] = df[["response_quality", "relevance", "helpfulness"]].mean(axis=1)
        df["safety_axis"] = 1 - df[["toxicity", "harmfulness"]].mean(axis=1)

        pts = df[["quality_axis", "safety_axis"]].values
        is_pareto = np.ones(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                    is_pareto[i] = False
                    break

        fig, ax = plt.subplots(figsize=(7, 6))
        unique_rounds = sorted(df["round"].unique())
        color_map = {r: plt.cm.tab10(r % 10) for r in unique_rounds}
        colors = df["round"].map(color_map)
        ax.scatter(df.loc[~is_pareto, "quality_axis"], df.loc[~is_pareto, "safety_axis"],
                   c=colors[~is_pareto], s=90, alpha=0.6, edgecolors="white", label="dominated")
        ax.scatter(df.loc[is_pareto, "quality_axis"], df.loc[is_pareto, "safety_axis"],
                   c=colors[is_pareto], s=160, marker="*", edgecolors="black",
                   linewidths=1, label="Pareto-optimal", zorder=5)
        front = df.loc[is_pareto].sort_values("quality_axis")
        ax.plot(front["quality_axis"], front["safety_axis"], "k--", linewidth=1, alpha=0.5)
        for _, row in df.iterrows():
            ax.annotate(row["cfg_label"], (row["quality_axis"], row["safety_axis"]),
                        fontsize=6, textcoords="offset points", xytext=(4, 3))
        ax.set_xlabel("Quality axis (mean of quality, relevance, helpfulness)")
        ax.set_ylabel("Safety axis (1 - mean of toxicity, harmfulness)")
        ax.set_title("Pareto Frontier — Quality vs Safety")
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "pareto_frontier.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "gsi_n" in df.columns and "objective" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        unique_rounds = sorted(df["round"].unique())
        for rnd in unique_rounds:
            sub = df[df["round"] == rnd]
            if not sub.empty:
                ax.scatter(sub["gsi_n"], sub["objective"], s=70, label=f"round {rnd}", edgecolors="white")
        if len(df) >= 3:
            z = np.polyfit(df["gsi_n"], df["objective"], min(2, len(df) - 1))
            xs = np.linspace(df["gsi_n"].min(), df["gsi_n"].max(), 100)
            ax.plot(xs, np.polyval(z, xs), "--", color="gray", label="trend")
        ax.set_xlabel("Number of candidates (gsi_n)")
        ax.set_ylabel("Objective")
        ax.set_title("Objective vs Number of Candidates")
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "objective_vs_gsi_n.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "objective" in df.columns:
        df_sorted = df.sort_values(["round", "cfg_label"]).reset_index(drop=True)
        best_so_far = df_sorted["objective"].cummax()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(df_sorted) + 1), best_so_far, marker="o", color="#2F6690")
        ax.set_xlabel("Configuration index (evaluation order)")
        ax.set_ylabel("Best objective so far")
        ax.set_title("Bayesian Optimization Convergence")
        fig.savefig(os.path.join(plot_dir, "convergence.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    cols = HP_NAMES + metrics + (["objective"] if "objective" in df.columns else [])
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 1.1 * len(cols) + 1))
    im = ax.imshow(corr.loc[HP_NAMES, metrics + (["objective"] if "objective" in df.columns else [])],
                    cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics) + (1 if "objective" in df.columns else 0)))
    ax.set_xticklabels(metrics + (["objective"] if "objective" in df.columns else []),
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(HP_NAMES)))
    ax.set_yticklabels(HP_NAMES, fontsize=8)
    for i in range(len(HP_NAMES)):
        for j in range(len(metrics) + (1 if "objective" in df.columns else 0)):
            val = corr.loc[HP_NAMES, metrics + (["objective"] if "objective" in df.columns else [])].iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(val) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Hyperparameter <-> Metric Correlation")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if surrogate is not None and isinstance(surrogate, dict) and "predict" in surrogate:
        fig, axes = plt.subplots(1, len(HP_NAMES), figsize=(3.2 * len(HP_NAMES), 3.2))
        for i, hp in enumerate(HP_NAMES):
            ax = axes[i]
            grid = np.linspace(0, 1, 60)
            Xs = np.tile(0.5, (60, len(HP_NAMES)))
            Xs[:, i] = grid
            mu, sigma = surrogate["predict"](Xs)
            lo, hi = SEARCH_SPACE[hp][1], SEARCH_SPACE[hp][2]
            xs_raw = lo + grid * (hi - lo)
            ax.plot(xs_raw, mu, color="#2F6690")
            ax.fill_between(xs_raw, mu - sigma, mu + sigma, alpha=0.2, color="#2F6690")
            ax.set_xlabel(hp, fontsize=8)
            if i == 0:
                ax.set_ylabel("GP-predicted objective", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle("GP Partial Dependence (other hyperparameters held at range midpoint)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "partial_dependence.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def upload_snapshot_to_hf(local_dir: str, repo_id: str, token: Optional[str],
                           commit_message: str) -> None:
    if not repo_id:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed -- skipping snapshot upload.")
        return

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=local_dir,
            token=token,
            commit_message=commit_message,
        )
        logger.info("Uploaded snapshot of %s -> hf.co/datasets/%s", local_dir, repo_id)
    except Exception as e:
        logger.warning("HF snapshot upload failed: %s", e)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=".", help="Path to repo root.")
    p.add_argument("--configs-per-round", type=int, default=7,
                    help="Number of configs evaluated in each round.")
    p.add_argument("--gpu-id", type=int, default=0, help="GPU physical ID to pin all runs to sequentially.")
    p.add_argument("--num-prompts", type=int, default=15, help="Number of prompts evaluated per config.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=None)
    p.add_argument("--extra-flag", action="append", default=[], help="Extra flag for generation.")
    p.add_argument("--min-expected-improvement", type=float, default=0.01,
                    help="EI threshold to stop search early.")
    p.add_argument("--cooldown-seconds", type=int, default=3600,
                    help="Pause time in seconds after each round completion (default 1 hour).")
    p.add_argument("--manual-resume", action="store_true",
                    help="Require user input in terminal to start the next config/evaluation.")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_root = args.output_root or repo_root
    log_dir = os.path.join(output_root, "runs", "bayes_search", "logs")
    plot_dir = os.path.join(output_root, "runs", "bayes_search", "plots")
    tribunal_root = os.path.join(output_root, "tribunal", "bayes_search")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    state_file = os.path.join(output_root, "runs", "bayes_search", "search_state.json")
    csv_file = os.path.join(plot_dir, "all_observations.csv")

    # Resume from checkpoint or initialize
    loaded = load_search_state(state_file)
    if loaded is not None:
        round_idx, configs_to_generate, configs_to_judge, evaluated_configs, seed = loaded
        logger.info(
            "Resuming search. Round %d | to generate: %d | to judge: %d | evaluated: %d",
            round_idx, len(configs_to_generate), len(configs_to_judge), len(evaluated_configs)
        )
    else:
        round_idx = 0
        seed = args.seed
        configs_to_generate = build_round0_configs(args.configs_per_round, seed=seed)
        configs_to_judge = []
        evaluated_configs = []
        logger.info("Initializing from scratch. Round 0 configs: %d", len(configs_to_generate))
        save_search_state(state_file, round_idx, configs_to_generate, configs_to_judge, evaluated_configs, seed)

    while True:
        # ── PHASE 1: Sequential generation (NO judge running) ──────────────────
        # Generate responses for every remaining config in this round before
        # touching the judge server at all.  This keeps VRAM usage low.
        while configs_to_generate:
            cfg = configs_to_generate[0]

            if args.manual_resume:
                print(f"\n[GENERATION] Ready for {cfg.label()} ({cfg.values}). Press Enter...")
                input()

            logger.info("=" * 80)
            logger.info("GENERATION | Round %d | config %s", round_idx, cfg.label())
            logger.info("Parameters: %s", cfg.values)
            logger.info("=" * 80)

            out_dir = os.path.join(repo_root, "runs", "bayes_search", f"round{cfg.round_idx}", cfg.label())
            os.makedirs(out_dir, exist_ok=True)
            script = os.path.join(repo_root, "evaluation", "benchmark_gsi_strategies_harmlessness.py")

            cmd = [
                sys.executable, script,
                "--num-prompts", str(args.num_prompts),
                "--max-tokens", str(args.max_tokens),
                "--output-dir", out_dir,
            ] + FIXED_FLAGS + cfg.cli_args() + args.extra_flag

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

            gen_log = os.path.join(log_dir, f"gen_{cfg.label()}_gpu{args.gpu_id}.log")
            logger.info("Logging generation to %s", gen_log)

            with open(gen_log, "w") as f:
                ret = subprocess.run(cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT)

            if ret.returncode != 0:
                logger.error("Generation FAILED (code %d) for %s. Log: %s", ret.returncode, cfg.label(), gen_log)
                print(f"\nGeneration failed for {cfg.label()}. Fix the issue then press Enter to retry, or Ctrl+C to abort.")
                input()
                continue  # retry same config

            logger.info("Generation done for %s. Sleeping 15 s to drain VRAM...", cfg.label())
            time.sleep(15)

            # Move config from generate-queue to judge-queue and checkpoint
            configs_to_generate.pop(0)
            configs_to_judge.append(cfg)
            save_search_state(state_file, round_idx, configs_to_generate, configs_to_judge, evaluated_configs, seed)
            logger.info("Progress: %d left to generate, %d queued for judging.", len(configs_to_generate), len(configs_to_judge))

        # ── PHASE 2: Judging (one server for the whole round) ──────────────────
        # Start the vLLM judge server ONCE, score every queued config, then
        # shut it down.  This avoids repeated cold-start overhead (2–3 min each).
        if configs_to_judge:
            logger.info("=" * 80)
            logger.info("JUDGING PHASE | Round %d | %d configs to judge.", round_idx, len(configs_to_judge))
            logger.info("Sleeping 60 seconds to allow GPU driver to fully release and clean VRAM...")
            time.sleep(60)
            logger.info("Starting judge server (this takes ~2-3 min)...")
            logger.info("=" * 80)

            port = 8000 + args.gpu_id
            judge_proc = start_judge_server(args.gpu_id, port, repo_root, log_dir, args.judge_model)

            judge_crashed = False
            try:
                ready = wait_for_server(port, timeout_s=900)
                if not ready:
                    raise RuntimeError("Judge server failed to become ready within 15 minutes.")

                while configs_to_judge:
                    cfg = configs_to_judge[0]

                    if args.manual_resume:
                        print(f"\n[JUDGING] Ready to score {cfg.label()}. Press Enter...")
                        input()

                    logger.info("Judging config %s...", cfg.label())
                    out_dir = os.path.join(repo_root, "runs", "bayes_search", f"round{cfg.round_idx}", cfg.label())

                    # Convert generation output to jsonl for tribunal
                    tribunal_inputs_dir = os.path.join(tribunal_root, "inputs", f"round{cfg.round_idx}", cfg.label())
                    jsonl_path = convert_json_to_jsonl(out_dir, tribunal_inputs_dir, model_name=cfg.label())
                    if jsonl_path is None:
                        raise RuntimeError(f"JSONL conversion failed for {cfg.label()}.")

                    results_dir = os.path.join(tribunal_root, "eval_results", f"round{cfg.round_idx}", cfg.label())
                    os.makedirs(results_dir, exist_ok=True)

                    tmp_input_dir = os.path.join(results_dir, "_tribunal_input_tmp")
                    os.makedirs(tmp_input_dir, exist_ok=True)
                    shutil.copy(jsonl_path, os.path.join(tmp_input_dir, os.path.basename(jsonl_path)))

                    cmd_eval = [
                        sys.executable, "-m", "tribunal.run_eval",
                        "--input", tmp_input_dir,
                        "--output", results_dir,
                        "--judge-url", f"http://localhost:{port}/v1",
                    ]
                    eval_log = os.path.join(log_dir, f"tribunal_{cfg.label()}_gpu{args.gpu_id}.log")
                    logger.info("Tribunal scoring → %s", eval_log)

                    with open(eval_log, "w") as f:
                        ret_eval = subprocess.run(
                            cmd_eval, cwd=os.path.join(repo_root, "tribunal"),
                            env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT
                        )

                    shutil.rmtree(tmp_input_dir, ignore_errors=True)

                    if ret_eval.returncode != 0:
                        raise RuntimeError(f"tribunal.run_eval failed (code {ret_eval.returncode}) for {cfg.label()}.")

                    metrics = read_metrics(results_dir, cfg.label())
                    if metrics is None:
                        raise RuntimeError(f"Could not read model_summary.csv for {cfg.label()}.")

                    obj = scalar_objective(metrics)
                    metrics_rec = {
                        "cfg_label": cfg.label(),
                        "round": cfg.round_idx,
                        **cfg.values,
                        **metrics,
                        "objective": obj,
                    }

                    # Commit result and save checkpoint immediately
                    evaluated_configs.append(metrics_rec)
                    configs_to_judge.pop(0)
                    save_search_state(state_file, round_idx, configs_to_generate, configs_to_judge, evaluated_configs, seed)
                    save_all_observations_csv(csv_file, evaluated_configs)
                    logger.info("Scored %s → objective=%.4f | %d configs left to judge.", cfg.label(), obj, len(configs_to_judge))

                    if args.hf_repo_id:
                        upload_snapshot_to_hf(
                            local_dir=os.path.join(output_root, "runs", "bayes_search"),
                            repo_id=args.hf_repo_id, token=args.hf_token,
                            commit_message=f"Round {round_idx} judged {cfg.label()}"
                        )

            except Exception as e:
                logger.error("Judge phase failed: %s", e)
                judge_crashed = True
                print("\nJudge server or scoring crashed. Fix the issue, then press Enter to restart the judge and retry, or Ctrl+C.")
                input()
                # Fall through to finally, then retry outer while True loop
            finally:
                logger.info("Stopping judge server...")
                stop_judge_server(judge_proc)
                logger.info("Judge stopped. Sleeping 15 s to drain VRAM...")
                time.sleep(15)

            if judge_crashed:
                continue  # restart from top of while True (re-enter judging phase)

        # ── ROUND COMPLETE ──────────────────────────────────────────────────────
        logger.info("=" * 80)
        logger.info("Round %d complete. Total evaluated configs: %d", round_idx, len(evaluated_configs))
        logger.info("=" * 80)

        X_obs = np.array([[r[n] for n in HP_NAMES] for r in evaluated_configs], dtype=float)
        y_obs = np.array([r["objective"] for r in evaluated_configs], dtype=float)

        if len(evaluated_configs) < 2:
            logger.error("Fewer than 2 observations. Cannot fit GP. Aborting.")
            break

        logger.info("Fitting GP surrogate and computing Expected Improvement...")
        proposals, surrogate, ei_values = propose_next_batch(X_obs, y_obs, args.configs_per_round, seed=seed + round_idx + 1)
        max_ei = max(ei_values) if ei_values else 0.0
        logger.info("Proposed EI values: %s  (max EI: %.4f)", ei_values, max_ei)

        if max_ei < args.min_expected_improvement:
            logger.info("=" * 80)
            logger.info("Max EI (%.4f) < threshold (%.4f). Search converged.", max_ei, args.min_expected_improvement)
            logger.info("=" * 80)
            break

        # Build next round
        round_idx += 1
        configs_to_generate = [
            HPConfig(cfg_id=f"cfg{i}", round_idx=round_idx, values=v)
            for i, v in enumerate(proposals)
        ]
        configs_to_judge = []
        logger.info("Round %d configs proposed: %s", round_idx, [c.values for c in configs_to_generate])

        save_search_state(state_file, round_idx, configs_to_generate, configs_to_judge, evaluated_configs, seed)

        try:
            make_plots(evaluated_configs, plot_dir, surrogate=surrogate)
            make_tribunal_comparison_plots(evaluated_configs, tribunal_root, repo_root, plot_dir, "running_summary")
        except Exception as e:
            logger.warning("Plot generation failed (non-fatal): %s", e)

        # 1-hour thermal cooldown between rounds
        cooldown_break(args.cooldown_seconds)

    # Finalize search
    best = max(evaluated_configs, key=lambda r: r["objective"])
    logger.info("=" * 80)
    logger.info("SEARCH COMPLETE.")
    logger.info("Best configuration found: %s", best)
    logger.info("=" * 80)

    with open(os.path.join(plot_dir, "best_config.json"), "w") as f:
        json.dump(best, f, indent=2)

    try:
        # Fit final built-in surrogate to render partial dependence plots
        X_obs = np.array([[r[n] for n in HP_NAMES] for r in evaluated_configs], dtype=float)
        y_obs = np.array([r["objective"] for r in evaluated_configs], dtype=float)
        _, surrogate_final = propose_next_batch_builtin(X_obs, y_obs, n_proposals=1, seed=seed)
        make_plots(evaluated_configs, plot_dir, surrogate=surrogate_final)
        make_tribunal_comparison_plots(evaluated_configs, tribunal_root, repo_root, plot_dir, "final")
    except Exception as e:
        logger.warning("Error creating final plots: %s", e)


if __name__ == "__main__":
    main()
