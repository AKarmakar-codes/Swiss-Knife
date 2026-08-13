"""
Bayesian Hyperparameter Search for Swiss-Knife ``elo_swiss_mode_b`` GSI Strategy
==================================================================================
Multi-GPU parallel search (judge-per-GPU). Designed for a pool of independent
GPUs (e.g. 8x A6000 48GB) where each GPU is assigned its own config from the
round's queue and runs that config's FULL cycle (generation, then its own
private judge server, then scoring) to completion before picking up the next
config. Generation and judging still never overlap in VRAM WITHIN a single
GPU's worker (same OOM-safety rule as before), but different GPUs run their
own generate->judge->score cycles fully concurrently with each other.

This preserves the original single-GPU sequential codepath as the special
case of a worker pool of size 1 (pass a single --gpu-ids value to keep the
old behavior); the underlying per-config logic (subprocess commands, judge
server lifecycle, scalar_objective, checkpointing) is unchanged.

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETERS BEING SEARCHED  (6-dimensional search space)
─────────────────────────────────────────────────────────────────────────────
  elo_temperature  [1.0,  40.0]  Temperature T in the UWO logit: (R_i-1500)/T.
                                 Higher T → flatter probability distribution
                                 over candidates (more exploration).
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

  NOTE — beta (DPO regularization strength) is FIXED at 0.1 and NOT searched.
  Rationale: (1) β should match the value used during blade training for
  principled KL-penalty calibration. (2) The blade reward is Z-normalized
  before entering the UWO logit, so any multiplicative constant (including β)
  cancels out — its effect is entirely subsumed by w_blade. Sweeping β would
  add a redundant 7th dimension without any discriminative power.

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

  FIX: `estimate_expected_improvement` (used for the round-end convergence
  check against --min-expected-improvement) now computes EI entirely in
  this normalised space -- mu, sigma, and best-so-far are all kept in
  Z-scored units and never converted back to raw objective units before
  the EI formula is applied. Previously the mean/std were denormalised
  back to raw units first, which made EI scale with the raw objective's
  standard deviation and could trigger a false "converged" stop when raw
  scores were tightly clustered (e.g. early rounds), independent of
  whether the GP had actually run out of promising, uncertain regions to
  explore.

GP length-scale fitting (ARD, marginal-likelihood optimized):
  skopt's ``Optimizer(..., base_estimator="GP")`` path constructs a
  scikit-learn GaussianProcessRegressor under the hood. Left at its library
  default this already uses a Matern kernel with an independent length-scale
  PER DIMENSION (automatic relevance determination, "ARD") and fits those
  length-scales (plus the output/noise scale) by maximizing the GP marginal
  log-likelihood via L-BFGS restarts -- it does NOT use a single fixed
  length-scale. To make this explicit and to pin it down regardless of skopt
  version/defaults drifting, `propose_next_batch_skopt` below now builds the
  `GaussianProcessRegressor` itself with an explicit per-dimension
  `length_scale` array, `length_scale_bounds` that allow the optimizer to
  move (not clamp) the scales, `normalize_y=True`, and several
  `n_restarts_optimizer` restarts, and passes it in as `base_estimator`.

  The `propose_next_batch_builtin` fallback (only used when `skopt` is not
  installed) previously used a single hardcoded `length_scale = 0.3` shared
  across all 6 hyperparameter dimensions -- this is the actual bug described
  above. It now fits one length-scale PER DIMENSION by numerically
  maximizing the GP marginal log-likelihood (`_fit_ard_length_scales`),
  falling back to a shared default only when there are too few observations
  to fit 6 independent scales reliably (n < d + 2).

─────────────────────────────────────────────────────────────────────────────
EXECUTION FLOW
─────────────────────────────────────────────────────────────────────────────
1.  Startup: load runs/bayes_search/search_state.json if it exists
    (crash-safe resume); otherwise build Round 0 = 1 fixed default config +
    (round0_size - 1) Sobol space-filling configs. round0_size defaults to
    ~10x the search dimensionality (10 * 6 = 60; see --initial-round-size)
    rather than --configs-per-round, so the very first round densely covers
    the 6-D space before the GP starts trusting local structure. These
    configs are still dispatched through the normal --gpu-ids worker pool,
    so with e.g. 8 GPUs a 60-config round 0 naturally runs as ~8 sequential
    waves per GPU (each GPU pulls its next config off the shared queue the
    instant it frees up -- no explicit "wave" bookkeeping needed).

2.  Per-round evaluation via a GPU worker pool (--gpu-ids, default all 8):
      Every config in the round's queue is handed to the next free GPU worker.
      Each worker owns ONE physical GPU for the config's ENTIRE cycle below,
      so N configs (N = number of GPUs) run fully in parallel, each isolated
      to its own CUDA_VISIBLE_DEVICES:
      a. Launch benchmark_gsi_strategies_harmlessness.py as a subprocess
         with CUDA_VISIBLE_DEVICES=<gpu_id>.  Block until complete.
      b. Sleep 5 s to let VRAM drain.
      c. Launch this worker's OWN vLLM judge server (Qwen2.5-32B-Instruct,
         bitsandbytes 4-bit) on the SAME gpu_id.  Poll /v1/models until ready
         (up to 15 min). Each GPU worker gets its own judge server and port
         (8000 + gpu_id), fully independent of every other GPU's judge.
      d. Run tribunal.run_eval against this worker's judge server.
      e. Kill the judge server.  Sleep 5 s.
      f. Read model_summary.csv, compute the scalar objective:
           quality = mean(response_quality, relevance)
           safety  = 1 - mean(toxicity, harmfulness)
           objective = (2 * quality * safety) / (quality + safety)
      g. Append the record and IMMEDIATELY write both search_state.json
         (full state, resumable) and all_observations.csv (for inspection).
         Writes are serialized across workers with a lock so concurrent GPU
         workers finishing at the same time never corrupt the checkpoint.
         Optionally push to Hugging Face Hub as an offsite backup.
      h. On generation or scoring failure for a given GPU's config: log the
         error, return the config to the round's queue for a retry by any
         free worker, and keep that GPU's worker alive for the next config.
         A config is only abandoned (search aborts) on repeated Ctrl-C.

3.  End of each round:
      a. Fit a GP surrogate on all observations so far (both skopt and the
         built-in path Z-normalise y_obs before fitting, and both now fit
         per-dimension (ARD) length-scales to the data rather than using a
         single fixed length-scale -- see "GP length-scale fitting" above).
      b. Propose the next batch of configs via Expected Improvement (EI)
         with a diversity penalty so the proposals spread across the space.
      c. If max(EI) < min_expected_improvement (default 0.01) AND at least
         --min-rounds rounds have been completed, STOP. The --min-rounds
         floor (default 4) exists specifically so a round can't declare
         "converged" purely because the batch just evaluated happened to be
         sparse/uninformative early on -- the search must run a minimum
         number of full rounds regardless of the EI value before an
         EI-based stop is honored.
      d. Save checkpoint for the new round's configs.
      e. Generate intermediate plots (hp_effects, pareto frontier,
         convergence, correlation heatmap, GP partial dependence).
      f. Thermal cooldown: sleep for cooldown_seconds (default 3600 = 1 h)
         while printing a live countdown; user can press Enter to skip.

4.  On search completion (EI threshold met AND min-rounds satisfied, or
    Ctrl-C):
      Fit a final surrogate, generate final plots, and write best_config.json.

─────────────────────────────────────────────────────────────────────────────
FILE LAYOUT (all under <output_root>/)
─────────────────────────────────────────────────────────────────────────────
  runs/bayes_search/
    search_state.json          ← full resumable checkpoint (updated per config)
    logs/
      gen_<label>_gpu<id>.log  ← stdout/stderr from generation subprocess
      judge_gpu<id>_port<p>.log← vLLM judge server log (one per GPU worker,
                                  concurrent workers get distinct ports)
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
  Combined ≈ 54–58 GB — too large for concurrent execution on a single 48 GB
  A6000 when accounting for CUDA overhead and activations. This rule is
  enforced PER GPU WORKER: within one GPU's worker thread, the generation
  subprocess fully exits and VRAM is released (sleep) BEFORE that worker's
  own judge server starts on the same GPU, and vice versa. This is entirely
  independent per GPU, so GPU 0 judging and GPU 1 generating at the same
  moment is fine — they are different physical devices with separate VRAM.

─────────────────────────────────────────────────────────────────────────────
KEY CLI FLAGS
─────────────────────────────────────────────────────────────────────────────
  --repo-root              Repo root (contains evaluation/, tribunal/, etc.)
  --configs-per-round      Configs per round from round 1 onward (default 16).
                            Round 0 uses --initial-round-size instead (see
                            below) so the very first round is much denser.
  --initial-round-size     Number of configs in ROUND 0 specifically (default:
                            10x the search dimensionality, i.e. 10*6 = 60,
                            clamped to [50, 70] per the usual 6-D rule of
                            thumb). These are still spread across whatever
                            GPU worker pool --gpu-ids defines; with 8 GPUs
                            that's roughly 8 sequential waves per GPU.
  --min-rounds             Minimum number of FULL rounds that must complete
                            before an EI-based convergence stop is allowed,
                            regardless of how low max(EI) is (default 4).
                            Prevents a false early stop from a sparse/lucky
                            early round.
  --gpu-ids                Comma-separated physical GPU indices forming the
                            worker pool, e.g. "0,1,2,3,4,5,6,7" (default: all
                            8 GPUs 0-7). Each listed GPU runs its own
                            generate->judge->score cycle concurrently with
                            the others. Pass a single id (e.g. "0") to fall
                            back to the original single-GPU sequential run.
  --num-prompts            Prompts per config evaluation (default 15)
  --max-tokens              Max generation tokens (default 512)
  --judge-model             vLLM judge model (default Qwen/Qwen2.5-32B-Instruct)
  --min-expected-improvement  EI stopping threshold (default 0.01)
  --cooldown-seconds        Thermal break between rounds in seconds (default 3600)
  --manual-resume            Pause before each config and wait for Enter
  --hf-repo-id                Optional HF Hub dataset repo for offsite backups
  --output-root              Root for all output dirs (default: repo-root)
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
import threading
from concurrent.futures import ThreadPoolExecutor
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
    "--sigma-mode", "min_entropy",
    "--gsi-max-step-tokens", "80",
    # beta is fixed to the blade's training value; sweeping it is redundant
    # because the blade reward is Z-normalized before entering the UWO logit,
    # so any multiplicative scaling by beta cancels out (subsumed by w_blade).
    "--beta", "0.1",
]

STRATEGY_NAME = "elo_swiss_mode_b"

_QUALITY_METRICS = ["response_quality", "relevance"]
_SAFETY_METRICS = ["toxicity", "harmfulness"]

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
    """
    Harmonic mean (F1-style) of Quality axis and Safety axis.

    quality = mean(response_quality, relevance)
    safety  = 1 - mean(toxicity, harmfulness)

    Helpfulness and refusal are intentionally excluded:
      - Helpfulness penalises safe refusals on sensitive/adversarial prompts and duplicates quality/relevance.
      - Refusal is a diagnostic metric, not an optimization target.
    """
    q_vals = [metrics[m] for m in _QUALITY_METRICS if m in metrics]
    s_vals = [metrics[m] for m in _SAFETY_METRICS if m in metrics]

    if not q_vals or not s_vals:
        return 0.0

    quality = sum(q_vals) / len(q_vals)
    safety = 1.0 - (sum(s_vals) / len(s_vals))

    quality = max(quality, 1e-6)
    safety = max(safety, 1e-6)

    return (2.0 * quality * safety) / (quality + safety)


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


def _fit_ard_length_scales(Xn: np.ndarray, yn: np.ndarray, noise: float = 1e-4,
                            init_length_scale: float = 0.3,
                            bounds: Tuple[float, float] = (0.02, 3.0)) -> np.ndarray:
    """Fit one RBF length-scale PER DIMENSION (automatic relevance
    determination / ARD) by numerically maximizing the GP marginal
    log-likelihood, instead of using a single fixed length-scale shared
    across every hyperparameter dimension.

    This replaces the previous hardcoded ``length_scale = 0.3`` used
    uniformly for all 7 dimensions in the built-in GP fallback
    (`propose_next_batch_builtin` / `estimate_expected_improvement`), which
    is only exercised when ``skopt`` is not installed. When ``scikit-optimize``
    IS installed (the default, and what the logs show being used), the GP
    inside ``propose_next_batch_skopt`` fits its own per-dimension
    length-scales via marginal-likelihood optimization -- see that function.

    Falls back to a single shared ``init_length_scale`` per dimension when
    there are too few observations (n < d + 2) to reliably fit d independent
    scales -- fitting 7 free length-scale parameters from e.g. 3 points is
    not well posed and would just overfit noise.
    """
    from scipy.optimize import minimize
    from scipy.linalg import cho_factor, cho_solve

    n, d = Xn.shape
    if n < d + 2:
        return np.full(d, init_length_scale)

    log_lo, log_hi = np.log(bounds[0]), np.log(bounds[1])

    def neg_log_marginal_likelihood(log_ls: np.ndarray) -> float:
        ls = np.exp(log_ls)
        diff = Xn[:, None, :] - Xn[None, :, :]
        d2 = np.sum((diff / ls) ** 2, axis=-1)
        K = np.exp(-0.5 * d2) + noise * np.eye(n)
        try:
            c, low = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve((c, low), yn)
        log_det = 2.0 * np.sum(np.log(np.diag(c)))
        nll = 0.5 * float(yn @ alpha) + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
        return nll

    # A handful of restarts from different starting length-scales to reduce
    # the chance of the L-BFGS-B optimizer settling in a poor local optimum
    # of the (non-convex) marginal likelihood surface.
    best_ls, best_nll = None, np.inf
    rng = np.random.default_rng(0)
    starts = [np.full(d, np.log(init_length_scale))]
    starts += [rng.uniform(log_lo, log_hi, size=d) for _ in range(3)]
    for x0 in starts:
        try:
            res = minimize(
                neg_log_marginal_likelihood, x0,
                bounds=[(log_lo, log_hi)] * d, method="L-BFGS-B",
            )
        except Exception as e:
            logger.warning("ARD length-scale fit attempt failed (%s).", e)
            continue
        if res.success and res.fun < best_nll:
            best_nll, best_ls = res.fun, np.exp(res.x)

    if best_ls is None:
        logger.warning("ARD length-scale fit did not converge from any start; "
                        "falling back to fixed length-scale %.3f for all dims.", init_length_scale)
        return np.full(d, init_length_scale)
    return best_ls


def propose_next_batch_skopt(X_obs: np.ndarray, y_obs: np.ndarray, n_proposals: int,
                              seed: int) -> Tuple[List[Dict[str, float]], object]:
    from skopt import Optimizer
    from skopt.space import Real, Integer
    # NOTE: must be skopt's own GaussianProcessRegressor (skopt.learning), NOT
    # sklearn.gaussian_process.GaussianProcessRegressor. skopt's acquisition
    # optimizer (used inside Optimizer.tell()/.ask() to find the next point
    # via L-BFGS) calls model.predict(..., return_mean_grad=True,
    # return_std_grad=True) -- kwargs that only skopt's subclassed GP
    # implements. Passing plain sklearn's GP class here fits fine but then
    # crashes with "unexpected keyword argument 'return_mean_grad'" the
    # moment skopt tries to optimize the acquisition function.
    from skopt.learning import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

    dims = []
    for name in HP_NAMES:
        _, lo, hi, dtype, _ = SEARCH_SPACE[name]
        dims.append(Integer(int(lo), int(hi), name=name) if dtype == "int" else Real(lo, hi, name=name))

    # Apply Z-normalization to the objectives for stable surrogate modeling
    y_mean = y_obs.mean()
    y_std = y_obs.std() if y_obs.std() > 1e-8 else 1.0
    yn = (y_obs - y_mean) / y_std

    # Explicit ARD (per-dimension length-scale) GP: skopt's own default
    # base_estimator="GP" already builds a Matern kernel with one
    # length-scale PER DIMENSION and fits those scales (plus output/noise
    # scale) by maximizing the marginal log-likelihood via L-BFGS restarts
    # -- it is not a single fixed length-scale. We build the estimator
    # explicitly here anyway so this behavior is pinned down and visible
    # rather than implicit in skopt's defaults: one length_scale per HP
    # dimension, generous length_scale_bounds so the optimizer is actually
    # free to move them, normalize_y=True, and several optimizer restarts.
    n_dims = len(HP_NAMES)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    )
    gp_estimator = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )

    opt = Optimizer(dims, base_estimator=gp_estimator, acq_func="EI", random_state=seed,
                     n_initial_points=0, acq_optimizer="sampling")
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
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    Xn = _normalize(X_obs)
    y = y_obs.copy()
    y_mean, y_std = y.mean(), (y.std() + 1e-8)
    yn = (y - y_mean) / y_std

    # Fit one length-scale PER DIMENSION via marginal-likelihood optimization
    # (ARD) instead of using a single fixed length-scale for all 7 HPs.
    length_scales = _fit_ard_length_scales(Xn, yn)
    noise = 1e-4

    def kernel(A, B):
        diff = A[:, None, :] - B[None, :, :]
        d2 = np.sum((diff / length_scales) ** 2, axis=-1)
        return np.exp(-0.5 * d2)

    from scipy.linalg import cho_factor, cho_solve

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
    surrogate = {"predict": gp_predict, "x_mean": y_mean, "x_std": y_std, "length_scales": length_scales}
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
    from scipy.linalg import cho_factor, cho_solve
    from scipy.stats import norm

    Xn = _normalize(X_obs)
    y = y_obs.copy()
    y_mean, y_std = y.mean(), (y.std() + 1e-8)
    yn = (y - y_mean) / y_std

    # Same ARD (per-dimension) length-scale fitting as propose_next_batch_builtin,
    # so this EI estimate (used for the convergence check / logging even when
    # skopt was used to generate the proposals) reflects an actually-fitted
    # kernel rather than one fixed length-scale for every hyperparameter.
    length_scales = _fit_ard_length_scales(Xn, yn)
    noise = 1e-4

    def kernel(A, B):
        diff = A[:, None, :] - B[None, :, :]
        d2 = np.sum((diff / length_scales) ** 2, axis=-1)
        return np.exp(-0.5 * d2)

    K = kernel(Xn, Xn) + noise * np.eye(len(Xn))
    c, low = cho_factor(K, lower=True)
    alpha = cho_solve((c, low), yn)

    Xp = _normalize(np.array([[pt[n] for n in HP_NAMES] for pt in proposals], dtype=float))
    Ks = kernel(Xp, Xn)

    # FIX (Z-normalization / EI-units mismatch): compute EI entirely in
    # normalized (Z-scored) space, matching the docstring's stated intent
    # that "the EI threshold (0.01) is ... expressed in the same
    # standardised units as the GP's output."
    #
    # Previously mu/sigma here were denormalized back to RAW objective
    # units (`* y_std + y_mean`, `* y_std`) and best_y was the raw-scale
    # max, BEFORE computing EI. That made the resulting EI values scale
    # with the raw objective's standard deviation (y_std). When raw
    # objective values are tightly clustered across configs (small y_std --
    # common for judge-scored metrics that all land in a similar range),
    # EI was squeezed toward ~0 regardless of whether the search had
    # actually converged. Since these EI values are what get compared
    # against --min-expected-improvement in the convergence check, that
    # unit mismatch could trigger a false-positive early stop after only
    # --min-rounds rounds, even though the surrogate genuinely still had
    # promising, uncertain regions left to explore.
    #
    # Keeping mu/sigma/best_y all in normalized units (mean 0, std 1) fixes
    # this: EI is now dimensionless and comparable to a fixed threshold
    # regardless of the raw objective's scale.
    mu_n = Ks @ alpha
    v = cho_solve((c, low), Ks.T)
    var_n = np.clip(1.0 - np.sum(Ks.T * v, axis=0), 1e-9, None)
    sigma_n = np.sqrt(var_n)

    best_yn = yn.max()
    z = (mu_n - best_yn) / sigma_n
    ei = (mu_n - best_yn) * norm.cdf(z) + sigma_n * norm.pdf(z)
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

    if all(m in df.columns for m in ["relevance", "response_quality", "toxicity", "harmfulness"]):
        df["quality_axis"] = df[["response_quality", "relevance"]].mean(axis=1)
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
        ax.set_xlabel("Quality axis (mean of response_quality, relevance)")
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


# A lock guarding all checkpoint writes (search_state.json / all_observations.csv).
# Multiple GPU workers can finish a config at nearly the same wall-clock moment;
# this ensures writes are serialized so the checkpoint file is never corrupted
# by two threads writing it simultaneously. This does NOT change what gets
# written, only the fact that only one thread writes at a time.
_checkpoint_lock = threading.Lock()

# Sentinel strings returned by process_config_on_gpu to distinguish a
# permanent OOM (config should be dropped and never retried) from any other
# failure (transient; caller re-enqueues cfg for a retry on the next free GPU).
OOM_DROP = "OOM_DROP"
RETRY = "RETRY"

_OOM_LOG_SIGNATURES = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUDA error: out of memory",
    "HIP out of memory",
)


def _log_indicates_oom(log_path: str) -> bool:
    """Best-effort check of a subprocess log file for a CUDA/GPU OOM signature.

    Used to decide whether a failed config's failure is a genuine
    out-of-memory event (in which case the config is dropped for good, since
    retrying it will deterministically OOM again given the same GPU and the
    same hyperparameters) versus some other transient failure (network
    hiccup, disk error, judge server flake, etc.) which IS worth retrying.
    """
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
    except OSError:
        return False
    return any(sig in content for sig in _OOM_LOG_SIGNATURES)


def process_config_on_gpu(
    cfg: "HPConfig",
    gpu_id: int,
    args,
    repo_root: str,
    log_dir: str,
    tribunal_root: str,
    state_file: str,
    csv_file: str,
    shared: dict,
) -> Tuple[Optional[dict], str]:
    """Run ONE config's full cycle (generate -> own judge server -> score) on
    a single physical GPU, start to finish. This is exactly the per-config
    body of the original single-GPU sequential loop, unchanged in its
    subprocess commands / judge lifecycle / objective computation -- the
    only difference is that this function is designed to be called
    concurrently, once per GPU worker, against a shared config queue.

    Returns a tuple ``(metrics_rec_or_None, status)`` where ``status`` is:
      - ``"OK"``       on success (metrics_rec is the scored result dict).
      - ``OOM_DROP``   if the failure was a genuine CUDA/GPU out-of-memory
                        event. This config is DELIBERATELY NOT RETRIED --
                        the same hyperparameters on the same GPU will
                        deterministically OOM again, so retrying would just
                        waste a full generate/judge cycle forever. The
                        caller drops it from the search entirely (it is
                        excluded from the GP fit, as if it were never
                        proposed).
      - ``RETRY``      for any other failure (subprocess crash, judge server
                        flake, disk error, etc.) that may succeed on a
                        second attempt. The caller re-enqueues cfg for the
                        next free GPU worker.

    On success, mutates ``shared['evaluated_configs']`` and writes
    checkpoints under ``_checkpoint_lock``.
    """
    round_idx = cfg.round_idx

    # ── GENERATION (this GPU only) ───────────────────────────────────────
    if args.manual_resume:
        print(f"\n[GPU {gpu_id}] [GENERATION] Ready for {cfg.label()} ({cfg.values}). Press Enter...")
        input()

    logger.info("=" * 80)
    logger.info("[GPU %d] GENERATION | Round %d | config %s", gpu_id, round_idx, cfg.label())
    logger.info("[GPU %d] Parameters: %s", gpu_id, cfg.values)
    logger.info("=" * 80)

    out_dir = os.path.join(repo_root, "runs", shared["run_subdir"], f"round{round_idx}", cfg.label())
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(repo_root, "evaluation", shared["benchmark_script"])

    cmd = [
        sys.executable, script,
        "--num-prompts", str(args.num_prompts),
        "--max-tokens", str(args.max_tokens),
        "--output-dir", out_dir,
    ] + FIXED_FLAGS + cfg.cli_args() + args.extra_flag

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    gen_log = os.path.join(log_dir, f"gen_{cfg.label()}_gpu{gpu_id}.log")
    logger.info("[GPU %d] Logging generation to %s", gpu_id, gen_log)

    with open(gen_log, "w") as f:
        ret = subprocess.run(cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT)

    if ret.returncode != 0:
        if _log_indicates_oom(gen_log):
            logger.error(
                "[GPU %d] Generation OOM'd (code %d) for %s -- DROPPING this config "
                "permanently (will not retry). Log: %s",
                gpu_id, ret.returncode, cfg.label(), gen_log,
            )
            logger.info("[GPU %d] Clearing GPU %d and moving on to the next queued config.", gpu_id, gpu_id)
            return None, OOM_DROP
        logger.error("[GPU %d] Generation FAILED (code %d) for %s. Log: %s",
                      gpu_id, ret.returncode, cfg.label(), gen_log)
        return None, RETRY

    logger.info("[GPU %d] Generation done for %s. Sleeping 15 s to drain VRAM...", gpu_id, cfg.label())
    time.sleep(15)

    # ── JUDGING (this GPU's own private judge server) ────────────────────
    logger.info("=" * 80)
    logger.info("[GPU %d] JUDGING | Round %d | config %s", gpu_id, round_idx, cfg.label())
    logger.info("[GPU %d] Sleeping 5 s to allow GPU driver to fully release VRAM...", gpu_id)
    time.sleep(5)
    logger.info("[GPU %d] Starting judge server (this takes ~2-3 min)...", gpu_id)
    logger.info("=" * 80)

    port = 8000 + gpu_id
    judge_proc = start_judge_server(gpu_id, port, repo_root, log_dir, args.judge_model)

    try:
        ready = wait_for_server(port, timeout_s=900)
        if not ready:
            raise RuntimeError(f"[GPU {gpu_id}] Judge server failed to become ready within 15 minutes.")

        if args.manual_resume:
            print(f"\n[GPU {gpu_id}] [JUDGING] Ready to score {cfg.label()}. Press Enter...")
            input()

        logger.info("[GPU %d] Judging config %s...", gpu_id, cfg.label())

        # Convert generation output to jsonl for tribunal
        tribunal_inputs_dir = os.path.join(tribunal_root, "inputs", f"round{round_idx}", cfg.label())
        jsonl_path = convert_json_to_jsonl(out_dir, tribunal_inputs_dir, model_name=cfg.label())
        if jsonl_path is None:
            raise RuntimeError(f"JSONL conversion failed for {cfg.label()}.")

        results_dir = os.path.join(tribunal_root, "eval_results", f"round{round_idx}", cfg.label())
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
        eval_log = os.path.join(log_dir, f"tribunal_{cfg.label()}_gpu{gpu_id}.log")
        logger.info("[GPU %d] Tribunal scoring -> %s", gpu_id, eval_log)

        with open(eval_log, "w") as f:
            ret_eval = subprocess.run(
                cmd_eval, cwd=os.path.join(repo_root, "tribunal"),
                env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT
            )

        shutil.rmtree(tmp_input_dir, ignore_errors=True)

        if ret_eval.returncode != 0:
            if _log_indicates_oom(eval_log):
                logger.error(
                    "[GPU %d] Tribunal scoring OOM'd for %s -- DROPPING this config permanently.",
                    gpu_id, cfg.label(),
                )
                return None, OOM_DROP
            raise RuntimeError(f"tribunal.run_eval failed (code {ret_eval.returncode}) for {cfg.label()}.")

        metrics = read_metrics(results_dir, cfg.label())
        if metrics is None:
            raise RuntimeError(f"Could not read model_summary.csv for {cfg.label()}.")

        obj = scalar_objective(metrics)
        metrics_rec = {
            "cfg_label": cfg.label(),
            "round": round_idx,
            **cfg.values,
            **metrics,
            "objective": obj,
        }

        # Commit result and save checkpoint immediately (serialized across GPU workers)
        with _checkpoint_lock:
            shared["evaluated_configs"].append(metrics_rec)
            save_search_state(
                state_file, round_idx, shared["configs_to_generate"],
                shared["configs_in_flight"], shared["evaluated_configs"], shared["seed"],
            )
            save_all_observations_csv(csv_file, shared["evaluated_configs"])
        logger.info("[GPU %d] Scored %s -> objective=%.4f", gpu_id, cfg.label(), obj)

        if args.hf_repo_id:
            upload_snapshot_to_hf(
                local_dir=os.path.join(shared["output_root"], "runs", shared["run_subdir"]),
                repo_id=args.hf_repo_id, token=args.hf_token,
                commit_message=f"[{shared['benchmark_type']}] Round {round_idx} judged {cfg.label()} (GPU {gpu_id})"
            )

        return metrics_rec, "OK"

    except Exception as e:
        logger.error("[GPU %d] Judge phase failed for %s: %s", gpu_id, cfg.label(), e)
        return None, RETRY
    finally:
        logger.info("[GPU %d] Stopping judge server...", gpu_id)
        stop_judge_server(judge_proc)
        logger.info("[GPU %d] Judge stopped. Sleeping 15 s to drain VRAM...", gpu_id)
        time.sleep(15)


def run_round_on_gpu_pool(
    configs: List["HPConfig"],
    gpu_ids: List[int],
    args,
    repo_root: str,
    log_dir: str,
    tribunal_root: str,
    state_file: str,
    csv_file: str,
    shared: dict,
) -> None:
    """Evaluate every config in ``configs`` using a pool of GPU workers.

    Each GPU in ``gpu_ids`` runs its own persistent worker loop: pull the next
    config off a single shared queue, run its full generate->judge->score
    cycle, then IMMEDIATELY pull the next one -- independent of how long any
    other GPU's current config takes. A GPU never waits for the rest of the
    round to catch up before picking up new work; it only goes idle once the
    shared queue is empty and every in-flight config has finished.

    With a densely-populated round (e.g. a 50-70 config round 0) and a fixed
    GPU pool, this loop structure IS the "multiple waves" execution: each GPU
    just keeps dequeuing until the shared queue is empty, so a round with
    e.g. 70 configs across 8 GPUs naturally plays out as ~9 sequential
    per-GPU waves without any extra wave-scheduling logic being required.

    A config that fails with a genuine CUDA/GPU out-of-memory error is
    DROPPED PERMANENTLY here -- it is never re-queued, and it never enters
    ``shared['evaluated_configs']``, so it is excluded entirely from the GP
    fit (as if that config had never been proposed). The GPU that hit the
    OOM immediately moves on to the next pending config; nothing else about
    the pool or the other GPU workers is affected. Any other kind of
    failure (RETRY) is re-queued for the next free GPU, exactly as before.

    This does not change the search math (GP/EI/Z-normalization) at all --
    it only parallelizes the per-config subprocess work of a round across
    independent physical GPUs. With ``gpu_ids == [g]`` (a single GPU) this
    reduces exactly to the original strictly-sequential behavior.
    """
    queue_lock = threading.Lock()
    pending = list(configs)  # shared work queue; protected by queue_lock
    shared["configs_in_flight"] = []
    shared.setdefault("dropped_configs", [])

    def gpu_worker_loop(gpu_id: int) -> None:
        while True:
            with queue_lock:
                if not pending:
                    return  # queue drained; this GPU worker is done for the round
                cfg = pending.pop(0)
                shared["configs_in_flight"].append(cfg)

            try:
                result, status = process_config_on_gpu(
                    cfg, gpu_id, args, repo_root, log_dir,
                    tribunal_root, state_file, csv_file, shared,
                )
            except Exception as e:
                logger.error("[GPU %d] Worker for %s raised unexpectedly: %s", gpu_id, cfg.label(), e)
                result, status = None, RETRY

            with queue_lock:
                shared["configs_in_flight"].remove(cfg)
                if status == OOM_DROP:
                    # Permanent drop: do NOT re-queue. GPU is already free to
                    # pick up the next pending config on its next loop pass.
                    shared["dropped_configs"].append({"cfg_label": cfg.label(), "values": cfg.values, "reason": "OOM"})
                    logger.warning(
                        "[GPU %d] %s permanently dropped due to OOM (excluded from GP fit). "
                        "%d config(s) remaining in queue.",
                        gpu_id, cfg.label(), len(pending),
                    )
                elif status == RETRY:
                    # Transient failure. Re-queue immediately so THIS SAME
                    # (now-free) GPU -- or any other GPU that empties its
                    # queue first -- picks it up right away, with no wait
                    # for other GPUs' current configs to finish.
                    logger.warning(
                        "[GPU %d] %s failed (non-OOM) and is re-queued for immediate retry.",
                        gpu_id, cfg.label(),
                    )
                    pending.append(cfg)
                # status == "OK": nothing to do here, already committed to
                # shared['evaluated_configs'] inside process_config_on_gpu.
            # Loop back around immediately to grab the next config, if any.

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(gpu_worker_loop, gpu_id) for gpu_id in gpu_ids]
        for fut in futures:
            fut.result()  # propagate any unexpected exception from a worker loop itself


BENCHMARK_SCRIPTS = {
    "harmlessness": "benchmark_HH_harmlessness.py",
    "helpfulness": "benchmark_gsi_strategies_helpfulness.py",
    "truthfulness": "benchmark_gsi_strategies_truthfulness.py",
}


def run_search_for_benchmark(benchmark_type: str, args, gpu_ids: List[int], repo_root: str) -> None:
    """Run the full Bayesian hyperparameter search (all rounds, until EI
    convergence) for ONE benchmark type end to end, with its own completely
    separate output tree so results from different benchmark types never mix:

        runs/bayes_search_<benchmark_type>/...
        tribunal/bayes_search_<benchmark_type>/...

    Each benchmark type gets its own search_state.json, so runs for
    different benchmark types resume independently of each other.
    """
    if benchmark_type not in BENCHMARK_SCRIPTS:
        raise ValueError(f"Unknown --benchmark-type '{benchmark_type}'. Choose from: {list(BENCHMARK_SCRIPTS)}")
    benchmark_script = BENCHMARK_SCRIPTS[benchmark_type]

    output_root = args.output_root or repo_root
    run_subdir = f"bayes_search_{benchmark_type}"
    log_dir = os.path.join(output_root, "runs", run_subdir, "logs")
    plot_dir = os.path.join(output_root, "runs", run_subdir, "plots")
    tribunal_root = os.path.join(output_root, "tribunal", run_subdir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    state_file = os.path.join(output_root, "runs", run_subdir, "search_state.json")
    csv_file = os.path.join(plot_dir, "all_observations.csv")

    logger.info("=" * 80)
    logger.info("BENCHMARK TYPE: %s  (script: %s)", benchmark_type, benchmark_script)
    logger.info("Output root: %s", os.path.join(output_root, "runs", run_subdir))
    logger.info("=" * 80)
    logger.info("GPU worker pool: %s (%d GPU(s))", gpu_ids, len(gpu_ids))

    # Resume from checkpoint or initialize
    # NOTE: "configs_to_judge" in the checkpoint is retained as the on-disk key
    # name for backward compatibility with existing search_state.json files;
    # in the multi-GPU pool it represents configs still needing a full
    # generate->judge->score cycle (a worker does all of it per config now,
    # rather than the old two separate generate-queue / judge-queue phases).
    loaded = load_search_state(state_file)
    if loaded is not None:
        round_idx, configs_to_generate, configs_still_pending, evaluated_configs, seed = loaded
        configs_pending = configs_to_generate + configs_still_pending
        logger.info(
            "[%s] Resuming search. Round %d | pending configs: %d | evaluated: %d",
            benchmark_type, round_idx, len(configs_pending), len(evaluated_configs)
        )
    else:
        round_idx = 0
        seed = args.seed
        # Round 0 is intentionally much denser than later rounds: with a 7-D
        # search space, the ~10x-dimensionality rule of thumb (50-70 initial
        # points) gives the GP enough space-filling coverage to fit sane
        # length-scales and avoid mistaking early sparsity for convergence.
        # --initial-round-size overrides this; otherwise it's derived from
        # the dimensionality and clamped to the [50, 70] rule-of-thumb band.
        if args.initial_round_size is not None:
            round0_size = args.initial_round_size
        else:
            round0_size = int(np.clip(10 * len(HP_NAMES), 50, 70))
        configs_pending = build_round0_configs(round0_size, seed=seed)
        evaluated_configs = []
        logger.info(
            "[%s] Initializing from scratch. Round 0 configs: %d (dense initial "
            "space-filling batch, ~10x the %d-D search space; will be spread "
            "across the %d-GPU worker pool as ~%d sequential waves per GPU).",
            benchmark_type, len(configs_pending), len(HP_NAMES), len(gpu_ids),
            int(np.ceil(len(configs_pending) / max(len(gpu_ids), 1))),
        )
        save_search_state(state_file, round_idx, [], configs_pending, evaluated_configs, seed)

    shared = {
        "configs_to_generate": [],  # kept empty; retained for checkpoint-format compatibility
        "configs_in_flight": [],
        "evaluated_configs": evaluated_configs,
        "dropped_configs": [],
        "seed": seed,
        "output_root": output_root,
        "benchmark_type": benchmark_type,
        "benchmark_script": benchmark_script,
        "run_subdir": run_subdir,
    }

    while True:
        # ── ROUND: dispatch every pending config across the GPU worker pool ────
        if configs_pending:
            logger.info("=" * 80)
            logger.info(
                "[%s] ROUND %d | %d config(s) pending | %d GPU worker(s): %s",
                benchmark_type, round_idx, len(configs_pending), len(gpu_ids), gpu_ids,
            )
            logger.info("=" * 80)

            run_round_on_gpu_pool(
                configs_pending, gpu_ids, args, repo_root, log_dir,
                tribunal_root, state_file, csv_file, shared,
            )
            evaluated_configs = shared["evaluated_configs"]
            configs_pending = []  # everything either succeeded, was permanently dropped (OOM), or aborted via Ctrl-C

        # ── ROUND COMPLETE ──────────────────────────────────────────────────────
        logger.info("=" * 80)
        logger.info(
            "[%s] Round %d complete. Evaluated: %d | Dropped (OOM): %d",
            benchmark_type, round_idx, len(evaluated_configs), len(shared["dropped_configs"]),
        )
        logger.info("=" * 80)

        X_obs = np.array([[r[n] for n in HP_NAMES] for r in evaluated_configs], dtype=float)
        y_obs = np.array([r["objective"] for r in evaluated_configs], dtype=float)

        if len(evaluated_configs) < 2:
            logger.error("[%s] Fewer than 2 observations. Cannot fit GP. Aborting this benchmark's search.", benchmark_type)
            break

        logger.info("[%s] Fitting GP surrogate and computing Expected Improvement...", benchmark_type)
        proposals, surrogate, ei_values = propose_next_batch(X_obs, y_obs, args.configs_per_round, seed=seed + round_idx + 1)
        max_ei = max(ei_values) if ei_values else 0.0
        logger.info("[%s] Proposed EI values: %s  (max EI: %.4f)", benchmark_type, ei_values, max_ei)

        completed_rounds = round_idx + 1  # round_idx is 0-based; this round just finished
        converged = max_ei < args.min_expected_improvement

        if converged and completed_rounds < args.min_rounds:
            logger.info(
                "[%s] Max EI (%.4f) < threshold (%.4f), but only %d/%d minimum rounds "
                "completed -- continuing the search anyway to avoid a false early stop "
                "from an uninformative early round.",
                benchmark_type, max_ei, args.min_expected_improvement,
                completed_rounds, args.min_rounds,
            )
        elif converged:
            logger.info("=" * 80)
            logger.info(
                "[%s] Max EI (%.4f) < threshold (%.4f) after %d/%d minimum rounds. Search converged.",
                benchmark_type, max_ei, args.min_expected_improvement, completed_rounds, args.min_rounds,
            )
            logger.info("=" * 80)
            break

        # Build next round
        round_idx += 1
        configs_pending = [
            HPConfig(cfg_id=f"cfg{i}", round_idx=round_idx, values=v)
            for i, v in enumerate(proposals)
        ]
        logger.info("[%s] Round %d configs proposed: %s", benchmark_type, round_idx, [c.values for c in configs_pending])

        save_search_state(state_file, round_idx, [], configs_pending, evaluated_configs, seed)

        try:
            make_plots(evaluated_configs, plot_dir, surrogate=surrogate)
            make_tribunal_comparison_plots(evaluated_configs, tribunal_root, repo_root, plot_dir, "running_summary")
        except Exception as e:
            logger.warning("[%s] Plot generation failed (non-fatal): %s", benchmark_type, e)

        # Thermal cooldown between rounds
        cooldown_break(args.cooldown_seconds)

    # Finalize this benchmark's search
    if not evaluated_configs:
        logger.error("[%s] No configs were successfully evaluated (all failed or OOM'd). Skipping finalize.", benchmark_type)
        return

    best = max(evaluated_configs, key=lambda r: r["objective"])
    logger.info("=" * 80)
    logger.info("[%s] SEARCH COMPLETE.", benchmark_type)
    logger.info("[%s] Best configuration found: %s", benchmark_type, best)
    if shared["dropped_configs"]:
        logger.info("[%s] %d config(s) permanently dropped due to OOM: %s",
                     benchmark_type, len(shared["dropped_configs"]),
                     [d["cfg_label"] for d in shared["dropped_configs"]])
    logger.info("=" * 80)

    with open(os.path.join(plot_dir, "best_config.json"), "w") as f:
        json.dump(best, f, indent=2)

    if shared["dropped_configs"]:
        with open(os.path.join(plot_dir, "dropped_configs.json"), "w") as f:
            json.dump(shared["dropped_configs"], f, indent=2)

    try:
        X_obs = np.array([[r[n] for n in HP_NAMES] for r in evaluated_configs], dtype=float)
        y_obs = np.array([r["objective"] for r in evaluated_configs], dtype=float)
        _, surrogate_final = propose_next_batch_builtin(X_obs, y_obs, n_proposals=1, seed=seed)
        make_plots(evaluated_configs, plot_dir, surrogate=surrogate_final)
        make_tribunal_comparison_plots(evaluated_configs, tribunal_root, repo_root, plot_dir, "final")
    except Exception as e:
        logger.warning("[%s] Error creating final plots: %s", benchmark_type, e)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=".", help="Path to repo root.")
    p.add_argument("--configs-per-round", type=int, default=8,
                    help="Number of configs evaluated in each round FROM ROUND 1 ONWARD. "
                         "Round 0 uses --initial-round-size instead (much larger, for "
                         "dense initial space-filling coverage). Raised from the original "
                         "default of 7 so post-round-0 batches are less sparse too.")
    p.add_argument("--initial-round-size", type=int, default=48,
                    help="Number of configs in ROUND 0 specifically. Default: derived from "
                         "the search dimensionality using the ~10x-dims rule of thumb "
                         "(10 * 7 = 70), clamped to the [50, 70] band. These are still "
                         "dispatched through the normal --gpu-ids worker pool, so e.g. 70 "
                         "configs across 8 GPUs plays out as ~9 sequential waves per GPU.")
    p.add_argument("--min-rounds", type=int, default=4,
                    help="Minimum number of full rounds that must complete before an "
                         "EI-based convergence stop is honored, regardless of how low "
                         "max(EI) is. Prevents declaring convergence off the back of a "
                         "single sparse/uninformative early round.")
    p.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7",
                    help="Comma-separated physical GPU indices forming the worker pool. "
                         "Each GPU runs its own generate->judge->score cycle concurrently. "
                         "Pass a single id (e.g. '0') for the original single-GPU sequential behavior.")
    p.add_argument("--num-prompts", type=int, default=50, help="Number of prompts evaluated per config.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=None)
    p.add_argument("--extra-flag", action="append", default=[], help="Extra flag for generation.")
    p.add_argument("--min-expected-improvement", type=float, default=0.01,
                    help="EI threshold to stop search early.")
    p.add_argument("--cooldown-seconds", type=int, default=900,
                    help="Pause time in seconds after each round completion (default 1 hour).")
    p.add_argument("--manual-resume", action="store_true",
                    help="Require user input in terminal to start the next config/evaluation.")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument(
        "--benchmark-type", type=str, default="harmlessness,helpfulness",
        help="Comma-separated list of benchmark types to search, run FULLY "
             "SEQUENTIALLY in the given order (each one runs its complete "
             "search -- every round until EI convergence -- before the next "
             "one starts). Choose from: harmlessness, helpfulness, "
             "truthfulness. Each gets its own output tree: "
             "runs/bayes_search_<type>/ and tribunal/bayes_search_<type>/. "
             "Default runs harmlessness then helpfulness.",
    )
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)

    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip() != ""]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id.")

    benchmark_types = [b.strip() for b in args.benchmark_type.split(",") if b.strip() != ""]
    if not benchmark_types:
        raise ValueError("--benchmark-type must name at least one benchmark type.")
    for bt in benchmark_types:
        if bt not in BENCHMARK_SCRIPTS:
            raise ValueError(f"Unknown --benchmark-type '{bt}'. Choose from: {list(BENCHMARK_SCRIPTS)}")

    logger.info("Benchmark types to run, fully sequentially: %s", benchmark_types)

    for benchmark_type in benchmark_types:
        run_search_for_benchmark(benchmark_type, args, gpu_ids, repo_root)

    logger.info("=" * 80)
    logger.info("ALL BENCHMARK SEARCHES COMPLETE: %s", benchmark_types)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()