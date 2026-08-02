#!/usr/bin/env python3
"""
Bayesian hyperparameter search over Swiss-Knife's ``elo_swiss_mode_b`` GSI strategy,
tailored for the TRUTHFULNESS blade and optimizing 6 key hyperparameters:
    - elo_temperature
    - beta
    - w_tournament
    - w_blade
    - uwo_lambda
    - elo_rounds
"""

import argparse
import glob
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("bayes_search_truthfulness")

STRATEGY_NAME = "elo_swiss_mode_b"

SEARCH_SPACE: Dict[str, Tuple[str, float, float, str, float]] = {
    "elo_temperature": ("--elo-temperature", 1.0,  40.0, "float", 15.0),
    "beta":            ("--beta",            0.01, 1.0,  "float", 0.1),
    "w_tournament":    ("--w-tournament",     0.0,  3.0,  "float", 1.0),
    "w_blade":         ("--w-blade",          0.0,  3.0,  "float", 1.0),
    "uwo_lambda":      ("--uwo-lambda",       0.0,  1.0,  "float", 0.5),
    "elo_rounds":      ("--elo-rounds",       2.0,  10.0, "int",   4.0),
}

HP_NAMES = sorted(list(SEARCH_SPACE.keys()))

FIXED_FLAGS = [
    "--strategies", "elo_swiss_mode_b",
    "--probabilistic",
    "--sigma-mode", "log_ratio_proxy",
    "--gsi-max-step-tokens", "80",
    "--gsi-n", "8",
    "--blade", "truthfulness",
]


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


def launch_generation(cfg: HPConfig, gpu_id: int, repo_root: str, num_prompts: int,
                       max_tokens: int, extra_flags: List[str], log_dir: str) -> subprocess.Popen:
    out_dir = os.path.join(repo_root, "runs", "bayes_search_truthfulness", f"round{cfg.round_idx}", cfg.label())
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(repo_root, "evaluation", "benchmark_gsi_strategies_truthfulness.py")

    cmd = [
        sys.executable, script,
        "--num-prompts", str(num_prompts),
        "--max-tokens", str(max_tokens),
        "--output-dir", out_dir,
    ] + FIXED_FLAGS + cfg.cli_args() + extra_flags

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"gen_{cfg.label()}_gpu{gpu_id}.log")
    logger.info("[GPU %d] launching generation for %s -> %s", gpu_id, cfg.label(), out_dir)
    logger.info("[GPU %d] cmd: %s", gpu_id, " ".join(cmd))
    f = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f
    proc._out_dir = out_dir
    proc._cfg = cfg
    proc._gpu_id = gpu_id
    return proc


def stream_process_logs(log_path: str, last_pos: int, prefix: str) -> int:
    if not os.path.exists(log_path):
        return last_pos
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(last_pos)
            lines = f.readlines()
            for line in lines:
                line_str = line.strip()
                if line_str:
                    logger.info("%s %s", prefix, line_str)
            return f.seek(0, 2)
    except Exception as e:
        logger.debug("Error streaming logs: %s", e)
        return last_pos


def run_generation_round_parallel(configs: List[HPConfig], gpu_ids: List[int], repo_root: str,
                                   num_prompts: int, max_tokens: int, extra_flags: List[str],
                                   log_dir: str, poll_s: int = 5) -> Dict[str, str]:
    out_dirs = {}
    config_queue = list(configs)
    active_jobs = []

    while config_queue or active_jobs:
        busy_gpus = {job["gpu_id"] for job in active_jobs}
        free_gpus = [g for g in gpu_ids if g not in busy_gpus]

        while free_gpus and config_queue:
            gpu = free_gpus.pop(0)
            cfg = config_queue.pop(0)
            proc = launch_generation(cfg, gpu, repo_root, num_prompts, max_tokens, extra_flags, log_dir)
            log_path = os.path.join(log_dir, f"gen_{cfg.label()}_gpu{gpu}.log")
            prefix = f"[{cfg.label()} - GPU {gpu}]"
            active_jobs.append({
                "proc": proc,
                "gpu_id": gpu,
                "cfg": cfg,
                "log_path": log_path,
                "prefix": prefix,
                "last_pos": 0,
                "t_start": time.time()
            })
            logger.info("Launched %s on GPU %d (remaining queue: %d)", cfg.label(), gpu, len(config_queue))

        time.sleep(poll_s)
        still_active = []
        for job in active_jobs:
            p = job["proc"]
            ret = p.poll()
            job["last_pos"] = stream_process_logs(job["log_path"], job["last_pos"], job["prefix"])

            if ret is None:
                still_active.append(job)
                elapsed = int(time.time() - job["t_start"])
                if elapsed % 30 < poll_s:
                    logger.info("%s still running (elapsed %ds)...", job["prefix"], elapsed)
            else:
                p._log_file.close()
                if ret != 0:
                    logger.error("%s generation FAILED (exit %d). See log for details.", job["prefix"], ret)
                else:
                    logger.info("%s completed successfully.", job["prefix"])
                    out_dirs[job["cfg"].label()] = p._out_dir

        active_jobs = still_active

    return out_dirs


def _normalize(X: np.ndarray) -> np.ndarray:
    X_norm = np.zeros_like(X)
    for i, name in enumerate(HP_NAMES):
        _, lo, hi, _, _ = SEARCH_SPACE[name]
        X_norm[:, i] = (X[:, i] - lo) / (hi - lo)
    return X_norm


def _denormalize_point(x_norm: np.ndarray) -> Dict[str, float]:
    vals = {}
    for i, name in enumerate(HP_NAMES):
        _, lo, hi, dtype, _ = SEARCH_SPACE[name]
        val = lo + x_norm[i] * (hi - lo)
        vals[name] = int(round(val)) if dtype == "int" else float(val)
    return vals


def propose_next_batch_skopt(X_obs: np.ndarray, y_obs: np.ndarray, n_proposals: int, seed: int
                             ) -> Tuple[List[Dict[str, float]], object]:
    from skopt import Optimizer
    from skopt.space import Real, Integer

    dimensions = []
    for name in HP_NAMES:
        _, lo, hi, dtype, _ = SEARCH_SPACE[name]
        if dtype == "int":
            dimensions.append(Integer(int(lo), int(hi), name=name))
        else:
            dimensions.append(Real(float(lo), float(hi), name=name))

    opt = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",
        random_state=seed,
        n_initial_points=0,
    )
    
    # Fit historical observations
    Xi = []
    for row in X_obs:
        pt = []
        for i, name in enumerate(HP_NAMES):
            _, _, _, dtype, _ = SEARCH_SPACE[name]
            pt.append(int(round(row[i])) if dtype == "int" else float(row[i]))
        Xi.append(pt)

    # Minimize -y since skopt is a minimizer
    yi = (-y_obs).tolist()
    opt.tell(Xi, yi)

    # Ask for batch
    points = opt.ask(n_points=n_proposals)
    proposals = []
    for pt in points:
        vals = {}
        for name, val in zip(HP_NAMES, pt):
            _, _, _, dtype, _ = SEARCH_SPACE[name]
            vals[name] = int(round(val)) if dtype == "int" else float(val)
        proposals.append(vals)

    def surrogate_predict(Xs):
        # Xs is normalized [0,1]
        Xs_raw = []
        for row in Xs:
            pt = []
            for i, name in enumerate(HP_NAMES):
                _, lo, hi, _, _ = SEARCH_SPACE[name]
                pt.append(lo + row[i] * (hi - lo))
            Xs_raw.append(pt)
        mu, sigma = opt.models[-1].predict(Xs_raw, return_std=True)
        # return maximized objective prediction (so negate mu)
        return -mu, sigma

    return proposals, {"predict": surrogate_predict}


def propose_next_batch_builtin(X_obs: np.ndarray, y_obs: np.ndarray, n_proposals: int, seed: int,
                                n_candidates: int = 1500) -> Tuple[List[Dict[str, float]], object]:
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


def recommend_num_rounds(n_dims: int, batch_size: int) -> int:
    return int(np.ceil((10 * n_dims) / batch_size))


def upload_snapshot_to_hf(local_dir: str, repo_id: str, token: str, commit_message: str):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        logger.info("Uploading local directory %s to HF hub repo %s...", local_dir, repo_id)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        logger.info("HF Upload Successful.")
    except Exception as e:
        logger.error("HF Upload Failed: %s", e)


def make_plots(records: List[dict], plot_dir: str, surrogate=None):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(records)
    if df.empty:
        logger.error("No records to plot.")
        return
    df.to_csv(os.path.join(plot_dir, "all_observations.csv"), index=False)

    metrics = [m for m in ["overlap_score", "blade_reward", "override_rate", "positive_rate", "avg_step_tokens"] if m in df.columns]

    unique_rounds = sorted(df["round"].unique())
    tab10_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    round_colors = {r: tab10_colors[i % len(tab10_colors)] for i, r in enumerate(unique_rounds)}
    colors = df["round"].map(round_colors)

    fig, axes = plt.subplots(len(HP_NAMES), len(metrics),
                              figsize=(3.2 * len(metrics), 2.6 * len(HP_NAMES)),
                              squeeze=False)
    for i, hp in enumerate(HP_NAMES):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
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
    fig.suptitle("Hyperparameter effect on each truthfulness metric",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(os.path.join(plot_dir, "hp_effects.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if "objective" in df.columns:
        df_sorted = df.sort_values(["round", "cfg_label"]).reset_index(drop=True)
        best_so_far = df_sorted["objective"].cummax()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(df_sorted) + 1), best_so_far, marker="o", color="#2F6690")
        
        accum = 0
        for r in unique_rounds[:-1]:
            accum += len(df_sorted[df_sorted["round"] == r])
            ax.axvline(accum + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            
        ax.set_xlabel("Configuration index (evaluation order)")
        ax.set_ylabel("Best objective so far (overlap_score)")
        ax.set_title("Bayesian Optimization Convergence (Truthfulness)")
        ax.legend(["best so far", "round boundary"])
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
    ax.set_title("Hyperparameter <-> Metric Correlation (Truthfulness)")
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
                ax.set_ylabel("GP-predicted objective (overlap_score)", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle("GP Partial Dependence (other hyperparameters held at range midpoint)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "partial_dependence.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Plots written to %s", plot_dir)


def evaluate_configs(configs: List[HPConfig], gpu_ids: List[int], repo_root: str,
                      log_dir: str, num_prompts: int, max_tokens: int,
                      extra_flags: List[str]) -> List[dict]:
    gen_out_dirs = run_generation_round_parallel(
        configs, gpu_ids, repo_root, num_prompts, max_tokens, extra_flags, log_dir
    )

    records = []
    for cfg in configs:
        out_dir = gen_out_dirs.get(cfg.label())
        if not out_dir:
            logger.error("Skipping %s -- generation output unavailable.", cfg.label())
            continue

        summary_path = os.path.join(out_dir, "gsi_truthfulness_benchmark_summary.json")
        if not os.path.exists(summary_path):
            logger.error("Skipping %s -- no summary file found at %s.", cfg.label(), summary_path)
            continue

        try:
            with open(summary_path) as f:
                data = json.load(f)
            
            results = data.get("results", {})
            strat_results = results.get("elo_swiss_mode_b", {})
            
            overlap_score = float(strat_results.get("avg_overlap_score", 0.0))
            blade_reward = float(strat_results.get("avg_blade_reward", 0.0))
            override_rate = float(strat_results.get("avg_override_rate", 0.0))
            positive_rate = float(strat_results.get("positive_rate", 0.0))
            avg_step_tokens = float(strat_results.get("avg_step_tokens", 0.0)) if strat_results.get("avg_step_tokens") is not None else 0.0
            
            metrics = {
                "overlap_score": overlap_score,
                "blade_reward": blade_reward,
                "override_rate": override_rate,
                "positive_rate": positive_rate,
                "avg_step_tokens": avg_step_tokens,
            }
            
            rec = {
                "cfg_label": cfg.label(),
                "round": cfg.round_idx,
                **cfg.values,
                **metrics,
                "objective": overlap_score
            }
            records.append(rec)
            logger.info("[%s] objective (overlap)=%.4f  metrics=%s", cfg.label(), overlap_score, metrics)
        except Exception as e:
            logger.error("Error reading metrics for %s: %s", cfg.label(), e)

    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=".", help="Path to the Swiss-Knife repo root.")
    p.add_argument("--num-configs", type=int, default=8,
                    help="Configs per round.")
    p.add_argument("--gpu-ids", default="0,1,2,3",
                    help="Comma-separated physical GPU ids to distribute configs across.")
    p.add_argument("--num-prompts", type=int, default=15, help="Prompts/questions per config.")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=None,
                    help="Where to write runs.")
    p.add_argument("--extra-flag", action="append", default=[],
                    help="Extra raw CLI flag to pass through to the benchmark script.")
    p.add_argument("--skip-round0", action="store_true",
                    help="Skip round 0 and load an existing round0 observation CSV.")
    p.add_argument("--round0-csv", default=None,
                    help="Path to a previously-saved all_observations.csv-style CSV.")
    p.add_argument("--num-rounds", type=int, default=None,
                    help="Total number of rounds to run.")
    p.add_argument("--min-expected-improvement", type=float, default=0.005,
                    help="Round-level pruning threshold.")
    p.add_argument("--hf-repo-id", default=None,
                    help="Hugging Face Hub dataset repo to upload results.")
    p.add_argument("--hf-token", default=None,
                    help="HF Hub token.")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    output_root = args.output_root or repo_root
    log_dir = os.path.join(output_root, "runs", "bayes_search_truthfulness", "logs")
    plot_dir = os.path.join(output_root, "runs", "bayes_search_truthfulness", "plots")
    os.makedirs(log_dir, exist_ok=True)

    t_wall_start = time.time()

    all_records = []
    start_round = 0

    if args.skip_round0:
        import pandas as pd
        assert args.round0_csv, "--round0-csv is required with --skip-round0"
        df0 = pd.read_csv(args.round0_csv)
        all_records = df0.to_dict("records")
        logger.info("Loaded %d historical records from %s.", len(all_records), args.round0_csv)
        if "round" in df0.columns and len(all_records) > 0:
            start_round = int(df0["round"].max()) + 1
        else:
            start_round = 1
        logger.info("Resuming optimization from Round %d.", start_round)

    num_rounds = args.num_rounds or recommend_num_rounds(len(HP_NAMES), args.num_configs)
    logger.info("Total optimization budget: %d rounds.", num_rounds)

    try:
        for rnd in range(start_round, num_rounds):
            logger.info("=" * 60)
            logger.info("STARTING ROUND %d", rnd)
            logger.info("=" * 60)

            if rnd == 0:
                round_configs = build_round0_configs(args.num_configs, seed=args.seed)
                logger.info("Round 0 Seeding: evaluating %d configs", len(round_configs))
            else:
                X_obs = np.array([[r[n] for n in HP_NAMES] for r in all_records], dtype=float)
                y_obs = np.array([r["objective"] for r in all_records], dtype=float)

                logger.info("Fitting Bayesian optimizer on %d observations...", len(all_records))
                proposals, surrogate, ei_values = propose_next_batch(X_obs, y_obs, args.num_configs, seed=args.seed + rnd)

                if args.min_expected_improvement > 0:
                    max_ei = max(ei_values)
                    logger.info("Expected Improvement values for proposals: %s (max: %.4f)", ei_values, max_ei)
                    if max_ei < args.min_expected_improvement:
                        logger.info("Max Expected Improvement (%.4f) below threshold (%.4f). Stopping early.",
                                    max_ei, args.min_expected_improvement)
                        break
                else:
                    logger.info("Expected Improvement values for proposals: %s", ei_values)

                round_configs = [HPConfig(cfg_id=f"cfg{i}", round_idx=rnd, values=v) for i, v in enumerate(proposals)]
                logger.info("Round %d proposed configs: %s", rnd, [c.values for c in round_configs])

            round_records = evaluate_configs(
                round_configs, gpu_ids, repo_root, log_dir,
                args.num_prompts, args.max_tokens, args.extra_flag
            )

            if not round_records:
                logger.error("Round %d returned no successful evaluations. Aborting loop.", rnd)
                break

            all_records.extend(round_records)

            os.makedirs(plot_dir, exist_ok=True)
            import pandas as pd
            pd.DataFrame(round_records).to_csv(os.path.join(plot_dir, f"round{rnd}_observations.csv"), index=False)
            
            # Save both locally in plots and in evaluation folder for standard reference
            all_df = pd.DataFrame(all_records)
            all_df.to_csv(os.path.join(plot_dir, "all_observations.csv"), index=False)
            all_df.to_csv(os.path.join(repo_root, "evaluation", "all_observations_truthfulness.csv"), index=False)

            try:
                X_all = np.array([[r[n] for n in HP_NAMES] for r in all_records], dtype=float)
                y_all = np.array([r["objective"] for r in all_records], dtype=float)
                _, surrogate_final = propose_next_batch_builtin(X_all, y_all, n_proposals=1, seed=args.seed)
            except Exception as e:
                logger.warning("Could not fit intermediate surrogate: %s", e)
                surrogate_final = None

            make_plots(all_records, plot_dir, surrogate=surrogate_final)

            best = max(all_records, key=lambda r: r["objective"])
            with open(os.path.join(plot_dir, "best_config.json"), "w") as f:
                json.dump(best, f, indent=2)

            if args.hf_repo_id:
                upload_snapshot_to_hf(
                    local_dir=os.path.join(output_root, "runs", "bayes_search_truthfulness"),
                    repo_id=args.hf_repo_id,
                    token=args.hf_token,
                    commit_message=f"Bayesian search truthfulness snapshot update after Round {rnd}"
                )

    finally:
        pass

    if all_records:
        best = max(all_records, key=lambda r: r["objective"])
        elapsed_h = (time.time() - t_wall_start) / 3600.0
        print("\n" + "=" * 78)
        print("OPTIMIZATION DONE. Summary:")
        print(f"  Total configs evaluated   : {len(all_records)}")
        print(f"  Best objective found      : {best['objective']:.4f}  (cfg={best['cfg_label']}, round={best['round']})")
        print(f"  Best hyperparameters      : { {n: best[n] for n in HP_NAMES} }")
        print(f"  Plots & CSVs directory    : {plot_dir}")
        print(f"  Total wall-clock time     : {elapsed_h:.2f} hours")
        print("=" * 78)
    else:
        logger.error("No configs were successfully evaluated.")


if __name__ == "__main__":
    main()
