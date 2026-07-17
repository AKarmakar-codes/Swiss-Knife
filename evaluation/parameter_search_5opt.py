#!/usr/bin/env python3
"""
Bayesian hyperparameter search over Swiss-Knife's ``elo_swiss_mode_b`` GSI strategy,
tailored for the HELPFULNESS blade and optimizing 5 key hyperparameters:
    - elo_temperature
    - beta
    - w_tournament
    - w_blade
    - uwo_lambda

Fixed parameters:
    - gsi_n = 8
    - elo_rounds = 4
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
logger = logging.getLogger("bayes_search")


SEARCH_SPACE: Dict[str, Tuple[str, float, float, str, float]] = {
    "elo_temperature": ("--elo-temperature", 1.0,  40.0, "float", 15.0),
    "beta":            ("--beta",            0.01, 1.0,  "float", 0.1),
    "w_tournament":    ("--w-tournament",     0.0,  3.0,  "float", 1.0),
    "w_blade":         ("--w-blade",          0.0,  3.0,  "float", 1.0),
    "uwo_lambda":      ("--uwo-lambda",       0.0,  1.0,  "float", 0.5),
    "elo_rounds":      ("--elo-rounds",       2.0,  10.0, "int",   4.0),
}
HP_NAMES = list(SEARCH_SPACE.keys())

FIXED_FLAGS = [
    "--strategies", "elo_swiss_mode_b",
    "--probabilistic",
    "--sigma-mode", "log_ratio_proxy",
    "--gsi-max-step-tokens", "80",
    "--gsi-n", "8",
    "--blade", "helpfulness",
]

STRATEGY_NAME = "elo_swiss_mode_b"

OBJECTIVE_WEIGHTS = {
    "helpfulness": 0.35,
    "relevance": 0.35,
    "response_quality": 0.3,
    "refusal": -0.5,
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


def launch_generation(cfg: HPConfig, gpu_id: int, repo_root: str, num_prompts: int,
                       max_tokens: int, extra_flags: List[str], log_dir: str) -> subprocess.Popen:
    out_dir = os.path.join(repo_root, "runs", "bayes_search", f"round{cfg.round_idx}", cfg.label())
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(repo_root, "evaluation", "benchmark_gsi_strategies_helpfulness.py")

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
            return f.seek(0, 2)  # Return current end of file position
    except Exception as e:
        logger.debug("Error streaming logs: %s", e)
        return last_pos


def run_process_and_stream_logs(cmd: List[str], cwd: str, env: dict, log_path: str, prefix: str, poll_s: float = 2.0) -> int:
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    last_pos = 0
    t_start = time.time()
    try:
        while True:
            ret = proc.poll()
            last_pos = stream_process_logs(log_path, last_pos, prefix)
            if ret is not None:
                return ret
            time.sleep(poll_s)
            elapsed = int(time.time() - t_start)
            if elapsed % 30 < poll_s:
                logger.info("%s still running (elapsed %ds)...", prefix, elapsed)
    finally:
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass


def run_generation_round(configs: List[HPConfig], gpu_ids: List[int], repo_root: str,
                          num_prompts: int, max_tokens: int, extra_flags: List[str],
                          log_dir: str, poll_s: int = 5) -> Dict[str, str]:
    assert len(configs) == len(gpu_ids), (
        f"{len(configs)} configs but {len(gpu_ids)} gpu ids -- must be 1:1 "
        "since each GPU runs exactly one config at a time."
    )
    procs = [
        launch_generation(cfg, gpu, repo_root, num_prompts, max_tokens, extra_flags, log_dir)
        for cfg, gpu in zip(configs, gpu_ids)
    ]
    logger.info("Launched %d generation processes across GPUs %s. Waiting...", len(procs), gpu_ids)

    proc_states = []
    for p in procs:
        log_path = os.path.join(log_dir, f"gen_{p._cfg.label()}_gpu{p._gpu_id}.log")
        prefix = f"[{p._cfg.label()} - GPU {p._gpu_id}]"
        proc_states.append({
            "proc": p,
            "log_path": log_path,
            "prefix": prefix,
            "last_pos": 0,
            "t_start": time.time()
        })

    out_dirs = {}
    pending = list(proc_states)
    while pending:
        time.sleep(poll_s)
        still_pending = []
        for state in pending:
            p = state["proc"]
            ret = p.poll()
            state["last_pos"] = stream_process_logs(state["log_path"], state["last_pos"], state["prefix"])
            if ret is None:
                still_pending.append(state)
                elapsed = int(time.time() - state["t_start"])
                if elapsed % 30 < poll_s:
                    logger.info("%s still running (elapsed %ds)...", state["prefix"], elapsed)
                continue
            p._log_file.close()
            if ret != 0:
                logger.error(
                    "%s generation FAILED (exit %d). See log for details.",
                    state["prefix"], ret,
                )
            else:
                logger.info("%s completed successfully.", state["prefix"])
                out_dirs[p._cfg.label()] = p._out_dir
        pending = still_pending
        if pending:
            logger.info("%d/%d generation jobs still running...", len(pending), len(procs))
    return out_dirs


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
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
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


def run_tribunal_for_config(gpu_id: int, port: int, repo_root: str,
                             jsonl_path: str, model_name: str, results_dir: str,
                             judge_model: str, log_dir: str,
                             sample_size: Optional[int] = None) -> bool:
    tmp_input_dir = os.path.join(results_dir, "_tribunal_input_tmp")
    os.makedirs(tmp_input_dir, exist_ok=True)
    staged = os.path.join(tmp_input_dir, os.path.basename(jsonl_path))
    shutil.copy(jsonl_path, staged)

    proc = start_judge_server(gpu_id, port, repo_root, log_dir, judge_model)
    try:
        ready = wait_for_server(port, timeout_s=900)
        if not ready:
            logger.error("[GPU %d] judge server on port %d never became ready.", gpu_id, port)
            return False

        cmd = [
            sys.executable, "-m", "tribunal.run_eval",
            "--task", "helpfulness",
            "--input", tmp_input_dir,
            "--output", results_dir,
            "--judge-url", f"http://localhost:{port}/v1",
        ]
        if sample_size is not None:
            cmd += ["--sample-size", str(sample_size)]
        tribunal_cwd = os.path.join(repo_root, "tribunal")
        log_path = os.path.join(log_dir, f"tribunal_{model_name}_gpu{gpu_id}.log")
        logger.info("[GPU %d] running tribunal for %s (cwd=%s)", gpu_id, model_name, tribunal_cwd)
        prefix = f"[tribunal - {model_name}]"
        
        ret = run_process_and_stream_logs(cmd, tribunal_cwd, os.environ.copy(), log_path, prefix)
        
        if ret != 0:
            logger.error("[GPU %d] tribunal run FAILED for %s (exit %d). See %s",
                          gpu_id, model_name, ret, log_path)
            return False
        return True
    finally:
        stop_judge_server(proc)
        shutil.rmtree(tmp_input_dir, ignore_errors=True)


def run_tribunal_round(configs: List[HPConfig], gpu_ids: List[int], repo_root: str,
                        gen_out_dirs: Dict[str, str], tribunal_root: str,
                        judge_model: str, log_dir: str,
                        max_parallel_judges: Optional[int] = None) -> Dict[str, str]:
    import concurrent.futures

    max_parallel = max_parallel_judges or len(configs)
    results_dirs = {}
    jobs = []
    for cfg, gpu in zip(configs, gpu_ids):
        gen_dir = gen_out_dirs.get(cfg.label())
        if not gen_dir:
            logger.warning("No generation output for %s -- skipping tribunal.", cfg.label())
            continue
        tribunal_inputs_dir = os.path.join(tribunal_root, "inputs", f"round{cfg.round_idx}", cfg.label())
        jsonl_path = convert_json_to_jsonl(gen_dir, tribunal_inputs_dir, model_name=cfg.label())
        if jsonl_path is None:
            continue
        results_dir = os.path.join(tribunal_root, "eval_results", f"round{cfg.round_idx}", cfg.label())
        os.makedirs(results_dir, exist_ok=True)
        port = 8000 + gpu
        jobs.append((cfg, gpu, port, jsonl_path, results_dir))

    def _run(job):
        cfg, gpu, port, jsonl_path, results_dir = job
        ok = run_tribunal_for_config(
            gpu_id=gpu, port=port, repo_root=repo_root, jsonl_path=jsonl_path,
            model_name=cfg.label(), results_dir=results_dir,
            judge_model=judge_model, log_dir=log_dir,
        )
        return cfg.label(), results_dir, ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        for label, results_dir, ok in ex.map(_run, jobs):
            if ok:
                results_dirs[label] = results_dir
            else:
                logger.error("Tribunal scoring failed for %s", label)
    return results_dirs


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

    opt = Optimizer(dims, base_estimator="GP", acq_func="EI", random_state=seed,
                     n_initial_points=0)
    X_list = [[float(v) if SEARCH_SPACE[n][3] == "float" else int(round(v))
               for n, v in zip(HP_NAMES, row)] for row in X_obs]
    opt.tell(X_list, (-y_obs).tolist())

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


def recommend_num_rounds(n_dims: int, configs_per_round: int) -> int:
    target_total_evals = 10 * n_dims
    remaining = max(target_total_evals - configs_per_round, 0)
    extra_rounds = int(np.ceil(remaining / configs_per_round))
    return max(2, 1 + extra_rounds)


def upload_snapshot_to_hf(local_dir: str, repo_id: str, token: Optional[str],
                           commit_message: str) -> None:
    if not repo_id:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed -- skipping HF snapshot upload.")
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
        logger.info("Uploaded snapshot of %s -> hf.co/datasets/%s (%s)", local_dir, repo_id, commit_message)
    except Exception as e:
        logger.warning("HF snapshot upload failed: %s", e)


def stage_combined_tribunal_dirs(records: List[dict], tribunal_root: str,
                                 group_name: str) -> Optional[Tuple[str, str]]:
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
            logger.warning(
                "Skipping %s in combined tribunal plots -- missing one of jsonl/eval_csv/model_summary.", cfg_label,
            )
            continue

        shutil.copy(src_jsonl, os.path.join(combined_inputs_dir, f"{cfg_label}.jsonl"))
        shutil.copy(src_eval_csv, os.path.join(combined_results_dir, f"{cfg_label}_eval.csv"))
        summary_rows.append(pd.read_csv(src_summary_csv))
        combined_rows.append(pd.read_csv(src_eval_csv))
        staged_any = True

    if not staged_any:
        logger.error("No configs had complete tribunal output for group '%s' -- skipping its comparison plots.",
                      group_name)
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
    logger.info("Tribunal-style comparison plots for '%s' written to %s", group_name, out_dir)


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

    metrics = [m for m in ["response_quality", "relevance", "helpfulness",
                            "toxicity", "harmfulness", "refusal"] if m in df.columns]

    # Generate colors dynamically for each round
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
    fig.suptitle("Hyperparameter effect on each tribunal metric",
                 fontsize=11, fontweight="bold")
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
        ax.set_xlabel("Quality axis  (mean of response_quality, relevance, helpfulness)")
        ax.set_ylabel("Safety axis  (1 - mean(toxicity, harmfulness))")
        ax.set_title("Pareto Frontier — Quality vs Safety across all tested configs")
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "pareto_frontier.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "objective" in df.columns:
        df_sorted = df.sort_values(["round", "cfg_label"]).reset_index(drop=True)
        best_so_far = df_sorted["objective"].cummax()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(df_sorted) + 1), best_so_far, marker="o", color="#2F6690")
        
        # Draw vertical lines for round boundaries
        accum = 0
        for r in unique_rounds[:-1]:
            accum += len(df_sorted[df_sorted["round"] == r])
            ax.axvline(accum + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            
        ax.set_xlabel("Configuration index (evaluation order)")
        ax.set_ylabel("Best objective so far")
        ax.set_title("Bayesian Optimization Convergence")
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

    logger.info("Plots written to %s", plot_dir)


def run_generation_round_sequential(configs: List[HPConfig], gpu_id: int, repo_root: str,
                                    num_prompts: int, max_tokens: int, extra_flags: List[str],
                                    log_dir: str, poll_s: int = 5) -> Dict[str, str]:
    out_dirs = {}
    for i, cfg in enumerate(configs):
        logger.info("=" * 60)
        logger.info("Sequential generation on GPU %d for config %d/%d (%s)",
                    gpu_id, i + 1, len(configs), cfg.label())
        logger.info("=" * 60)
        
        p = launch_generation(cfg, gpu_id, repo_root, num_prompts, max_tokens, extra_flags, log_dir)
        log_path = os.path.join(log_dir, f"gen_{cfg.label()}_gpu{gpu_id}.log")
        prefix = f"[{cfg.label()} - GPU {gpu_id}]"
        last_pos = 0
        t_start = time.time()
        
        while True:
            ret = p.poll()
            last_pos = stream_process_logs(log_path, last_pos, prefix)
            if ret is not None:
                break
            time.sleep(poll_s)
            elapsed = int(time.time() - t_start)
            if elapsed % 30 < poll_s:
                logger.info("%s still running (elapsed %ds)...", prefix, elapsed)
                
        p._log_file.close()
        if ret != 0:
            logger.error("%s generation FAILED (exit %d)", prefix, ret)
        else:
            logger.info("%s completed successfully.", prefix)
            out_dirs[cfg.label()] = p._out_dir
    return out_dirs


def run_tribunal_round_persistent(configs: List[HPConfig], port: int, repo_root: str,
                                  gen_out_dirs: Dict[str, str], tribunal_root: str,
                                  log_dir: str) -> Dict[str, str]:
    results_dirs = {}
    for cfg in configs:
        gen_dir = gen_out_dirs.get(cfg.label())
        if not gen_dir:
            continue
        tribunal_inputs_dir = os.path.join(tribunal_root, "inputs", f"round{cfg.round_idx}", cfg.label())
        jsonl_path = convert_json_to_jsonl(gen_dir, tribunal_inputs_dir, model_name=cfg.label())
        if jsonl_path is None:
            continue
        results_dir = os.path.join(tribunal_root, "eval_results", f"round{cfg.round_idx}", cfg.label())
        os.makedirs(results_dir, exist_ok=True)
        
        tmp_input_dir = os.path.join(results_dir, "_tribunal_input_tmp")
        os.makedirs(tmp_input_dir, exist_ok=True)
        staged = os.path.join(tmp_input_dir, os.path.basename(jsonl_path))
        shutil.copy(jsonl_path, staged)
        
        cmd = [
            sys.executable, "-m", "tribunal.run_eval",
            "--task", "helpfulness",
            "--input", tmp_input_dir,
            "--output", results_dir,
            "--judge-url", f"http://localhost:{port}/v1",
        ]
        tribunal_cwd = os.path.join(repo_root, "tribunal")
        log_path = os.path.join(log_dir, f"tribunal_{cfg.label()}_persistent_port{port}.log")
        logger.info("Running tribunal for %s using persistent judge on port %d", cfg.label(), port)
        prefix = f"[tribunal - {cfg.label()}]"
        
        ret = run_process_and_stream_logs(cmd, tribunal_cwd, os.environ.copy(), log_path, prefix)
        
        shutil.rmtree(tmp_input_dir, ignore_errors=True)
        if ret == 0:
            results_dirs[cfg.label()] = results_dir
        else:
            logger.error("Tribunal scoring failed for %s (exit %d)", cfg.label(), ret)
    return results_dirs


def evaluate_configs(configs: List[HPConfig], gpu_ids: List[int], repo_root: str,
                      tribunal_root: str, log_dir: str, num_prompts: int, max_tokens: int,
                      extra_flags: List[str], judge_model: str,
                      max_parallel_judges: Optional[int],
                      gpu_split: bool = False,
                      persistent_judge_port: Optional[int] = None) -> List[dict]:
    if gpu_split:
        generator_gpu = gpu_ids[0]
        gen_out_dirs = run_generation_round_sequential(
            configs, generator_gpu, repo_root, num_prompts, max_tokens, extra_flags, log_dir
        )
        tribunal_dirs = run_tribunal_round_persistent(
            configs, persistent_judge_port, repo_root, gen_out_dirs, tribunal_root, log_dir
        )
    else:
        gen_out_dirs = run_generation_round(
            configs, gpu_ids, repo_root, num_prompts, max_tokens, extra_flags, log_dir,
        )
        tribunal_dirs = run_tribunal_round(
            configs, gpu_ids, repo_root, gen_out_dirs, tribunal_root, judge_model, log_dir,
            max_parallel_judges=max_parallel_judges,
        )

    records = []
    for cfg in configs:
        results_dir = tribunal_dirs.get(cfg.label())
        if not results_dir:
            logger.error("Skipping %s in observation table -- tribunal scoring unavailable.", cfg.label())
            continue
        metrics = read_metrics(results_dir, cfg.label())
        if metrics is None:
            continue
        obj = scalar_objective(metrics)
        rec = {"cfg_label": cfg.label(), "round": cfg.round_idx, **cfg.values, **metrics, "objective": obj}
        records.append(rec)
        logger.info("[%s] objective=%.4f  metrics=%s", cfg.label(), obj, metrics)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=".", help="Path to the Swiss-Knife-main repo root.")
    p.add_argument("--num-configs", type=int, default=8,
                    help="Configs per round (also = number of GPUs used in parallel). Default 8.")
    p.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7",
                    help="Comma-separated physical GPU ids to use, 1 per config.")
    p.add_argument("--gpu-split", action="store_true",
                    help="Enable 2-GPU split mode: GPU 0 is generator, GPU 1 is persistent judge.")
    p.add_argument("--num-prompts", type=int, default=15, help="Prompts per config.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--max-parallel-judges", type=int, default=None,
                    help="Cap concurrent judge servers.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=None,
                    help="Where to write runs/ and tribunal outputs.")
    p.add_argument("--extra-flag", action="append", default=[],
                    help="Extra raw CLI flag to pass through to the benchmark script.")
    p.add_argument("--skip-round0", action="store_true",
                    help="Skip round 0 and load an existing round0 observation CSV.")
    p.add_argument("--round0-csv", default=None,
                    help="Path to a previously-saved all_observations.csv-style CSV.")
    p.add_argument("--num-rounds", type=int, default=None,
                    help="Total number of rounds to run.")
    p.add_argument("--min-expected-improvement", type=float, default=0.01,
                    help="Round-level pruning threshold.")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM GPU memory utilization fraction (0.0 to 1.0). "
                         "Defaults to 0.45 if sharing a single GPU, otherwise 0.90.")
    p.add_argument("--hf-repo-id", default=None,
                    help="Hugging Face Hub dataset repo to upload results.")
    p.add_argument("--hf-token", default=None,
                    help="HF Hub token.")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    gpu_split = args.gpu_split
    if not gpu_split and len(gpu_ids) == 2:
        logger.info("Exactly 2 GPUs specified. Automatically enabling --gpu-split mode.")
        gpu_split = True

    if gpu_split:
        if len(gpu_ids) != 2:
            logger.error("2-GPU split mode requires exactly 2 GPU IDs (e.g. --gpu-ids 0,1). Got: %s", gpu_ids)
            sys.exit(1)
        if gpu_ids[0] == gpu_ids[1]:
            logger.info("GPU Split Mode enabled in SHARED mode on GPU %d (96GB/large VRAM setup).", gpu_ids[0])
        else:
            logger.info("GPU Split Mode enabled. Generator GPU: %d, Judge GPU: %d", gpu_ids[0], gpu_ids[1])
    else:
        # Standard parallel mode check
        if len(gpu_ids) != args.num_configs:
            logger.warning("num-configs=%d but %d gpu-ids given; truncating/padding gpu-ids to match.",
                            args.num_configs, len(gpu_ids))
            gpu_ids = (gpu_ids * ((args.num_configs // len(gpu_ids)) + 1))[: args.num_configs]

    output_root = args.output_root or repo_root
    log_dir = os.path.join(output_root, "runs", "bayes_search", "logs")
    tribunal_root = os.path.join(output_root, "tribunal", "bayes_search")
    plot_dir = os.path.join(output_root, "runs", "bayes_search", "plots")
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

    judge_proc = None
    persistent_port = None
    if gpu_split:
        judge_gpu = gpu_ids[1]
        persistent_port = 8000 + judge_gpu
        logger.info("Starting persistent judge server on GPU %d (port %d)", judge_gpu, persistent_port)
        
        # Calculate dynamic GPU memory utilization to avoid OOM when sharing the same GPU
        if args.gpu_memory_utilization is not None:
            gpu_mem_util = args.gpu_memory_utilization
        else:
            gpu_mem_util = 0.45 if gpu_ids[0] == gpu_ids[1] else 0.90
        
        logger.info("vLLM GPU memory utilization set to %.2f", gpu_mem_util)
        judge_proc = start_judge_server(
            judge_gpu, persistent_port, repo_root, log_dir, args.judge_model, gpu_mem_util=gpu_mem_util
        )
        ready = wait_for_server(persistent_port, timeout_s=900)
        if not ready:
            logger.error("Persistent judge server on port %d never became ready. Aborting.", persistent_port)
            stop_judge_server(judge_proc)
            sys.exit(1)

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
                round_configs, gpu_ids, repo_root, tribunal_root, log_dir,
                args.num_prompts, args.max_tokens, args.extra_flag, args.judge_model,
                args.max_parallel_judges,
                gpu_split=gpu_split,
                persistent_judge_port=persistent_port
            )

            if not round_records:
                logger.error("Round %d returned no successful evaluations. Aborting loop.", rnd)
                break

            all_records.extend(round_records)

            os.makedirs(plot_dir, exist_ok=True)
            import pandas as pd
            pd.DataFrame(round_records).to_csv(os.path.join(plot_dir, f"round{rnd}_observations.csv"), index=False)
            pd.DataFrame(all_records).to_csv(os.path.join(plot_dir, "all_observations.csv"), index=False)

            try:
                X_all = np.array([[r[n] for n in HP_NAMES] for r in all_records], dtype=float)
                y_all = np.array([r["objective"] for r in all_records], dtype=float)
                _, surrogate_final = propose_next_batch_builtin(X_all, y_all, n_proposals=1, seed=args.seed)
            except Exception as e:
                logger.warning("Could not fit intermediate surrogate: %s", e)
                surrogate_final = None

            make_plots(all_records, plot_dir, surrogate=surrogate_final)
            make_tribunal_comparison_plots(all_records, tribunal_root, repo_root, plot_dir, "all")

            best = max(all_records, key=lambda r: r["objective"])
            with open(os.path.join(plot_dir, "best_config.json"), "w") as f:
                json.dump(best, f, indent=2)

            if args.hf_repo_id:
                upload_snapshot_to_hf(
                    local_dir=os.path.join(output_root, "runs", "bayes_search"),
                    repo_id=args.hf_repo_id,
                    token=args.hf_token,
                    commit_message=f"Bayesian search snapshot update after Round {rnd}"
                )

    finally:
        if judge_proc is not None:
            logger.info("Stopping persistent judge server...")
            stop_judge_server(judge_proc)

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
