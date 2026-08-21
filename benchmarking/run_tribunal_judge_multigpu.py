"""
run_tribunal_judge_multigpu.py — Multi-GPU parallel judge scoring for tribunal inputs
=======================================================================================
Replaces the (nonexistent) "tribunal/run_tribunal_eval.py --parallel" step referenced
in RUN_PIPELINE.md. There is no --parallel flag anywhere in tribunal/tribunal/run_eval.py,
and no run_tribunal_eval.py in the repo — this script IS the parallel runner.

WHY A NEW SCRIPT INSTEAD OF JUST LOOPING run_eval.py MANUALLY ON EACH GPU
---------------------------------------------------------------------------
tribunal.pipeline.run() ends by globbing every "*_eval.csv" file in its --output
folder and overwriting model_summary.csv / summary.csv / report/index.html in that
same folder. If N parallel processes are pointed at the SAME --output folder, they
race on those three files — whichever process finishes last silently clobbers the
aggregate summary from every other process (the per-file *_eval.csv scores
themselves are fine; only the aggregation step is unsafe to share).

This script avoids that entirely, following the same judge-per-GPU pattern already
used in evaluation/parameter_search_optimized.py:
  1. Split the .jsonl files in --input-dir round-robin across --gpu-ids.
  2. Each GPU gets its OWN symlinked input dir, OWN output dir, and OWN private
     vLLM judge server on port (--port-base + gpu_id). Nothing is shared.
  3. Each GPU's tribunal.run_eval subprocess runs against only its own slice,
     writing only into its own private output dir — no cross-worker races.
  4. Once every GPU is done, all workers' judge servers are stopped, and the
     final merge step runs ONCE, single-threaded: copy every worker's *_eval.csv
     into the final --output-dir (raises loudly on any filename collision instead
     of silently overwriting), then call pipeline.combine_results /
     generate_summary / aggregate_by_model / build_report exactly once, safely.

USAGE
-----
    python benchmarking/run_tribunal_judge_multigpu.py \
        --input-dir tribunal/inputs/hhh_pareto \
        --output-dir tribunal/eval_results/hhh_pareto \
        --gpu-ids 0 1 2 3 4 5 6 7 \
        --workers-per-gpu 8

Place this file at <repo_root>/benchmarking/run_tribunal_judge_multigpu.py
(it locates repo_root relative to its own path).
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("multigpu_judge")

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIBUNAL_DIR = REPO_ROOT / "tribunal"
JUDGE_API_KEY = "EMPTY"


# ── Judge server lifecycle (same approach as evaluation/parameter_search_optimized.py) ──

def start_judge_server(gpu_id: int, port: int, judge_model: str, log_dir: Path,
                        gpu_mem_util: float = 0.90) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", judge_model,
        "--dtype", "half",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--port", str(port),
        "--api-key", JUDGE_API_KEY,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"judge_gpu{gpu_id}_port{port}.log"
    logger.info("[GPU %d] starting judge server on port %d (%s)", gpu_id, port, judge_model)
    f = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f  # keep a ref so we can close it later
    return proc


def wait_for_server(port: int, timeout_s: int = 900, poll_s: int = 5) -> bool:
    url = f"http://localhost:{port}/v1/models"
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
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


# ── File sharding ─────────────────────────────────────────────────────────────

def split_round_robin(files, n_buckets):
    buckets = [[] for _ in range(n_buckets)]
    for i, f in enumerate(sorted(files)):
        buckets[i % n_buckets].append(f)
    return buckets


def stage_input_dir(files, work_dir: Path) -> Path:
    """Symlink only this worker's files into a private dir, so
    tribunal.data.resolve_input_files (glob '*.jsonl') only sees its slice."""
    in_dir = work_dir / "input"
    if in_dir.exists():
        shutil.rmtree(in_dir)
    in_dir.mkdir(parents=True)
    for f in files:
        f = Path(f).resolve()
        os.symlink(f, in_dir / f.name)
    return in_dir


# ── Per-GPU worker: judge server up -> run_eval on its own slice -> judge server down ──

def run_gpu_worker(gpu_id: int, files, args, work_root: Path) -> Path:
    """Returns this worker's private output dir (containing *_eval.csv files)."""
    work_dir = work_root / f"gpu{gpu_id}"
    in_dir = stage_input_dir(files, work_dir)
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    port = args.port_base + gpu_id
    log_dir = work_root / "logs"
    judge_proc = start_judge_server(gpu_id, port, args.judge_model, log_dir, args.gpu_mem_util)

    try:
        logger.info("[GPU %d] waiting for judge server on port %d ...", gpu_id, port)
        if not wait_for_server(port, timeout_s=args.judge_startup_timeout):
            raise RuntimeError(f"[GPU {gpu_id}] judge server on port {port} never became ready.")

        logger.info("[GPU %d] scoring %d file(s): %s", gpu_id, len(files),
                    ", ".join(Path(f).name for f in files))

        cmd = [
            sys.executable, "-m", "tribunal.run_eval",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--judge-url", f"http://localhost:{port}/v1",
            "--max-workers", str(args.workers_per_gpu),
        ]
        if args.no_detoxify:
            cmd.append("--no-detoxify")
        if args.include_humour:
            cmd.append("--include-humour")

        eval_log = log_dir / f"run_eval_gpu{gpu_id}.log"
        with open(eval_log, "w") as f:
            ret = subprocess.run(cmd, cwd=str(TRIBUNAL_DIR), env=os.environ.copy(),
                                  stdout=f, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            raise RuntimeError(
                f"[GPU {gpu_id}] tribunal.run_eval failed (exit {ret.returncode}); "
                f"see {eval_log}"
            )

        n_written = len(list(out_dir.glob("*_eval.csv")))
        logger.info("[GPU %d] done. %d/%d files scored -> %s",
                    gpu_id, n_written, len(files), out_dir)
        return out_dir

    finally:
        logger.info("[GPU %d] stopping judge server...", gpu_id)
        stop_judge_server(judge_proc)
        time.sleep(5)  # let VRAM drain before this GPU is reused for anything else


# ── Final merge: runs ONCE, single-threaded, after every GPU worker has exited ──

def merge_worker_outputs(worker_out_dirs, final_output_dir: Path, tribunal_config: dict):
    sys.path.insert(0, str(TRIBUNAL_DIR))
    from tribunal.pipeline import combine_results, generate_summary, aggregate_by_model, build_report

    final_output_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for wdir in worker_out_dirs:
        for csv_file in glob.glob(str(wdir / "*_eval.csv")):
            dest = final_output_dir / Path(csv_file).name
            if dest.exists():
                raise RuntimeError(
                    f"Filename collision merging judge outputs: {dest} already exists. "
                    f"Two input .jsonl files produced the same model_name stem across "
                    f"different GPU shards — rename the source files in tribunal/inputs/ "
                    f"so every <method>__<config>.jsonl stem is unique, then re-run."
                )
            shutil.copy2(csv_file, dest)
            n_copied += 1

    logger.info("Merged %d per-file *_eval.csv results into %s", n_copied, final_output_dir)

    combined = combine_results(str(final_output_dir))
    if combined is None or combined.empty:
        logger.warning("No results to summarize after merge.")
        return

    generate_summary(combined, str(final_output_dir), tribunal_config)
    agg = aggregate_by_model(combined)
    agg.round(4).to_csv(final_output_dir / "model_summary.csv", index=False)
    build_report(agg, str(final_output_dir / "report"))

    logger.info("Final merged report: %s", final_output_dir / "report" / "index.html")
    logger.info("Final merged tables: %s, %s",
                final_output_dir / "model_summary.csv", final_output_dir / "summary.csv")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU parallel tribunal judge scoring.")
    p.add_argument("--input-dir", required=True, help="Folder of .jsonl files to judge.")
    p.add_argument("--output-dir", required=True, help="Final merged output folder.")
    p.add_argument("--gpu-ids", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--port-base", type=int, default=8000)
    p.add_argument("--workers-per-gpu", type=int, default=8,
                   help="Parallel HTTP threads PER GPU hitting that GPU's own judge "
                        "server (passed through as --max-workers to tribunal.run_eval).")
    p.add_argument("--judge-startup-timeout", type=int, default=900)
    p.add_argument("--no-detoxify", action="store_true")
    p.add_argument("--include-humour", action="store_true")
    p.add_argument("--work-dir", default=None,
                   help="Scratch dir for per-GPU input/output staging "
                        "(default: <output-dir>/_multigpu_work).")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_root = Path(args.work_dir).resolve() if args.work_dir else output_dir / "_multigpu_work"

    files = sorted(str(f) for f in input_dir.glob("*.jsonl"))
    if not files:
        logger.error("No .jsonl files found in %s", input_dir)
        return

    buckets = split_round_robin(files, len(args.gpu_ids))
    active = [(gid, fb) for gid, fb in zip(args.gpu_ids, buckets) if fb]
    logger.info("Sharding %d files across %d GPU(s): %s",
                len(files), len(active), {gid: len(fb) for gid, fb in active})

    worker_out_dirs = []
    failures = []
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {
            pool.submit(run_gpu_worker, gid, fb, args, work_root): gid
            for gid, fb in active
        }
        for fut in as_completed(futures):
            gid = futures[fut]
            try:
                worker_out_dirs.append(fut.result())
            except Exception as e:
                logger.error("[GPU %d] worker failed: %s", gid, e)
                failures.append(gid)

    if failures:
        logger.error(
            "GPU worker(s) %s failed — their files were NOT scored. "
            "Check logs under %s/logs/ then re-run this script (files already "
            "scored by successful workers are untouched; you can re-point "
            "--input-dir at just the missing files if you want to avoid re-judging "
            "everything).", failures, work_root
        )

    if not worker_out_dirs:
        logger.error("No GPU worker produced output — nothing to merge.")
        return

    tribunal_config = {
        "include_humour": args.include_humour,
        "include_honesty": True,
    }
    merge_worker_outputs(worker_out_dirs, output_dir, tribunal_config)

    if failures:
        logger.warning("Merge complete, but %d GPU(s) failed — results are PARTIAL.", len(failures))
        sys.exit(1)
    else:
        logger.info("All GPUs succeeded. Merge complete.")


if __name__ == "__main__":
    main()