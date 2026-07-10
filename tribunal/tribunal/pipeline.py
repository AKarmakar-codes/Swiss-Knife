"""Scoring orchestration: read every model file in the input folder, score each
response, tag it with the model (the filename), then aggregate and build the
report.

One file equals one model. The filename (without extension) is the model label.
If a run was sharded across several files, concatenate them into one file first.
"""

import gc
import glob
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

from .config import CONFIG, ALL_METRICS, SAFETY_METRICS
from .data import read_jsonl, validate_record, resolve_input_files
from .metrics import build_metrics, DetoxifyCalculator
from .judge import VLLMJudge
from .report import aggregate_by_model, build_report


def setup_logging(output_folder: str) -> logging.Logger:
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(
        output_folder, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("tribunal")


def evaluate_single_record(
    record: Dict[str, Any],
    metrics: Dict[str, GEval],
    detoxify: Optional[DetoxifyCalculator],
) -> Dict[str, Any]:
    result = {
        "id": record["id"],
        "prompt": record["prompt"],
        "response": record["response"],
    }
    for name in metrics:
        result[f"{name}_score"] = None
        result[f"{name}_reason"] = None
        result[f"{name}_abstained"] = False

    if detoxify is not None:
        d_score, d_reason = detoxify.score(record["response"])
        result["toxicity_detoxify_score"] = d_score
        result["toxicity_detoxify_reason"] = d_reason

    tc = LLMTestCase(input=record["prompt"], actual_output=record["response"])

    for name, metric in metrics.items():
        try:
            metric.measure(tc)
            score = metric.score
            if score is None:
                result[f"{name}_abstained"] = True
            result[f"{name}_score"] = score
            result[f"{name}_reason"] = getattr(metric, "reason", "")
        except Exception as e:
            result[f"{name}_abstained"] = True
            result[f"{name}_reason"] = f"error: {e}"
            logging.error(f"Metric {name} failed on record {record['id']}: {e}")

    return result


def run_evaluation_single_file(
    input_path: str,
    output_path: str,
    model_name: str,
    metrics: Dict[str, GEval],
    detoxify: Optional[DetoxifyCalculator],
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Score one model's file. Resumes per record and flushes periodically."""
    filename = Path(input_path).name
    records = read_jsonl(input_path)
    if sample_size:
        records = records[:sample_size]

    processed_ids = set()
    file_exists = os.path.exists(output_path)
    if file_exists:
        try:
            processed_ids = set(pd.read_csv(output_path, usecols=["id"])["id"].astype(str))
            logging.info(f"Resuming {filename}: {len(processed_ids)} already scored")
        except Exception as e:
            logging.warning(f"Could not read existing results for resume: {e}")

    stats = {"total": len(records), "valid": 0, "invalid": {}}
    todo = []
    for r in records:
        ok, reason = validate_record(r)
        if ok:
            stats["valid"] += 1
            if str(r["id"]) not in processed_ids:
                todo.append(r)
        else:
            stats["invalid"][reason] = stats["invalid"].get(reason, 0) + 1

    if not todo:
        logging.info(f"Nothing new to score in {filename}")
        return stats

    print(f"  scoring {len(todo)} records ({len(records) - len(todo)} skipped or cached)")
    write_mode = "a" if file_exists else "w"
    write_header = not file_exists
    buffer = []
    for idx, record in enumerate(tqdm(todo, desc=model_name)):
        row = {"model": model_name, **evaluate_single_record(record, metrics, detoxify)}
        buffer.append(row)
        if (idx + 1) % CONFIG["save_every"] == 0 or (idx + 1) == len(todo):
            pd.DataFrame(buffer).to_csv(output_path, mode=write_mode, header=write_header, index=False)
            buffer, write_mode, write_header = [], "a", False
            gc.collect()

    return stats


def combine_results(output_folder: str) -> Optional[pd.DataFrame]:
    """Concatenate every per-model score CSV into one table."""
    files = [
        f for f in glob.glob(os.path.join(output_folder, "*_eval.csv"))
        if "combined" not in f and "summary" not in f
    ]
    if not files:
        return None
    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    combined.to_csv(os.path.join(output_folder, "combined_results.csv"), index=False)
    return combined


def generate_summary(combined: pd.DataFrame, output_folder: str) -> pd.DataFrame:
    """Per-model, per-metric statistics, including abstention counts."""
    metric_names = list(ALL_METRICS)
    if CONFIG["use_detoxify"]:
        metric_names.append("toxicity_detoxify")

    rows = []
    for model, g in combined.groupby("model"):
        for name in metric_names:
            col = f"{name}_score"
            if col not in g.columns:
                continue
            scores = pd.to_numeric(g[col], errors="coerce").dropna()
            abst_col = f"{name}_abstained"
            abstained = int(g[abst_col].sum()) if abst_col in g.columns else 0
            is_safety = name in SAFETY_METRICS or name.startswith("toxicity")
            rows.append({
                "model": model,
                "metric": name,
                "group": "safety" if is_safety else "quality",
                "n_judged": len(scores),
                "n_abstained": abstained,
                "mean": round(scores.mean(), 4) if len(scores) else np.nan,
                "median": round(scores.median(), 4) if len(scores) else np.nan,
                "std": round(scores.std(), 4) if len(scores) else np.nan,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_folder, "summary.csv"), index=False)
    return summary


def run(config: dict = CONFIG) -> None:
    """Score every model file in the input folder, then build the report."""
    out = config["output_folder"]
    os.makedirs(out, exist_ok=True)
    setup_logging(out)

    files = resolve_input_files(config["input_path"])
    if not files:
        print(f"No .jsonl files found in {config['input_path']}.")
        print("Drop one file per model into that folder (filename = model name) and rerun.")
        return

    print(f"input:  {config['input_path']} ({len(files)} model file/s)")
    print(f"output: {out}")
    print(f"judge:  {config['judge_model']} @ {config['vllm_url']}")
    print(f"detoxify cross-check: {config['use_detoxify']}")

    try:
        judge = VLLMJudge()
    except Exception as e:
        print(f"Could not connect to the judge server: {e}")
        print("Start it first with: python serve_judge.py")
        return

    metrics = build_metrics(judge)
    detoxify = DetoxifyCalculator() if config["use_detoxify"] else None

    for i, input_path in enumerate(files, 1):
        model_name = Path(input_path).stem
        output_path = os.path.join(out, f"{model_name}_eval.csv")
        print(f"\n[{i}/{len(files)}] {model_name}")
        start = time.time()
        try:
            stats = run_evaluation_single_file(
                input_path, output_path, model_name, metrics, detoxify, config["sample_size"]
            )
            print(f"  done in {time.time() - start:.1f}s "
                  f"(valid {stats['valid']}/{stats['total']}, invalid {stats['invalid']})")
        except Exception as e:
            logging.error(f"Failed on {model_name}: {e}")
            import traceback
            traceback.print_exc()

    combined = combine_results(out)
    if combined is None or combined.empty:
        print("\nNo results to summarize.")
        return

    generate_summary(combined, out)
    agg = aggregate_by_model(combined)
    agg.round(4).to_csv(os.path.join(out, "model_summary.csv"), index=False)
    report_dir = os.path.join(out, "report")
    build_report(agg, report_dir)

    print(f"\nScored {len(agg)} model/s.")
    print(f"Report:  {os.path.join(report_dir, 'index.html')}")
    print(f"Tables:  {os.path.join(out, 'model_summary.csv')}, {os.path.join(out, 'summary.csv')}")
