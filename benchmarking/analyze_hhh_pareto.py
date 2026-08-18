"""
analyze_hhh_pareto.py — 3D HHH Pareto Results Analyzer
========================================================

Reads Tribunal G-Eval results (model_summary.csv) and computes:
  1. Per-method mean Quality, Safety, Honesty scores
  2. 3-objective Harmonic Mean F1
  3. Schott's Spacing Metric (Δ) — measures Pareto frontier uniformity
  4. Paired bootstrap 95% CIs vs. Swiss-Knife on F1

Produces:
  - runs/hhh_pareto/plots/summary_table.csv
  - runs/hhh_pareto/plots/pareto_2d_projections.png

Usage:
  python benchmarking/analyze_hhh_pareto.py \\
    --summary tribunal/eval_results/hhh_pareto/model_summary.csv \\
    --out-dir  runs/hhh_pareto/plots
"""

import os
import sys
import json
import logging
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HHHAnalyzer")

# Tribunal metric column names — adjust if your judge uses different names
COL_QUALITY  = "response_quality"
COL_SAFETY   = "harmfulness"     # lower is better in raw; we invert below
COL_HONESTY  = "truthfulness"
COL_MODEL    = "model"           # e.g. "swiss__w_help100_hone000_harm000"


# ── Core metrics ──────────────────────────────────────────────────────────────

def harmonic_f1(q: float, s: float, h: float, eps: float = 1e-6) -> float:
    """3-objective harmonic mean (all scores assumed ∈ [0,1], higher = better)."""
    vals = [max(eps, q), max(eps, s), max(eps, h)]
    return 3.0 / sum(1.0 / v for v in vals)


def schotts_spacing(points: np.ndarray) -> float:
    """Schott's Spacing Metric Δ.  Lower = more uniformly spread frontier.

    Points shape: (N, 2).  Sorts along first objective.
    """
    if len(points) < 2:
        return 0.0
    pts = points[np.argsort(points[:, 0])]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d_bar = d.mean()
    if d_bar == 0:
        return 0.0
    return float((d[0] + d[-1] + np.abs(d - d_bar).sum()) /
                 (d[0] + d[-1] + len(d) * d_bar))


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 42):
    """Paired bootstrap CI of mean(a - b).  Returns (mean_diff, lo, hi, p_val)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(np.mean(boots <= 0), np.mean(boots >= 0))  # two-sided approx
    return float(diff.mean()), float(lo), float(hi), float(p)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(summary_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)

    if COL_MODEL not in df.columns:
        raise ValueError(f"Expected column '{COL_MODEL}' in {summary_csv}")

    # Extract method name from model label (e.g. "swiss__w_help100..." → "swiss")
    df["method"] = df[COL_MODEL].str.split("__").str[0]

    # safety: raw "harmfulness" is lower-is-better → flip to higher-is-better
    if COL_SAFETY in df.columns:
        df["safety_score"] = 1.0 - df[COL_SAFETY].clip(0, 1)
    else:
        logger.warning("Column '%s' not found — safety_score set to 0.", COL_SAFETY)
        df["safety_score"] = 0.0

    df["quality_score"] = df[COL_QUALITY].clip(0, 1) if COL_QUALITY in df.columns else 0.0
    df["honesty_score"]  = df[COL_HONESTY].clip(0, 1) if COL_HONESTY in df.columns else 0.0
    df["f1"] = df.apply(
        lambda r: harmonic_f1(r["quality_score"], r["safety_score"], r["honesty_score"]), axis=1
    )

    methods = df["method"].unique()
    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        q_mean = sub["quality_score"].mean()
        s_mean = sub["safety_score"].mean()
        h_mean = sub["honesty_score"].mean()
        f1_mean = sub["f1"].mean()
        pts = sub[["quality_score", "safety_score"]].dropna().to_numpy()
        delta = schotts_spacing(pts)

        # Load latency & throughput from runs/hhh_pareto/<m>/*.json if available
        latencies, throughputs = [], []
        method_dir = os.path.join("runs", "hhh_pareto", m)
        if os.path.exists(method_dir):
            for fname in os.listdir(method_dir):
                if fname.endswith(".json"):
                    with open(os.path.join(method_dir, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    for r in data.get("responses", []):
                        st = r.get("stats", {})
                        if "total_time_s" in st:
                            latencies.append(st["total_time_s"])
                        if "tokens_per_second" in st:
                            throughputs.append(st["tokens_per_second"])

        avg_lat = round(float(np.mean(latencies)), 3) if latencies else "—"
        avg_tps = round(float(np.mean(throughputs)), 2) if throughputs else "—"

        rows.append({
            "method": m,
            "quality": round(q_mean, 4),
            "safety": round(s_mean, 4),
            "honesty": round(h_mean, 4),
            "f1": round(f1_mean, 4),
            "spacing_delta": round(delta, 4),
            "latency_s": avg_lat,
            "tok_per_sec": avg_tps,
        })

    summary = pd.DataFrame(rows).sort_values("f1", ascending=False)

    # Bootstrap CIs vs Swiss-Knife
    if "swiss" in df["method"].values:
        swiss_f1 = df[df["method"] == "swiss"]["f1"].to_numpy()
        for row in rows:
            m = row["method"]
            if m == "swiss":
                row["ci95"] = "—"
                row["p_val"] = "—"
                continue
            other_f1 = df[df["method"] == m]["f1"].to_numpy()
            n = min(len(swiss_f1), len(other_f1))
            diff, lo, hi, p = bootstrap_ci(swiss_f1[:n], other_f1[:n])
            row["ci95"] = f"[{lo:.3f}, {hi:.3f}]"
            row["p_val"] = f"{p:.3f}"

    out_csv = os.path.join(out_dir, "summary_table.csv")
    summary.to_csv(out_csv, index=False)
    logger.info("Summary table → %s", out_csv)

    print("\n" + "=" * 72)
    print("  3D HHH Pareto Benchmark — Results")
    print("=" * 72)
    print(summary.to_string(index=False))
    print("=" * 72 + "\n")

    # ── 2D Pareto Projection Plots ─────────────────────────────────────────────
    pairs = [
        ("quality_score", "safety_score",  "Helpfulness ↑", "Safety ↑"),
        ("quality_score", "honesty_score",  "Helpfulness ↑", "Honesty ↑"),
        ("safety_score",  "honesty_score",  "Safety ↑",      "Honesty ↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax, (cx, cy, xlabel, ylabel) in zip(axes, pairs):
        for i, m in enumerate(methods):
            sub = df[df["method"] == m]
            ax.scatter(sub[cx], sub[cy], label=m,
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       alpha=0.8, s=60)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("3D HHH Pareto Frontiers — Swiss-Knife vs Baselines", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "pareto_2d_projections.png")
    plt.savefig(plot_path, dpi=300)
    logger.info("Plot saved → %s", plot_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary",  default="tribunal/eval_results/hhh_pareto/model_summary.csv")
    p.add_argument("--out-dir",  default="runs/hhh_pareto/plots")
    args = p.parse_args()
    analyze(args.summary, args.out_dir)


if __name__ == "__main__":
    main()
