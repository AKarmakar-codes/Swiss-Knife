"""
Swiss Knife → Tribunal Converter + Evaluator
============================================

Converts per-strategy JSON files from runs/stochastic_strategies_knockout_benchmark/
into the .jsonl format expected by the Tribunal judge (github.com/samarthraina/tribunal).

Then optionally runs Tribunal and plots results.

Usage:
    # Step 1: Convert JSONs → .jsonl
    python evaluation/prepare_tribunal_eval.py --mode convert \
        --input-dir runs/stochastic_strategies_knockout_benchmark \
        --output-dir tribunal/inputs

    # Step 2: Plot existing tribunal results (after running tribunal separately)
    python evaluation/prepare_tribunal_eval.py --mode plot \
        --results-dir tribunal/eval_results \
        --plot-dir runs/tribunal_plots
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion logic: our JSON → tribunal .jsonl
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_FILES = {
    "baseline_argmax_harmlessness": "baseline_argmax_harmlessness_results.json",
    "stochastic_knockout_dropout":  "stochastic_knockout_dropout_results.json",
    "stochastic_knockout_proj":     "stochastic_knockout_proj_results.json",
    "stochastic_knockout_subsample":"stochastic_knockout_subsample_results.json",
}


def convert(input_dir: str, output_dir: str):
    """Convert per-strategy JSON files → one .jsonl per strategy for tribunal."""
    os.makedirs(output_dir, exist_ok=True)

    for strategy_name, filename in STRATEGY_FILES.items():
        src = os.path.join(input_dir, filename)
        if not os.path.exists(src):
            logger.warning("Missing: %s — skipping", src)
            continue

        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        responses = data.get("responses", [])
        if not responses:
            logger.warning("%s has no responses — skipping", strategy_name)
            continue

        dst = os.path.join(output_dir, f"{strategy_name}.jsonl")
        written = 0
        with open(dst, "w", encoding="utf-8") as out:
            for resp in responses:
                # Skip entries that errored during generation
                if resp.get("error") or not resp.get("generated", "").strip():
                    logger.debug("Skipping empty/errored entry idx=%d", resp.get("prompt_idx", -1))
                    continue

                record = {
                    "id":       resp["prompt_idx"],
                    "prompt":   resp["prompt"].strip(),
                    "response": resp["generated"].strip(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        logger.info("✓ %s → %s  (%d records)", strategy_name, dst, written)

    logger.info("Conversion complete. Drop the .jsonl files from '%s' into tribunal's inputs/ folder.", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting logic: tribunal CSV results → graphs
# ─────────────────────────────────────────────────────────────────────────────

def plot(results_dir: str, plot_dir: str):
    """Read tribunal's model_summary.csv and per-model CSVs and produce charts."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.error("pandas and matplotlib are required for plotting. Install with: pip install pandas matplotlib")
        sys.exit(1)

    summary_path = os.path.join(results_dir, "model_summary.csv")
    combined_path = os.path.join(results_dir, "combined_results.csv")

    if not os.path.exists(summary_path):
        logger.error("model_summary.csv not found at %s — run tribunal first.", results_dir)
        sys.exit(1)

    os.makedirs(plot_dir, exist_ok=True)

    summary = pd.read_csv(summary_path)
    logger.info("Loaded model_summary.csv with columns: %s", list(summary.columns))

    # Normalise strategy name column
    name_col = next((c for c in summary.columns if "model" in c.lower()), summary.columns[0])
    summary = summary.rename(columns={name_col: "strategy"})

    # Pretty labels
    LABELS = {
        "baseline_argmax_harmlessness": "Baseline\n(Argmax)",
        "stochastic_knockout_dropout":  "Stochastic\nDropout",
        "stochastic_knockout_proj":     "Stochastic\nProj",
        "stochastic_knockout_subsample":"Stochastic\nSubsample",
    }

    # ── Drop any extra models that were in tribunal's sample inputs ────────────
    KNOWN_STRATEGIES = list(LABELS.keys())
    before = len(summary)
    summary = summary[summary["strategy"].isin(KNOWN_STRATEGIES)].copy()
    summary["strategy"] = pd.Categorical(summary["strategy"], categories=KNOWN_STRATEGIES, ordered=True)
    summary = summary.sort_values("strategy").reset_index(drop=True)
    if len(summary) < before:
        logger.info("Filtered out %d non-Swiss-Knife model(s) from model_summary.csv", before - len(summary))

    summary["label"] = summary["strategy"].map(LABELS).fillna(summary["strategy"])

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    palette = {s: COLORS[i] for i, s in enumerate(KNOWN_STRATEGIES)}


    quality_metrics = ["response_quality", "relevance", "helpfulness"]
    safety_metrics  = ["toxicity", "harmfulness", "refusal"]
    all_metrics     = quality_metrics + safety_metrics

    available = [m for m in all_metrics if m in summary.columns]

    # ── 1. Per-rubric bar chart ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(available), figsize=(3 * len(available), 5), sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        vals  = summary[metric].values
        strats = summary["label"].values
        bars = ax.bar(strats, vals, color=[palette[s] for s in summary["strategy"]], width=0.6, edgecolor="white")
        ax.set_title(metric.replace("_", "\n"), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(strats, fontsize=8, rotation=15, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Swiss Knife Knockout — Per-Rubric Scores (Tribunal Judge)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    per_rubric_path = os.path.join(plot_dir, "per_rubric_scores.png")
    fig.savefig(per_rubric_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", per_rubric_path)

    # ── 2. Safety vs Quality scatter (tribunal's trade-off plane) ─────────────
    q_cols = [c for c in quality_metrics if c in summary.columns]
    s_cols = [c for c in ["toxicity", "harmfulness"] if c in summary.columns]

    if q_cols and s_cols:
        summary["quality_mean"]  = summary[q_cols].mean(axis=1)
        summary["safety_score"]  = 1 - summary[s_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(7, 6))
        for _, row in summary.iterrows():
            color = palette[row["strategy"]]
            ax.scatter(row["quality_mean"], row["safety_score"], s=200,
                       color=color, zorder=3, edgecolors="white", linewidths=1.5)
            ax.annotate(row["label"], (row["quality_mean"], row["safety_score"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        ax.set_xlabel("Response Quality (mean of quality rubrics)", fontsize=11)
        ax.set_ylabel("Safety Score  (1 − mean(toxicity, harmfulness))", fontsize=11)
        ax.set_title("Safety vs Quality Trade-off Plane\n(Swiss Knife Knockout Strategies)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Quadrant labels
        ax.text(0.02, 0.97, "Safe\nLow Quality", transform=ax.transAxes, fontsize=8, color="gray", va="top")
        ax.text(0.75, 0.97, "Safe\nHigh Quality", transform=ax.transAxes, fontsize=8, color="gray", va="top")
        ax.text(0.02, 0.03, "Unsafe\nLow Quality", transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
        ax.text(0.75, 0.03, "Unsafe\nHigh Quality", transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

        scatter_path = os.path.join(plot_dir, "safety_vs_quality.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", scatter_path)

    # ── 3. Radar chart ────────────────────────────────────────────────────────
    radar_metrics = [m for m in ["response_quality", "relevance", "helpfulness",
                                  "refusal", "harmfulness", "toxicity"] if m in summary.columns]
    if len(radar_metrics) >= 3:
        N = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for _, row in summary.iterrows():
            vals = [row[m] for m in radar_metrics]
            vals += vals[:1]
            color = palette[row["strategy"]]
            ax.plot(angles, vals, "o-", linewidth=2, color=color, label=row["label"])
            ax.fill(angles, vals, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in radar_metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.set_title("All-Metrics Radar\n(Swiss Knife Knockout Strategies)", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
        ax.spines["polar"].set_visible(False)

        radar_path = os.path.join(plot_dir, "radar_all_metrics.png")
        fig.savefig(radar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", radar_path)

    # ── 4. Per-response score distributions (box plots) ───────────────────────
    if os.path.exists(combined_path):
        combined = pd.read_csv(combined_path)
        model_col = next((c for c in combined.columns if "model" in c.lower()), None)
        if model_col and q_cols:
            combined["quality_mean"] = combined[[c for c in q_cols if c in combined.columns]].mean(axis=1)
            combined["label"] = combined[model_col].map(LABELS).fillna(combined[model_col])

            strat_order = [LABELS.get(s, s) for s in summary["strategy"] if LABELS.get(s, s) in combined["label"].unique()]
            fig, ax = plt.subplots(figsize=(9, 5))
            data_grouped = [combined[combined["label"] == lbl]["quality_mean"].dropna().values
                            for lbl in strat_order]
            bp = ax.boxplot(data_grouped, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            ax.set_xticks(range(1, len(strat_order) + 1))
            ax.set_xticklabels(strat_order)
            for patch, lbl in zip(bp["boxes"], strat_order):
                strat_name = next((k for k, v in LABELS.items() if v == lbl), lbl)
                patch.set_facecolor(palette.get(strat_name, "#888"))

            ax.set_ylabel("Response Quality Score", fontsize=11)
            ax.set_title("Per-Response Quality Distribution by Strategy", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            box_path = os.path.join(plot_dir, "quality_distribution.png")
            fig.savefig(box_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", box_path)

    logger.info("All plots written to: %s", plot_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Convert Swiss Knife results for Tribunal and/or plot scores.")
    p.add_argument("--mode", choices=["convert", "plot", "both"], default="convert",
                   help="'convert': JSON→.jsonl only; 'plot': plot tribunal results; 'both': do both")
    p.add_argument("--input-dir",  default="runs/stochastic_strategies_knockout_benchmark",
                   help="Directory with per-strategy *_results.json files")
    p.add_argument("--output-dir", default="tribunal/inputs",
                   help="Where to write the .jsonl files for tribunal")
    p.add_argument("--results-dir", default="tribunal/eval_results",
                   help="Where tribunal wrote its CSV outputs (for --mode plot)")
    p.add_argument("--plot-dir", default="runs/tribunal_plots",
                   help="Where to save the generated plots")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("convert", "both"):
        convert(args.input_dir, args.output_dir)
    if args.mode in ("plot", "both"):
        plot(args.results_dir, args.plot_dir)


if __name__ == "__main__":
    main()
