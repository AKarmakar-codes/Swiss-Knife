"""
Swiss Knife → Tribunal Converter + Evaluator
============================================

Converts per-strategy JSON files from runs/stochastic_strategies_knockout_benchmark/
into the .jsonl format expected by the Tribunal judge (github.com/samarthraina/tribunal).

Then optionally runs Tribunal and plots results.

Usage:
    # Step 1: Convert JSONs → .jsonl
    python evaluation/prepare_tribunal_eval.py --mode convert \\
        --input-dir runs/stochastic_strategies_knockout_benchmark \\
        --output-dir tribunal/inputs

    # Step 2: Plot existing tribunal results (after running tribunal separately)
    python evaluation/prepare_tribunal_eval.py --mode plot \\
        --inputs-dir tribunal/inputs \\
        --results-dir tribunal/eval_results \\
        --plot-dir runs/tribunal_plots
"""

import os
import sys
import json
import glob
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion logic: our JSON → tribunal .jsonl
# ─────────────────────────────────────────────────────────────────────────────

def convert(input_dir: str, output_dir: str):
    """Convert per-strategy JSON files → one .jsonl per strategy for tribunal."""
    os.makedirs(output_dir, exist_ok=True)

    json_paths = glob.glob(os.path.join(input_dir, "*_results.json"))
    if not json_paths:
        logger.warning("No *_results.json files found in %s", input_dir)
        return

    for src in json_paths:
        filename = os.path.basename(src)
        strategy_name = filename.replace("_results.json", "")
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

# Known pretty-labels; any strategy not listed here gets a title-cased fallback.
_PRETTY_LABELS = {
    "baseline_argmax_harmlessness": "Baseline\n(Argmax)",
    "baseline_adapter":             "Baseline\n(Adapter)",
    "baseline_adapter_softmax":     "Baseline\n(Adapter Softmax)",
    "stochastic_knockout_dropout":  "Stochastic\nDropout",
    "stochastic_knockout_proj":     "Stochastic\nProj",
    "stochastic_knockout_subsample":"Stochastic\nSubsample",
    "gsi_softmax":                  "GSI SBoN",
    "gsi_pairwise":                 "GSI Pairwise",
    "gsi_swiss":                    "Swiss",
    "gsi_elo":                      "Swiss+Elo",
    "swiss":                        "Swiss",
    "elo_swiss":                    "Swiss+Elo",
    "gsi_gumbel":                   "GSI Gumbel",
}

_COLORS = [
    "#4C72B0", "#DD8452", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#64B5CD", "#6AAB6E",
]


def _pretty(name: str) -> str:
    """Return a human-readable label for a strategy name."""
    return _PRETTY_LABELS.get(name, name.replace("_", "\n").title())


def _discover_strategies(inputs_dir: str, results_dir: str) -> list[str]:
    """
    Return an ordered list of strategy names that have BOTH:
      • a .jsonl file in inputs_dir
      • a corresponding *_eval.csv in results_dir

    The order follows the alphabetical order of the .jsonl files so that
    results are reproducible regardless of filesystem ordering.
    """
    jsonl_files = sorted(glob.glob(os.path.join(inputs_dir, "*.jsonl")))
    if not jsonl_files:
        logger.error("No .jsonl files found in %s", inputs_dir)
        return []

    discovered = []
    for jf in jsonl_files:
        strategy = os.path.splitext(os.path.basename(jf))[0]
        eval_csv = os.path.join(results_dir, f"{strategy}_eval.csv")
        if os.path.exists(eval_csv):
            discovered.append(strategy)
        else:
            logger.warning("No eval CSV found for '%s' (expected %s) — skipping", strategy, eval_csv)

    if not discovered:
        logger.error(
            "No strategies could be matched between '%s' and '%s'.",
            inputs_dir, results_dir,
        )
    else:
        logger.info("Discovered %d strategies: %s", len(discovered), discovered)

    return discovered


def plot(results_dir: str, plot_dir: str, inputs_dir: str):
    """Read tribunal's model_summary.csv and per-model CSVs and produce charts."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("pandas and matplotlib are required for plotting. Install with: pip install pandas matplotlib")
        sys.exit(1)

    summary_path  = os.path.join(results_dir, "model_summary.csv")
    combined_path = os.path.join(results_dir, "combined_results.csv")

    if not os.path.exists(summary_path):
        logger.error("model_summary.csv not found at %s — run tribunal first.", results_dir)
        sys.exit(1)

    os.makedirs(plot_dir, exist_ok=True)

    # ── Discover which strategies to plot ────────────────────────────────────
    strategies = _discover_strategies(inputs_dir, results_dir)
    if not strategies:
        sys.exit(1)

    palette = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(strategies)}

    # ── Load & filter model_summary.csv ──────────────────────────────────────
    summary = pd.read_csv(summary_path)
    logger.info("Loaded model_summary.csv with columns: %s", list(summary.columns))

    name_col = next((c for c in summary.columns if "model" in c.lower()), summary.columns[0])
    summary = summary.rename(columns={name_col: "strategy"})

    before = len(summary)
    summary = summary[summary["strategy"].isin(strategies)].copy()
    summary["strategy"] = pd.Categorical(summary["strategy"], categories=strategies, ordered=True)
    summary = summary.sort_values("strategy").reset_index(drop=True)
    if len(summary) < before:
        logger.info("Filtered out %d non-Swiss-Knife row(s) from model_summary.csv", before - len(summary))

    summary["label"] = summary["strategy"].astype(str).map(_pretty)

    quality_metrics = ["response_quality", "relevance", "helpfulness"]
    safety_metrics  = ["toxicity", "harmfulness", "refusal"]
    all_metrics     = quality_metrics + safety_metrics

    available = [m for m in all_metrics if m in summary.columns]

    # ── 1. Per-rubric bar chart ───────────────────────────────────────────────
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(3 * len(available), 5), sharey=False)
        if len(available) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available):
            vals   = summary[metric].values
            strats = summary["label"].values
            bars   = ax.bar(strats, vals,
                            color=[palette[s] for s in summary["strategy"]],
                            width=0.6, edgecolor="white")
            ax.set_title(metric.replace("_", "\n"), fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(strats)))
            ax.set_xticklabels(strats, fontsize=6, rotation=15, ha="right")
            ax.tick_params(axis="y", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)

        fig.suptitle("Swiss Knife — Per-Rubric Scores (Tribunal Judge)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        per_rubric_path = os.path.join(plot_dir, "per_rubric_scores.png")
        fig.savefig(per_rubric_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", per_rubric_path)

    # ── 2. Safety vs Quality scatter ─────────────────────────────────────────
    q_cols = [c for c in quality_metrics if c in summary.columns]
    s_cols = [c for c in ["toxicity", "harmfulness"] if c in summary.columns]

    if q_cols and s_cols:
        summary["quality_mean"] = summary[q_cols].mean(axis=1)
        summary["safety_score"] = 1 - summary[s_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(7, 6))
        for _, row in summary.iterrows():
            color = palette[row["strategy"]]
            ax.scatter(row["quality_mean"], row["safety_score"], s=200,
                       color=color, zorder=3, edgecolors="white", linewidths=1.5)
            ax.annotate(row["label"], (row["quality_mean"], row["safety_score"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        ax.set_xlabel("Response Quality (mean of quality rubrics)", fontsize=11)
        ax.set_ylabel("Safety Score  (1 − mean(toxicity, harmfulness))", fontsize=11)
        ax.set_title("Safety vs Quality Trade-off Plane\n(Swiss Knife Strategies)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(0.02, 0.97, "Safe\nLow Quality",  transform=ax.transAxes, fontsize=8, color="gray", va="top")
        ax.text(0.75, 0.97, "Safe\nHigh Quality", transform=ax.transAxes, fontsize=8, color="gray", va="top")
        ax.text(0.02, 0.03, "Unsafe\nLow Quality",  transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
        ax.text(0.75, 0.03, "Unsafe\nHigh Quality", transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

        scatter_path = os.path.join(plot_dir, "safety_vs_quality.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", scatter_path)

    # ── 3. Radar chart ────────────────────────────────────────────────────────
    radar_metrics = [m for m in ["response_quality", "relevance", "helpfulness",
                                  "refusal", "harmfulness", "toxicity"] if m in summary.columns]
    if len(radar_metrics) >= 3:
        import numpy as np
        N      = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for _, row in summary.iterrows():
            vals  = [row[m] for m in radar_metrics]
            vals += vals[:1]
            color = palette[row["strategy"]]
            ax.plot(angles, vals, "o-", linewidth=2, color=color, label=row["label"])
            ax.fill(angles, vals, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in radar_metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.set_title("All-Metrics Radar\n(Swiss Knife Strategies)", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
        ax.spines["polar"].set_visible(False)

        radar_path = os.path.join(plot_dir, "radar_all_metrics.png")
        fig.savefig(radar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", radar_path)

    # ── 4. Per-response quality distribution (box plots) ─────────────────────
    if os.path.exists(combined_path) and q_cols:
        combined  = pd.read_csv(combined_path)
        model_col = next((c for c in combined.columns if "model" in c.lower()), None)
        if model_col:
            combined = combined[combined[model_col].isin(strategies)].copy()
            combined["quality_mean"] = combined[[c for c in q_cols if c in combined.columns]].mean(axis=1)
            combined["label"]        = combined[model_col].map(_pretty)

            strat_order = [_pretty(s) for s in strategies if _pretty(s) in combined["label"].unique()]
            if strat_order:
                fig, ax = plt.subplots(figsize=(max(9, 2 * len(strat_order)), 5))
                data_grouped = [combined[combined["label"] == lbl]["quality_mean"].dropna().values
                                for lbl in strat_order]
                bp = ax.boxplot(data_grouped, patch_artist=True,
                                medianprops=dict(color="white", linewidth=2))
                ax.set_xticks(range(1, len(strat_order) + 1))
                ax.set_xticklabels(strat_order, fontsize=8)
                for patch, lbl in zip(bp["boxes"], strat_order):
                    strat_name = next((s for s in strategies if _pretty(s) == lbl), lbl)
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
    p.add_argument("--task", choices=["harmlessness", "helpfulness"], default="harmlessness",
                   help="Evaluation task to run. Adjusts default input/output/results/plots directories.")
    p.add_argument("--mode", choices=["convert", "plot", "both"], default="convert",
                   help="'convert': JSON→.jsonl only; 'plot': plot tribunal results; 'both': do both")
    p.add_argument("--input-dir",  default=None,
                   help="Directory with per-strategy *_results.json files (for --mode convert). "
                        "Defaults: runs/stochastic_strategies_knockout_benchmark for harmlessness, "
                        "runs/gsi_helpfulness_benchmark/qwen3B for helpfulness.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the .jsonl files for tribunal. "
                        "Defaults: tribunal/inputs/harmlessness or tribunal/inputs/helpfulness.")
    p.add_argument("--inputs-dir", default=None,
                   help="Directory containing tribunal .jsonl input files (for --mode plot). "
                        "Defaults: tribunal/inputs/harmlessness or tribunal/inputs/helpfulness.")
    p.add_argument("--results-dir", default=None,
                   help="Where tribunal wrote its CSV outputs (for --mode plot). "
                        "Defaults: tribunal/eval_results/harmlessness or tribunal/eval_results/helpfulness.")
    p.add_argument("--plot-dir", default=None,
                   help="Where to save the generated plots. "
                        "Defaults: runs/tribunal_plots/harmlessness or runs/tribunal_plots/helpfulness.")
    return p.parse_args()


def main():
    args = parse_args()
    task = args.task

    # Resolve defaults based on task
    input_dir = args.input_dir
    if input_dir is None:
        if task == "helpfulness":
            input_dir = "runs/gsi_helpfulness_benchmark/qwen3B"
        else:
            input_dir = "runs/stochastic_strategies_knockout_benchmark"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"tribunal/inputs/{task}"

    inputs_dir = args.inputs_dir
    if inputs_dir is None:
        inputs_dir = f"tribunal/inputs/{task}"

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = f"tribunal/eval_results/{task}"

    plot_dir = args.plot_dir
    if plot_dir is None:
        plot_dir = f"runs/tribunal_plots/{task}"

    if args.mode in ("convert", "both"):
        convert(input_dir, output_dir)
    if args.mode in ("plot", "both"):
        # For 'both', the output-dir of convert becomes the inputs-dir for plot
        final_inputs_dir = output_dir if args.mode == "both" else inputs_dir
        plot(results_dir, plot_dir, final_inputs_dir)


if __name__ == "__main__":
    main()
