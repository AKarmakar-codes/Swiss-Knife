"""
Swiss Knife — Overnight β × elo_temperature Sweep
==================================================

Runs the full pipeline you described end-to-end, unattended:

  1. For every (beta, elo_temperature) pair in a 5x5 grid (25 configs),
     runs benchmark_gsi_strategies_harmlessness_no_th.py for all 3
     strategies (gsi_softmax_no_th, gsi_swiss_no_th, gsi_elo_no_th),
     15 prompts each -> 75 *_results.json files total.
     Up to 6 configs run concurrently, one per GPU (GPUs 1-6; GPU 0 is
     left alone because your vLLM tribunal judge server is on it).
  2. Converts all 75 *_results.json -> *.jsonl for the tribunal.
  3. Runs the tribunal (tribunal/tribunal/run_eval.py) once, pointed at
     the folder of all 75 jsonl files (tribunal is linear/single-process
     and expects to be pointed at a folder, per config.py/run_eval.py).
  4. Reads tribunal/eval_results overnight/model_summary.csv and reports
     the best (beta, elo_temperature) pair per strategy and overall,
     using a composite score = mean(quality metrics) - mean(safety metrics)
     (quality: response_quality, relevance, helpfulness;
      safety: toxicity, harmfulness, refusal -- these are "bad" scores
      per tribunal's own framing in prepare_tribunal_eval.py, so lower is
      better and we subtract them).
  5. Plots graphs from that same model_summary.csv:
       - ONE bar chart per (beta, temperature) config (25 total), each
         showing all 3 strategies side-by-side across the 6 tribunal
         rubrics, saved to runs/overnight_sweep/plots/grid/ with a
         filename tied to that config's own tag (never overwrites another
         config's plot).
       - TWO unified overview plots covering the whole sweep at once:
         a per-strategy heatmap of composite_score over the full
         (beta, temperature) grid, and a safety-vs-quality scatter of all
         75 points colored by strategy. Saved to
         runs/overnight_sweep/plots/ under their own fixed names, which
         is safe because they're written exactly once per sweep run and
         live in a different location than the 25 per-config files.

IMPORTANT CAVEATS -- please read before trusting the "best config" output:
  - elo_temperature ONLY affects the gsi_elo_no_th strategy (confirmed by
    grep: Model_mechanics/config.py, gsi_elo_no_th.py, speculative_generator.py).
    gsi_softmax_no_th and gsi_swiss_no_th do not consume elo_temperature at
    all, so their 5 "temperature" repeats per beta are exact repeats of the
    same generation call modulo whatever randomness --seed doesn't pin down.
    This is expected and NOT a bug in this script -- it mirrors how the
    underlying benchmark script itself splits temperature usage. The
    per-strategy "best config" numbers for softmax/swiss are really just
    reporting beta sensitivity; only the gsi_elo_no_th numbers reflect a
    genuine 2D sweep. The same is true of their heatmap rows in the unified
    overview plot: expect near-identical values across the temperature axis
    for softmax/swiss, and only real variation for elo.
  - 15 prompts x 25 configs x 3 strategies is a small sample per cell.
    Treat "best" as a point estimate, not a statistically significant
    winner -- especially for cells that differ by a few thousandths.
  - This script assumes each GPU has enough VRAM to hold one full copy of
    base model + blade model + drafter model concurrently with the other
    5 GPUs doing the same (i.e. it does NOT do any cross-GPU memory
    sharing -- it is just 6 independent single-GPU worker processes).
  - This script does NOT start the tribunal vLLM judge server -- you said
    that's already running on GPU 0, so it is left untouched. If it is
    not up when tribunal runs, tribunal will fail to connect and this
    script will report that failure rather than silently hanging forever.
  - Config tags (and therefore jsonl/plot filenames) are formatted to 4
    decimal places (e.g. beta1.2500_temp13.7500). With the default ranges
    all 25 tags are distinct, but if you widen --beta-steps/--temp-steps
    enough that two different float values round to the same 4-decimal
    string, they WOULD collide and overwrite each other's files. main()
    asserts uniqueness up front and aborts before running anything if
    that ever happens, rather than silently clobbering results overnight.

Usage:
    cd Swiss-Knife
    nohup python evaluation/run_overnight_sweep.py > overnight_sweep.log 2>&1 &

    # or with custom ranges / GPU list:
    python evaluation/run_overnight_sweep.py \\
        --beta-min 1.0 --beta-max 2.5 --beta-steps 5 \\
        --temp-min 10.0 --temp-max 15.0 --temp-steps 5 \\
        --gpus 1 2 3 4 5 6 \\
        --num-prompts 15

Everything (generation logs, jsonl conversion, tribunal run, final report,
plots) happens in this one process. Check `overnight_sweep.log`,
`runs/overnight_sweep/FINAL_REPORT.md`, and
`runs/overnight_sweep/plots/` when you wake up.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STRATEGIES = ["gsi_softmax_no_th", "gsi_swiss_no_th", "gsi_elo_no_th"]

# Fixed strategy -> color mapping so a given strategy is always the same
# color across all 25 per-config plots and the unified overview.
STRATEGY_COLORS = {
    "gsi_softmax_no_th": "#4C72B0",
    "gsi_swiss_no_th": "#DD8452",
    "gsi_elo_no_th": "#C44E52",
}
STRATEGY_LABELS = {
    "gsi_softmax_no_th": "Softmax",
    "gsi_swiss_no_th": "Swiss",
    "gsi_elo_no_th": "Elo",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("overnight_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def linspace(lo: float, hi: float, n: int):
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 6) for i in range(n)]


def cfg_tag(beta: float, temp: float) -> str:
    """Filesystem-safe tag identifying a (beta, temperature) config."""
    return f"beta{beta:.4f}_temp{temp:.4f}"


def assert_unique_tags(betas: list, temps: list):
    """
    Abort loudly, before any generation happens, if two distinct
    (beta, temp) float pairs would round to the same cfg_tag() string --
    that would make two configs silently share a grid/ directory, a
    tribunal_inputs/*.jsonl filename, and a plots/grid/*.png filename.
    """
    seen = {}
    for b in betas:
        for t in temps:
            tag = cfg_tag(b, t)
            if tag in seen and seen[tag] != (b, t):
                raise ValueError(
                    f"cfg_tag collision: (beta={b}, temp={t}) and "
                    f"(beta={seen[tag][0]}, temp={seen[tag][1]}) both round to "
                    f"'{tag}'. Reduce --beta-steps/--temp-steps or widen the "
                    f"range so all grid points are distinct to 4 decimal places."
                )
            seen[tag] = (b, t)


def run_one_config(
    beta: float,
    temp: float,
    gpu_id: int,
    num_prompts: int,
    max_tokens: int,
    gsi_n: int,
    output_root: str,
    python_bin: str,
) -> dict:
    """
    Run benchmark_gsi_strategies_harmlessness_no_th.py for ALL strategies at
    this (beta, temp) pair, pinned to one physical GPU via CUDA_VISIBLE_DEVICES.
    Writes results into output_root/<tag>/ (one dir per config, containing
    the 3 *_results.json files for that config, named by strategy as the
    script itself names them).
    """
    tag = cfg_tag(beta, temp)
    out_dir = os.path.join(output_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Keep HF/torch from trying to see other GPUs at all inside this process.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        python_bin,
        os.path.join(REPO_ROOT, "evaluation", "benchmark_gsi_strategies_harmlessness_no_th.py"),
        "--strategies", *STRATEGIES,
        "--num-prompts", str(num_prompts),
        "--gsi-n", str(gsi_n),
        "--beta", str(beta),
        "--elo-temperature", str(temp),
        "--max-tokens", str(max_tokens),
        "--blade", "harmlessness",
        "--output-dir", out_dir,
        "--skip-existing",
    ]

    t0 = time.perf_counter()
    logger.info("[GPU %d] START beta=%.4f temp=%.4f -> %s", gpu_id, beta, temp, out_dir)
    with open(log_path, "w") as logf:
        logf.write(f"CMD: {' '.join(cmd)}\nGPU: {gpu_id}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - t0

    ok = proc.returncode == 0
    level = logger.info if ok else logger.error
    level(
        "[GPU %d] %s beta=%.4f temp=%.4f in %.1fs (rc=%d) — log: %s",
        gpu_id, "DONE" if ok else "FAILED", beta, temp, elapsed, proc.returncode, log_path,
    )

    return {
        "beta": beta,
        "temp": temp,
        "gpu": gpu_id,
        "tag": tag,
        "out_dir": out_dir,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log_path": log_path,
    }


def run_grid(
    betas: list,
    temps: list,
    gpus: list,
    num_prompts: int,
    max_tokens: int,
    gsi_n: int,
    output_root: str,
    python_bin: str,
) -> list:
    """
    Runs all len(betas)*len(temps) configs, at most len(gpus) at once,
    each config pinned to one GPU. Returns list of per-config result dicts.
    """
    configs = [(b, t) for b in betas for t in temps]
    logger.info(
        "Scheduling %d configs across %d GPUs (%s)",
        len(configs), len(gpus), gpus,
    )

    results = []
    pending = list(configs)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}  # future -> gpu_id

        def submit_next(gpu_id):
            if not pending:
                return
            beta, temp = pending.pop(0)
            fut = pool.submit(
                run_one_config, beta, temp, gpu_id,
                num_prompts, max_tokens, gsi_n, output_root, python_bin,
            )
            futures[fut] = gpu_id

        # Prime: one task per GPU.
        for gpu_id in gpus:
            submit_next(gpu_id)

        # As each task completes, immediately hand that GPU the next
        # pending config, until nothing is left pending or in flight.
        while futures:
            fut = next(as_completed(list(futures.keys())))
            gpu_id = futures.pop(fut)
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "beta": None, "temp": None, "gpu": gpu_id,
                    "returncode": -1, "error": str(e),
                }
                logger.error("[GPU %d] worker raised: %s\n%s", gpu_id, e, traceback.format_exc())
            results.append(res)
            submit_next(gpu_id)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: convert all *_results.json (across all config dirs) -> tribunal jsonl
# ─────────────────────────────────────────────────────────────────────────────

def convert_all_to_jsonl(grid_root: str, jsonl_root: str) -> int:
    """
    Walks grid_root/<cfg_tag>/<strategy>_results.json and writes
    jsonl_root/<strategy>__<cfg_tag>.jsonl  (unique per strategy+config,
    so the tribunal -- which uses the .jsonl STEM as the model name --
    can distinguish all 75 files).
    Returns number of jsonl files written.
    """
    os.makedirs(jsonl_root, exist_ok=True)
    written = 0

    for cfg_tag_dir in sorted(os.listdir(grid_root)):
        cfg_dir = os.path.join(grid_root, cfg_tag_dir)
        if not os.path.isdir(cfg_dir):
            continue
        for strategy in STRATEGIES:
            src = os.path.join(cfg_dir, f"{strategy}_results.json")
            if not os.path.exists(src):
                logger.warning("Missing %s (config %s failed or was skipped) — no jsonl written", src, cfg_tag_dir)
                continue
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
            responses = data.get("responses", [])
            if not responses:
                logger.warning("%s has no responses — skipping", src)
                continue

            model_name = f"{strategy}__{cfg_tag_dir}"
            dst = os.path.join(jsonl_root, f"{model_name}.jsonl")
            n = 0
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
                    n += 1
            if n == 0:
                os.remove(dst)
                continue
            written += 1
            logger.info("wrote %s (%d records)", dst, n)

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: run the tribunal (linear, single process, folder-mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_tribunal(jsonl_root: str, tribunal_output_dir: str, python_bin: str, judge_url: str = None) -> bool:
    tribunal_dir = os.path.join(REPO_ROOT, "tribunal")
    cmd = [
        python_bin, "-m", "tribunal.run_eval",
        "--input", jsonl_root,
        "--output", tribunal_output_dir,
    ]
    if judge_url:
        cmd += ["--judge-url", judge_url]

    log_path = os.path.join(tribunal_output_dir, "tribunal_run.log")
    os.makedirs(tribunal_output_dir, exist_ok=True)
    logger.info("Running tribunal: %s (cwd=%s)", " ".join(cmd), tribunal_dir)

    env = os.environ.copy()
    # Tribunal's judge server is already up on GPU 0 per your setup; the
    # tribunal client process itself doesn't need a GPU, but make sure it
    # doesn't accidentally grab one of the sweep GPUs mid-sweep-tail.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, cwd=tribunal_dir, env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    (logger.info if ok else logger.error)(
        "Tribunal %s in %.1fs (rc=%d) — log: %s",
        "DONE" if ok else "FAILED", elapsed, proc.returncode, log_path,
    )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: analyze model_summary.csv -> best (beta, temp) pair
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_METRICS = ["response_quality", "relevance", "helpfulness"]
SAFETY_METRICS = ["toxicity", "harmfulness", "refusal"]
ALL_METRICS = QUALITY_METRICS + SAFETY_METRICS


def parse_model_name(model_name: str):
    """'gsi_elo_no_th__beta1.2500_temp13.7500' -> ('gsi_elo_no_th', 1.25, 13.75)"""
    if "__" not in model_name:
        return None, None, None
    strategy, tag = model_name.split("__", 1)
    beta = temp = None
    for token in tag.split("_"):
        if token.startswith("beta"):
            beta = float(token[len("beta"):])
        elif token.startswith("temp"):
            temp = float(token[len("temp"):])
    return strategy, beta, temp


def load_scored_df(tribunal_output_dir: str):
    """
    Shared loader used by both analyze_results() and the plotting functions,
    so both operate on the exact same parsed/derived columns.
    Returns None if model_summary.csv is missing.
    """
    import pandas as pd

    summary_path = os.path.join(tribunal_output_dir, "model_summary.csv")
    if not os.path.exists(summary_path):
        return None

    df = pd.read_csv(summary_path)
    name_col = next((c for c in df.columns if "model" in c.lower()), df.columns[0])
    df = df.rename(columns={name_col: "model"})

    parsed = df["model"].apply(parse_model_name)
    df["strategy"] = parsed.apply(lambda x: x[0])
    df["beta"] = parsed.apply(lambda x: x[1])
    df["temperature"] = parsed.apply(lambda x: x[2])

    q_cols = [c for c in QUALITY_METRICS if c in df.columns]
    s_cols = [c for c in SAFETY_METRICS if c in df.columns]

    df["quality_mean"] = df[q_cols].mean(axis=1) if q_cols else float("nan")
    df["safety_bad_mean"] = df[s_cols].mean(axis=1) if s_cols else float("nan")
    df["composite_score"] = df["quality_mean"] - df["safety_bad_mean"]

    return df


def analyze_results(tribunal_output_dir: str, report_path: str):
    df = load_scored_df(tribunal_output_dir)
    if df is None:
        logger.error(
            "No model_summary.csv at %s — tribunal likely failed. See tribunal_run.log.",
            os.path.join(tribunal_output_dir, "model_summary.csv"),
        )
        with open(report_path, "w") as f:
            f.write("# FAILED\n\nNo model_summary.csv produced. Check tribunal_run.log in the same folder "
                    "as this report, and each config's run.log under runs/overnight_sweep/grid/<tag>/run.log.\n")
        return

    q_cols = [c for c in QUALITY_METRICS if c in df.columns]
    s_cols = [c for c in SAFETY_METRICS if c in df.columns]
    if not q_cols and not s_cols:
        logger.error("model_summary.csv has none of the expected metric columns: %s", list(df.columns))

    lines = []
    lines.append("# Overnight β × elo_temperature Sweep — Final Report\n")
    lines.append(f"_Generated: {datetime.now().isoformat()}_\n")
    lines.append(
        "\ncomposite_score = mean(response_quality, relevance, helpfulness) "
        "- mean(toxicity, harmfulness, refusal). Higher composite_score is better "
        "(more helpful/relevant/quality, less toxic/harmful/refusing).\n"
    )
    lines.append(
        "\n**Read the caveats at the top of run_overnight_sweep.py before trusting this** "
        "— elo_temperature only affects gsi_elo_no_th; softmax/swiss rows only really tell "
        "you about beta sensitivity, and n=15 prompts/cell is a small sample.\n"
    )

    lines.append("\n## Best (beta, temperature) per strategy\n")
    lines.append("| strategy | beta | temperature | composite_score | quality_mean | safety_bad_mean | " +
                  " | ".join(q_cols + s_cols) + " |")
    lines.append("|---|---|---|---|---|---|" + "---|" * len(q_cols + s_cols))

    for strat, g in df.groupby("strategy"):
        best_row = g.loc[g["composite_score"].idxmax()]
        metric_vals = " | ".join(f"{best_row[c]:.4f}" for c in (q_cols + s_cols))
        lines.append(
            f"| {strat} | {best_row['beta']:.4f} | {best_row['temperature']:.4f} | "
            f"{best_row['composite_score']:.4f} | {best_row['quality_mean']:.4f} | "
            f"{best_row['safety_bad_mean']:.4f} | {metric_vals} |"
        )

    lines.append("\n## Best overall (beta, temperature) across all strategies\n")
    best_overall = df.loc[df["composite_score"].idxmax()]
    lines.append(
        f"**{best_overall['strategy']}** at **beta={best_overall['beta']:.4f}, "
        f"elo_temperature={best_overall['temperature']:.4f}** — "
        f"composite_score={best_overall['composite_score']:.4f}\n"
    )

    lines.append("\n## Full grid (sorted by composite_score, descending)\n")
    df_sorted = df.sort_values("composite_score", ascending=False)
    cols_to_show = ["strategy", "beta", "temperature", "composite_score", "quality_mean", "safety_bad_mean"] + q_cols + s_cols
    lines.append(df_sorted[cols_to_show].round(4).to_markdown(index=False))

    lines.append(
        "\n## Plots\n\n"
        "- Per-config comparisons (all 3 strategies, all 6 rubrics), one file per "
        "(beta, temperature) cell: `plots/grid/beta{B:.4f}_temp{T:.4f}.png`\n"
        "- Unified overview across the whole sweep: "
        "`plots/unified_composite_heatmap.png` and `plots/unified_safety_vs_quality.png`\n"
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    df_sorted.to_csv(report_path.replace(".md", "_full_grid.csv"), index=False)

    logger.info("Report written to %s", report_path)
    logger.info(
        "Best overall: %s beta=%.4f temp=%.4f composite=%.4f",
        best_overall["strategy"], best_overall["beta"], best_overall["temperature"],
        best_overall["composite_score"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: plots
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_order(strategies_present):
    """Keep a stable, consistent left-to-right strategy order everywhere."""
    return [s for s in STRATEGIES if s in strategies_present]


def plot_per_config_grid(df, plots_root: str):
    """
    One bar chart PER (beta, temperature) config (25 total with the default
    grid): x-axis = the 6 tribunal rubrics, one colored bar per strategy per
    rubric, so you can see all 3 strategies side-by-side for that exact
    config. Filenames are keyed off (beta, temperature) so each config's
    plot lands in its own file and never overwrites another config's.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid_dir = os.path.join(plots_root, "grid")
    os.makedirs(grid_dir, exist_ok=True)

    metrics = [m for m in ALL_METRICS if m in df.columns]
    if not metrics:
        logger.warning("No known rubric columns found — skipping per-config plots.")
        return 0

    written = 0
    for (beta, temp), g in df.groupby(["beta", "temperature"]):
        strategies = _strategy_order(g["strategy"].unique())
        if not strategies:
            continue

        n_strats = len(strategies)
        x = np.arange(len(metrics))
        width = 0.8 / max(n_strats, 1)

        fig, ax = plt.subplots(figsize=(2.2 * len(metrics), 5))
        for i, strat in enumerate(strategies):
            row = g[g["strategy"] == strat].iloc[0]
            vals = [row[m] for m in metrics]
            offset = (i - (n_strats - 1) / 2) * width
            bars = ax.bar(
                x + offset, vals, width,
                color=STRATEGY_COLORS.get(strat, "#888888"),
                edgecolor="white",
                label=STRATEGY_LABELS.get(strat, strat),
            )
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("score (0-1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_title(
            f"beta={beta:.4f}  elo_temperature={temp:.4f}",
            fontsize=11, fontweight="bold",
        )
        fig.suptitle("Swiss Knife — Strategy comparison at this config", fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(grid_dir, f"{cfg_tag(beta, temp)}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written += 1

    logger.info("Wrote %d per-config plots to %s", written, grid_dir)
    return written


def plot_unified_overview(df, plots_root: str):
    """
    Two plots that summarize the ENTIRE sweep at once (all 25 configs x 3
    strategies), each written to its own fixed filename under plots_root
    (not plots_root/grid, so they can never collide with the 25 per-config
    files above):

      1. unified_composite_heatmap.png — one heatmap per strategy, beta on
         the y-axis, elo_temperature on the x-axis, cell = composite_score.
         For gsi_softmax_no_th / gsi_swiss_no_th expect near-identical rows
         across temperature (they don't consume elo_temperature — see the
         caveats at the top of this script); gsi_elo_no_th's heatmap is the
         only one reflecting a genuine 2D sweep.
      2. unified_safety_vs_quality.png — scatter of quality_mean (x) vs
         safety_bad_mean (y, lower/left is better/safer) for all 75 points,
         colored by strategy, so you can see the whole sweep's
         quality/safety trade-off frontier in one view.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(plots_root, exist_ok=True)
    strategies = _strategy_order(df["strategy"].unique())
    if not strategies:
        logger.warning("No known strategies found — skipping unified overview plots.")
        return

    # ── 1. Per-strategy composite_score heatmap over the (beta, temp) grid ──
    betas = sorted(df["beta"].unique())
    temps = sorted(df["temperature"].unique())

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        sub = df[df["strategy"] == strat]
        grid = np.full((len(betas), len(temps)), np.nan)
        for _, row in sub.iterrows():
            i = betas.index(row["beta"])
            j = temps.index(row["temperature"])
            grid[i, j] = row["composite_score"]

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([f"{t:.2f}" for t in temps], fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"{b:.3f}" for b in betas], fontsize=8)
        ax.set_xlabel("elo_temperature", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("beta", fontsize=9)
        ax.set_title(STRATEGY_LABELS.get(strat, strat), fontsize=10, fontweight="bold")

        for i in range(len(betas)):
            for j in range(len(temps)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Swiss Knife — composite_score across the full sweep, per strategy",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    heatmap_path = os.path.join(plots_root, "unified_composite_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote unified heatmap to %s", heatmap_path)

    # ── 2. Safety-vs-quality scatter, all 75 points, colored by strategy ────
    fig, ax = plt.subplots(figsize=(7, 6))
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        ax.scatter(
            sub["quality_mean"], sub["safety_bad_mean"],
            color=STRATEGY_COLORS.get(strat, "#888888"),
            label=STRATEGY_LABELS.get(strat, strat),
            s=60, edgecolor="white", alpha=0.85,
        )
    ax.set_xlabel("quality_mean (higher is better)", fontsize=9)
    ax.set_ylabel("safety_bad_mean (lower is better)", fontsize=9)
    ax.set_title("Swiss Knife — quality vs. safety across the whole sweep", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    scatter_path = os.path.join(plots_root, "unified_safety_vs_quality.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote unified scatter to %s", scatter_path)


def generate_all_plots(tribunal_output_dir: str, plots_root: str):
    df = load_scored_df(tribunal_output_dir)
    if df is None:
        logger.error("Skipping plots — no model_summary.csv at %s", tribunal_output_dir)
        return
    plot_per_config_grid(df, plots_root)
    plot_unified_overview(df, plots_root)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Overnight beta x elo_temperature sweep + tribunal eval")
    p.add_argument("--beta-min", type=float, default=1.0)
    p.add_argument("--beta-max", type=float, default=2.5)
    p.add_argument("--beta-steps", type=int, default=5)
    p.add_argument("--temp-min", type=float, default=10.0)
    p.add_argument("--temp-max", type=float, default=15.0)
    p.add_argument("--temp-steps", type=int, default=5)
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                   help="Physical GPU ids to use for generation (GPU 0 is left for the tribunal judge server).")
    p.add_argument("--num-prompts", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--gsi-n", type=int, default=8)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--judge-url", type=str, default=None,
                   help="Override tribunal vLLM judge URL (default: tribunal's own config.py default, http://localhost:8000/v1).")
    p.add_argument("--output-root", type=str, default="runs/overnight_sweep")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip step 1 (assume grid/ already populated) and go straight to convert+tribunal+report+plots.")
    p.add_argument("--skip-tribunal", action="store_true",
                   help="Skip step 3 (assume tribunal already run) and go straight to the report+plots.")
    p.add_argument("--skip-plots", action="store_true",
                   help="Skip step 5 (plot generation) and stop after FINAL_REPORT.md.")
    return p.parse_args()


def main():
    args = parse_args()

    betas = linspace(args.beta_min, args.beta_max, args.beta_steps)
    temps = linspace(args.temp_min, args.temp_max, args.temp_steps)
    assert_unique_tags(betas, temps)

    output_root = os.path.join(REPO_ROOT, args.output_root)
    grid_root = os.path.join(output_root, "grid")
    jsonl_root = os.path.join(output_root, "tribunal_inputs")
    tribunal_output_dir = os.path.join(output_root, "tribunal_results")
    plots_root = os.path.join(output_root, "plots")
    report_path = os.path.join(output_root, "FINAL_REPORT.md")
    os.makedirs(output_root, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Overnight sweep starting")
    logger.info("betas (%d): %s", len(betas), betas)
    logger.info("temps (%d): %s", len(temps), temps)
    logger.info("total configs: %d  |  strategies per config: %d  |  total generation runs: %d",
                len(betas) * len(temps), len(STRATEGIES), len(betas) * len(temps) * len(STRATEGIES))
    logger.info("GPUs: %s", args.gpus)
    logger.info("output root: %s", output_root)
    logger.info("=" * 70)

    t_all = time.perf_counter()

    # ── Step 1: generation grid ────────────────────────────────────────
    if not args.skip_generation:
        os.makedirs(grid_root, exist_ok=True)
        grid_results = run_grid(
            betas, temps, args.gpus,
            args.num_prompts, args.max_tokens, args.gsi_n,
            grid_root, args.python_bin,
        )
        with open(os.path.join(output_root, "grid_run_summary.json"), "w") as f:
            json.dump(grid_results, f, indent=2)

        n_failed = sum(1 for r in grid_results if r.get("returncode", -1) != 0)
        logger.info("Generation grid complete: %d/%d configs OK (%d failed).",
                    len(grid_results) - n_failed, len(grid_results), n_failed)
        if n_failed:
            for r in grid_results:
                if r.get("returncode", -1) != 0:
                    logger.error("FAILED config beta=%s temp=%s — see %s", r.get("beta"), r.get("temp"), r.get("log_path"))
    else:
        logger.info("Skipping generation step (--skip-generation).")

    # ── Step 2: convert to jsonl ────────────────────────────────────────
    n_jsonl = convert_all_to_jsonl(grid_root, jsonl_root)
    logger.info("Converted %d/%d expected jsonl files (%d configs x %d strategies).",
                n_jsonl, len(betas) * len(temps) * len(STRATEGIES), len(betas) * len(temps), len(STRATEGIES))

    # ── Step 3: run tribunal ────────────────────────────────────────────
    if not args.skip_tribunal:
        ok = run_tribunal(jsonl_root, tribunal_output_dir, args.python_bin, args.judge_url)
        if not ok:
            logger.error("Tribunal run failed — check %s/tribunal_run.log. "
                         "Is the vLLM judge server on GPU 0 still up?", tribunal_output_dir)
    else:
        logger.info("Skipping tribunal step (--skip-tribunal).")

    # ── Step 4: analyze + best config ──────────────────────────────────
    try:
        analyze_results(tribunal_output_dir, report_path)
    except Exception as e:
        logger.error("Analysis failed: %s\n%s", e, traceback.format_exc())

    # ── Step 5: plots ───────────────────────────────────────────────────
    if not args.skip_plots:
        try:
            generate_all_plots(tribunal_output_dir, plots_root)
        except Exception as e:
            logger.error("Plot generation failed: %s\n%s", e, traceback.format_exc())
    else:
        logger.info("Skipping plot step (--skip-plots).")

    elapsed_total = time.perf_counter() - t_all
    logger.info("=" * 70)
    logger.info("Overnight sweep finished in %.1f min. Report: %s  Plots: %s",
                elapsed_total / 60, report_path, plots_root)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()