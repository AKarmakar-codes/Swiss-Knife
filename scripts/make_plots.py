"""Generate matplotlib plots from a Swiss Knife experiment run directory.

Reads the demo log files and per-round JSONL scores dump (produced when
--scores-log is passed to Model_mechanics.main) and emits PNG plots into
<runs_dir>/plots/.

Plots produced:
    1. alpha_override_curve.png — α vs override-rate (Demo 2)
    2. K_latency.png           — K vs wall-clock generation time (Demo 5)
    3. score_scales.png        — raw draft vs blade reward histograms,
                                 the visual proof of the scale-mismatch bug
    4. calibration_invariance.png — bar chart of identical-vs-baseline
                                    across the bias sweep (Demo 6)

Usage:
    python scripts/make_plots.py runs/<timestamp>
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display server required
import matplotlib.pyplot as plt
import numpy as np

PAT_MATCH = re.compile(
    r"Δdraft=(-?[\d.]+)\s+Δblade=(-?[\d.]+)\s+score=(-?[\d.]+)"
)
PAT_TIME = re.compile(r"Generation time:\s*([\d.]+)s")
PAT_GEN = re.compile(
    r"GENERATED OUTPUT\s*\n=+\s*\n(.*?)\n=+\s*\n", re.DOTALL
)


# ── data extraction helpers ────────────────────────────────────────────

def read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None


def override_rate(path):
    text = read(path)
    if text is None:
        return None
    matches = PAT_MATCH.findall(text)
    if not matches:
        return None
    flipped = sum(
        1 for d, _b, s in matches if (float(d) > 0) != (float(s) > 0)
    )
    return 100 * flipped / len(matches)


def gen_time(path):
    text = read(path)
    if text is None:
        return None
    m = PAT_TIME.search(text)
    return float(m.group(1)) if m else None


def gen_output(path):
    text = read(path)
    if text is None:
        return None
    m = PAT_GEN.search(text)
    return m.group(1).strip() if m else None


# ── plot routines ──────────────────────────────────────────────────────

def plot_alpha_override(runs_dir, out_dir):
    alphas = [1.0, 0.7, 0.5, 0.3, 0.0]
    pairs = []
    for a in alphas:
        f = runs_dir / f"demo2_alpha_sweep/alpha_{a}.txt"
        r = override_rate(f)
        if r is not None:
            pairs.append((a, r))
    if not pairs:
        return None

    xs, ys = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhspan(15, 40, alpha=0.10, color="green",
               label="Healthy operating band (15–40%)")
    ax.plot(xs, ys, marker="o", markersize=9, linewidth=2.4,
            color="#1f77b4", label="Override rate")
    for x, y in pairs:
        ax.annotate(f"{y:.0f}%",
                    xy=(x, y), xytext=(0, 9),
                    textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"$\alpha$  (1.0 = pure draft,  0.0 = pure blade)",
                  fontsize=11)
    ax.set_ylabel("Override rate %  (blade flips draft's preferred winner)",
                  fontsize=11)
    ax.set_title("Demo 2 — α steering gradient (helpfulness blade, normalized scores)",
                 fontsize=12, fontweight="bold")
    ax.invert_xaxis()  # left = pure-draft, right = pure-blade
    ax.set_ylim(-3, max(60, max(ys) + 10))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = out_dir / "alpha_override_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_K_latency(runs_dir, out_dir):
    Ks = [2, 4, 8, 16]
    pairs = []
    for K in Ks:
        f = runs_dir / f"demo5_K_sweep/K{K}.txt"
        t = gen_time(f)
        if t is not None:
            pairs.append((K, t))
    if not pairs:
        return None

    xs, ys = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar([str(k) for k in xs], ys, color="#ff7f0e",
                  edgecolor="black", linewidth=0.6)
    base = ys[0]
    for bar, t in zip(bars, ys):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                height + max(ys) * 0.015,
                f"{t:.1f}s\n({t/base:.2f}×)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("K  (candidates per tournament round)", fontsize=11)
    ax.set_ylabel("Wall-clock time (s)  for 60 generated tokens", fontsize=11)
    ax.set_title("Demo 5 — Latency vs K (L=5, helpfulness blade)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(ys) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "K_latency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_score_scales(runs_dir, out_dir):
    jsonl = runs_dir / "demo2_alpha_sweep/scores_alpha_0.5.jsonl"
    if not jsonl.exists():
        return None

    drafts, blades_raw = [], []
    for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        drafts.extend(d.get("draft_raw", []))
        blades_raw.extend(d.get("blade_raw", []))

    if not drafts or not blades_raw:
        return None

    d_std = float(np.std(drafts))
    b_std = float(np.std(blades_raw))
    ratio = d_std / max(b_std, 1e-9)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ax[0].hist(drafts, bins=40, color="#1f77b4",
               edgecolor="black", linewidth=0.4, alpha=0.85)
    ax[0].set_title(f"Raw draft span log-prob   (n={len(drafts)})\n"
                    f"μ = {np.mean(drafts):.2f},  σ = {d_std:.2f}",
                    fontsize=11)
    ax[0].set_xlabel(r"$\log\,\pi_{\mathrm{draft}}(\mathrm{span}\,|\,x\oplus y)$",
                     fontsize=11)
    ax[0].set_ylabel("count")
    ax[0].grid(alpha=0.3)

    ax[1].hist(blades_raw, bins=40, color="#d62728",
               edgecolor="black", linewidth=0.4, alpha=0.85)
    ax[1].set_title(f"Raw blade reward   (n={len(blades_raw)})\n"
                    f"μ = {np.mean(blades_raw):.4f},  σ = {b_std:.4f}",
                    fontsize=11)
    ax[1].set_xlabel(r"$\beta\,[\,\log\pi_{\mathrm{blade}} - \log\pi_{\mathrm{ref}}\,]$",
                     fontsize=11)
    ax[1].set_ylabel("count")
    ax[1].grid(alpha=0.3)

    fig.suptitle(
        f"Score-scale mismatch: draft σ is ~{ratio:.0f}× the blade σ → "
        f"the un-normalised tournament collapses to draft-argmax at every α > 0.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "score_scales.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_calibration(runs_dir, out_dir):
    biases = [0, 100, 1000, 10000]
    baseline = gen_output(runs_dir / "demo6_calibration/bias_0.txt")
    same = []
    have_any = False
    for b in biases:
        f = runs_dir / f"demo6_calibration/bias_{b}.txt"
        out = gen_output(f)
        if out is None:
            same.append(None)
        else:
            have_any = True
            same.append(1 if (out == baseline) else 0)

    if not have_any:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_vals = [0 if v is None else v for v in same]
    colors = ["#2ca02c" if v == 1 else "#d62728" for v in plot_vals]
    ax.bar([str(b) for b in biases], plot_vals,
           color=colors, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(same):
        if v is None:
            label, color = "?", "gray"
        elif v == 1:
            label, color = "✓", "green"
        else:
            label, color = "✗", "red"
        ax.text(i, plot_vals[i] + 0.05, label,
                ha="center", va="bottom",
                fontsize=24, fontweight="bold", color=color)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Output identical to bias=0 baseline   (1 = yes)",
                  fontsize=11)
    ax.set_xlabel("blade-bias  (constant added to every blade score)",
                  fontsize=11)
    ax.set_title("Demo 6 — Kernel-level calibration invariance  "
                 "(--no-normalize, α = 0)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "calibration_invariance.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ── entry point ────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print("Usage: make_plots.py <runs_dir>", file=sys.stderr)
        sys.exit(1)
    runs_dir = Path(sys.argv[1])
    if not runs_dir.is_dir():
        print(f"Not a directory: {runs_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = runs_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    routines = [
        plot_alpha_override,
        plot_K_latency,
        plot_score_scales,
        plot_calibration,
    ]
    saved = []
    for fn in routines:
        try:
            p = fn(runs_dir, out_dir)
            if p is not None:
                print(f"  ✓ {p}")
                saved.append(p)
            else:
                print(f"  - {fn.__name__}: missing inputs, skipped")
        except Exception as e:
            print(f"  ✗ {fn.__name__} failed: {e}", file=sys.stderr)

    print(f"\nGenerated {len(saved)} plot(s) in {out_dir}/")


if __name__ == "__main__":
    main()
