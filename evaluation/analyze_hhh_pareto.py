"""
Swiss-Knife — HHH Pareto Frontier Analysis
===========================================

Turns the per-response reward CSV from `score_hhh_rewards.py` into the numbers
and figures the paper reports.

WHAT IS MEASURED, AND WHY EACH ONE
----------------------------------
MOD scores a steering method by the *area of its Pareto frontier*, which it says
"reflects the optimality and uniformity of the solution distribution"
(Shi et al., 2024, §4.1). Those are two different properties, and MOD reports one
number for both. We separate them, because CBN is a claim about the second:

  1. HYPERVOLUME (optimality). How far out the frontier sits. 2-D area per
     objective pair and 3-D volume over all of HHH, against a common reference
     point so methods are comparable. Higher is better.

  2. SPACING Δ (uniformity). Schott's spacing metric on the min-max-normalised
     frontier: the coefficient of variation of nearest-neighbour distances
     between frontier points. LOWER is better. This is the metric that tests
     CBN's actual claim — that equal steps in w produce equal steps along the
     frontier. Without CBN, a blade whose implicit rewards have larger magnitude
     dominates the mix, w-space bunches near that blade's vertex, and Δ rises
     even when hypervolume is unchanged.

  3. COVERAGE. The range spanned on each axis between w=0 and w=1. A method can
     have low Δ simply by not moving; coverage is the guard against that, and
     must be read next to Δ.

  4. PAIRED BOOTSTRAP at matched w. Per-prompt differences between two methods at
     the same weight vector, 10k resamples. Aggregate frontier metrics have no
     error bars on their own; this supplies them.

  5. THROUGHPUT. tok/s and forward passes per generated token. MOD pays K vocab
     forward passes per token; we pay K blade passes per step over N candidates.
     Reporting both stops either method from claiming a free lunch.

Invalid rows (empty generations, flagged `valid=0` by the scorer) are excluded
from every statistic and counted separately, rather than being scored as zero.

Run:
    python evaluation/analyze_hhh_pareto.py
    python evaluation/analyze_hhh_pareto.py --rewards runs/hhh_pareto/hhh_rewards.csv \
        --objective-set armorm --out-dir runs/hhh_pareto/analysis
    python evaluation/analyze_hhh_pareto.py --objective-set mod   # MOD's own two RMs
"""

import os
import sys
import csv
import json
import math
import random
import logging
import argparse
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HHHParetoAnalysis")

# Objective sets. 'armorm' is the primary 3-D HHH instrument; 'mod' reproduces
# MOD's own two axes so our frontier can be overlaid on their Figure 3.
OBJECTIVE_SETS = {
    "armorm": {
        "helpfulness":  "armorm_helpfulness",
        "honesty":      "armorm_honesty",
        "harmlessness": "armorm_harmlessness",
    },
    "mod": {
        "helpfulness":  "mod_helpful",
        "harmlessness": "mod_harmless",
    },
}

# ── Figure styling ───────────────────────────────────────────────────────────
# Three categorical series only. The reference palette's first three slots are
# the ones that clear the all-pairs CVD and normal-vision floors, which is the
# relevant pairlist for a scatter/frontier plot where every series is compared
# against every other. `base` is deliberately NOT a fourth categorical hue: it
# is a single unsteered reference point, so it takes neutral ink.
# Marker shape duplicates the hue (secondary encoding) so the figure survives
# greyscale printing, and every frontier is direct-labelled — required relief for
# the aqua step, which sits below 3:1 against a white surface.
METHOD_STYLE = {
    "swiss": {"color": "#2a78d6", "marker": "o", "label": "Swiss-Knife (CBN)"},
    "mod":   {"color": "#eb6834", "marker": "s", "label": "MOD"},
    "rs":    {"color": "#1baf7a", "marker": "^", "label": "Rewarded Soups"},
    "base":  {"color": "#6b6b63", "marker": "*", "label": "SFT backbone"},
}
METHOD_ORDER = ["swiss", "mod", "rs", "base"]

GRID_COLOR = "#e3e3dd"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5c55"


# ═════════════════════════════════════════════════════════════════════════════
# Pareto geometry
# ═════════════════════════════════════════════════════════════════════════════

def pareto_front(points: List[Tuple[float, ...]]) -> List[int]:
    """Indices of the non-dominated points (all objectives maximised).

    p dominates q iff p >= q on every axis and p > q on at least one.
    """
    keep = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if all(qk >= pk for qk, pk in zip(q, p)) and any(qk > pk for qk, pk in zip(q, p)):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def hv_2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """Exact 2-D hypervolume (area) for maximisation against reference `ref`."""
    pts = [p for p in points if p[0] > ref[0] and p[1] > ref[1]]
    if not pts:
        return 0.0
    idx = pareto_front(pts)
    front = sorted((pts[i] for i in idx), key=lambda p: p[0], reverse=True)
    area, prev_y = 0.0, ref[1]
    for x, y in front:
        if y > prev_y:
            area += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return area


def hv_3d(points: List[Tuple[float, float, float]], ref: Tuple[float, float, float]) -> float:
    """3-D hypervolume by z-sweep: slab height x 2-D area of the points above it.

    O(n^2 log n), which is irrelevant at the ~15 frontier points a weight grid
    produces, and exact — no Monte-Carlo error to explain to a reviewer.
    """
    pts = [p for p in points if all(pk > rk for pk, rk in zip(p, ref))]
    if not pts:
        return 0.0
    by_z = sorted(pts, key=lambda p: p[2], reverse=True)
    total = 0.0
    for k in range(len(by_z)):
        z_hi = by_z[k][2]
        z_lo = by_z[k + 1][2] if k + 1 < len(by_z) else ref[2]
        height = z_hi - z_lo
        if height <= 0:
            continue
        total += hv_2d([(p[0], p[1]) for p in by_z[: k + 1]], (ref[0], ref[1])) * height
    return total


def spacing_delta(front: List[Tuple[float, ...]]) -> Optional[float]:
    """Schott's spacing metric: CV of nearest-neighbour distances. Lower = more uniform.

    Callers must pass points already normalised to a common [0,1] box, otherwise
    this reports the objective units rather than the frontier's regularity.
    """
    if len(front) < 3:
        return None
    dists = []
    for i, p in enumerate(front):
        best = min(
            math.dist(p, q) for j, q in enumerate(front) if j != i
        )
        dists.append(best)
    mean_d = sum(dists) / len(dists)
    if mean_d <= 1e-12:
        return 0.0
    var = sum((d - mean_d) ** 2 for d in dists) / (len(dists) - 1)
    return math.sqrt(var) / mean_d


def normalise(points: List[Tuple[float, ...]], lo: List[float], hi: List[float]):
    out = []
    for p in points:
        out.append(tuple(
            (v - l) / (h - l) if h > l else 0.0 for v, l, h in zip(p, lo, hi)
        ))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════════

def paired_bootstrap(a: Dict[str, float], b: Dict[str, float],
                     n_boot: int = 10000, seed: int = 42) -> Optional[dict]:
    """Paired difference a-b over shared prompt ids, with a percentile CI."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 5:
        return None
    diffs = [a[k] - b[k] for k in shared]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    wins = sum(1 for d in diffs if d > 0) / n
    return {
        "n": n,
        "mean_diff": round(mean, 5),
        "ci_low": round(means[int(0.025 * n_boot)], 5),
        "ci_high": round(means[int(0.975 * n_boot)], 5),
        "win_rate": round(wins, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Load & aggregate
# ═════════════════════════════════════════════════════════════════════════════

def load_rewards(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    logger.info("Read %d reward rows from %s", len(rows), path)
    return rows


def aggregate(rows: List[dict], objectives: Dict[str, str]):
    """Group by (method, config).

    Returns (agg, per_prompt, invalid_counts) where
      agg[(m,c)]        = {"w": {...}, "means": {axis: mean}, "n": int, "tok_s": mean}
      per_prompt[(m,c)] = {axis: {prompt_id: value}}
    """
    agg, per_prompt, invalid = {}, {}, defaultdict(int)
    buckets = defaultdict(lambda: defaultdict(list))
    tok_s = defaultdict(list)
    weights = {}

    for r in rows:
        key = (r["method"], r["config"])
        weights[key] = {
            "helpfulness":  float(r.get("w_helpfulness") or 0.0),
            "honesty":      float(r.get("w_honesty") or 0.0),
            "harmlessness": float(r.get("w_harmlessness") or 0.0),
        }
        if (r.get("valid") or "1") != "1":
            invalid[key] += 1
            continue

        ok = True
        vals = {}
        for axis, col in objectives.items():
            raw = r.get(col)
            if raw is None or raw == "":
                ok = False
                break
            vals[axis] = float(raw)
        if not ok:
            invalid[key] += 1
            continue

        for axis, v in vals.items():
            buckets[key][axis].append((r["id"], v))
        if r.get("tokens_per_second"):
            try:
                tok_s[key].append(float(r["tokens_per_second"]))
            except ValueError:
                pass

    for key, axes in buckets.items():
        means, pp = {}, {}
        n = 0
        for axis, pairs in axes.items():
            means[axis] = sum(v for _, v in pairs) / len(pairs)
            pp[axis] = {pid: v for pid, v in pairs}
            n = len(pairs)
        agg[key] = {
            "w": weights[key],
            "means": means,
            "n": n,
            "tok_s": (sum(tok_s[key]) / len(tok_s[key])) if tok_s[key] else None,
        }
        per_prompt[key] = pp

    logger.info("Aggregated %d (method, config) cells.", len(agg))
    return agg, per_prompt, dict(invalid)


# ═════════════════════════════════════════════════════════════════════════════
# Frontier metrics per method
# ═════════════════════════════════════════════════════════════════════════════

def frontier_metrics(agg: dict, axes: List[str]) -> dict:
    """Hypervolume, spacing and coverage per method, on a shared reference point."""
    methods = sorted({m for m, _ in agg}, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

    # Common reference = componentwise minimum over every point of every method,
    # nudged down so a method that owns the minimum still contributes volume.
    all_pts = [tuple(v["means"][a] for a in axes) for v in agg.values()
               if all(a in v["means"] for a in axes)]
    if not all_pts:
        return {}
    lo = [min(p[i] for p in all_pts) for i in range(len(axes))]
    hi = [max(p[i] for p in all_pts) for i in range(len(axes))]
    span = [h - l for l, h in zip(lo, hi)]
    ref = tuple(l - 0.02 * (s if s > 0 else 1.0) for l, s in zip(lo, span))

    out = {}
    for m in methods:
        cells = {c: v for (mm, c), v in agg.items()
                 if mm == m and all(a in v["means"] for a in axes)}
        if not cells:
            continue
        configs = sorted(cells)
        pts = [tuple(cells[c]["means"][a] for a in axes) for c in configs]

        idx = pareto_front(pts)
        front_pts = [pts[i] for i in idx]
        front_norm = normalise(front_pts, lo, hi)

        hv = hv_3d(pts, ref) if len(axes) == 3 else hv_2d(pts, ref)
        out[m] = {
            "n_configs": len(pts),
            "n_frontier": len(front_pts),
            "hypervolume": round(hv, 6),
            "spacing_delta": (round(spacing_delta(front_norm), 4)
                              if spacing_delta(front_norm) is not None else None),
            "coverage": {a: round(max(p[i] for p in pts) - min(p[i] for p in pts), 5)
                         for i, a in enumerate(axes)},
            "mean_tok_s": (round(sum(c["tok_s"] for c in cells.values() if c["tok_s"]) /
                                 max(sum(1 for c in cells.values() if c["tok_s"]), 1), 2)
                           if any(c["tok_s"] for c in cells.values()) else None),
            "frontier_configs": [configs[i] for i in idx],
        }
    return {"reference_point": [round(r, 6) for r in ref],
            "axis_min": [round(v, 6) for v in lo],
            "axis_max": [round(v, 6) for v in hi],
            "methods": out}


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

def plot_pairwise_frontiers(agg: dict, axes: List[str], out_path: str, objective_set: str):
    """One panel per objective pair — the shape of MOD's Figure 3."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping figures.")
        return

    pairs = list(itertools.combinations(axes, 2))
    fig, axarr = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 4.4))
    if len(pairs) == 1:
        axarr = [axarr]

    for panel, (ax, (ax_x, ax_y)) in enumerate(zip(axarr, pairs)):
        for slot, m in enumerate(METHOD_ORDER):
            cells = {c: v for (mm, c), v in agg.items()
                     if mm == m and ax_x in v["means"] and ax_y in v["means"]}
            if not cells:
                continue
            style = METHOD_STYLE[m]

            # The unsteered backbone is one point, not a frontier: no line, and
            # neutral ink rather than a categorical hue.
            if m == "base":
                for v in cells.values():
                    ax.plot(v["means"][ax_x], v["means"][ax_y], style["marker"],
                            color=style["color"], markersize=11, zorder=5,
                            markeredgecolor="white", markeredgewidth=1.2)
                continue

            pts = [(v["means"][ax_x], v["means"][ax_y]) for v in cells.values()]
            idx = pareto_front(pts)
            front = sorted((pts[i] for i in idx), key=lambda p: p[0])

            # All sampled points, recessive; the non-dominated frontier, emphasised.
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style["marker"],
                    color=style["color"], markersize=5, alpha=0.35, linestyle="none", zorder=2)
            ax.plot([p[0] for p in front], [p[1] for p in front], style["marker"] + "-",
                    color=style["color"], linewidth=2.0, markersize=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.0)

            # Direct labels on the FIRST panel only, vertically staggered by
            # method slot. Labelling every panel collides: the three frontiers
            # converge at their endpoints, so the annotations land on top of each
            # other. One labelled panel plus the shared legend and distinct
            # marker shapes still identifies every series without colour alone,
            # and the emitted CSV/LaTeX tables carry the table-view relief the
            # low-contrast aqua step requires.
            if front and panel == 0:
                ax.annotate(style["label"], xy=front[-1],
                            xytext=(7, 4 + 11 * slot), textcoords="offset points",
                            fontsize=8.5, color=TEXT_SECONDARY, zorder=6)

        ax.set_xlabel(f"{ax_x} reward →", fontsize=9.5, color=TEXT_PRIMARY)
        ax.set_ylabel(f"{ax_y} reward →", fontsize=9.5, color=TEXT_PRIMARY)
        # Headroom so the staggered labels in panel 0 are never clipped.
        ax.margins(x=0.16, y=0.10)
        ax.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)

    # Legend present in addition to direct labels, so identity is never colour-alone.
    handles = []
    from matplotlib.lines import Line2D
    for m in METHOD_ORDER:
        if any(mm == m for mm, _ in agg):
            s = METHOD_STYLE[m]
            handles.append(Line2D([], [], color=s["color"], marker=s["marker"],
                                  linewidth=2.0 if m != "base" else 0,
                                  markersize=7, label=s["label"]))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"HHH Pareto frontiers — {objective_set} rewards", fontsize=11,
                 color=TEXT_PRIMARY, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure → %s (+ .pdf)", out_path)


# ═════════════════════════════════════════════════════════════════════════════
# Tables
# ═════════════════════════════════════════════════════════════════════════════

def write_tables(agg: dict, axes: List[str], metrics: dict, invalid: dict,
                 out_dir: str, objective_set: str):
    os.makedirs(out_dir, exist_ok=True)

    # Per-config means — the raw data behind every frontier point.
    per_config = os.path.join(out_dir, f"pareto_points_{objective_set}.csv")
    with open(per_config, "w", encoding="utf-8", newline="") as f:
        cols = (["method", "config", "n_valid", "n_invalid", "mean_tok_s"]
                + [f"w_{b}" for b in ("helpfulness", "honesty", "harmlessness")]
                + [f"reward_{a}" for a in axes])
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for (m, c), v in sorted(agg.items()):
            row = {"method": m, "config": c, "n_valid": v["n"],
                   "n_invalid": invalid.get((m, c), 0),
                   "mean_tok_s": round(v["tok_s"], 2) if v["tok_s"] else ""}
            for b in ("helpfulness", "honesty", "harmlessness"):
                row[f"w_{b}"] = v["w"].get(b, 0.0)
            for a in axes:
                row[f"reward_{a}"] = round(v["means"].get(a, float("nan")), 5)
            w.writerow(row)
    logger.info("Wrote %s", per_config)

    # Frontier summary + LaTeX
    summary = os.path.join(out_dir, f"frontier_summary_{objective_set}.csv")
    with open(summary, "w", encoding="utf-8", newline="") as f:
        cols = ["method", "n_configs", "n_frontier", "hypervolume", "spacing_delta",
                "mean_tok_s"] + [f"coverage_{a}" for a in axes]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m, d in metrics.get("methods", {}).items():
            row = {"method": m, "n_configs": d["n_configs"], "n_frontier": d["n_frontier"],
                   "hypervolume": d["hypervolume"], "spacing_delta": d["spacing_delta"],
                   "mean_tok_s": d["mean_tok_s"]}
            for a in axes:
                row[f"coverage_{a}"] = d["coverage"][a]
            w.writerow(row)
    logger.info("Wrote %s", summary)

    tex = os.path.join(out_dir, f"frontier_table_{objective_set}.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by evaluation/analyze_hhh_pareto.py\n")
        f.write("\\begin{tabular}{l" + "r" * (3 + len(axes)) + "}\n\\toprule\n")
        f.write("Method & HV $\\uparrow$ & $\\Delta$ $\\downarrow$ & Tok/s $\\uparrow$ & "
                + " & ".join(f"Cov.\\ {a[:4]}." for a in axes) + " \\\\\n\\midrule\n")
        for m, d in metrics.get("methods", {}).items():
            label = METHOD_STYLE.get(m, {}).get("label", m)
            delta = f"{d['spacing_delta']:.3f}" if d["spacing_delta"] is not None else "--"
            toks = f"{d['mean_tok_s']:.1f}" if d["mean_tok_s"] else "--"
            covs = " & ".join(f"{d['coverage'][a]:.3f}" for a in axes)
            f.write(f"{label} & {d['hypervolume']:.4f} & {delta} & {toks} & {covs} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info("Wrote %s", tex)


def write_contrasts(per_prompt: dict, agg: dict, axes: List[str],
                    out_dir: str, objective_set: str, n_boot: int, seed: int):
    """Paired bootstrap, ours vs each baseline, at every matched weight vector."""
    def wkey(v):
        return tuple(round(v["w"].get(b, 0.0), 4) for b in ("helpfulness", "honesty", "harmlessness"))

    ours = {wkey(v): (m, c) for (m, c), v in agg.items() if m == "swiss"}
    rows = []
    for (m, c), v in sorted(agg.items()):
        if m == "swiss" or m == "base":
            continue
        match = ours.get(wkey(v))
        if match is None:
            continue
        for axis in axes:
            a = per_prompt.get(match, {}).get(axis, {})
            b = per_prompt.get((m, c), {}).get(axis, {})
            res = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
            if res is None:
                continue
            rows.append({
                "objective_set": objective_set, "axis": axis,
                "config": c, "baseline": m,
                "w_helpfulness": v["w"].get("helpfulness", 0.0),
                "w_honesty": v["w"].get("honesty", 0.0),
                "w_harmlessness": v["w"].get("harmlessness", 0.0),
                **res,
                "significant": int(res["ci_low"] > 0 or res["ci_high"] < 0),
            })

    if not rows:
        logger.warning("No matched (method, w) pairs — nothing to contrast.")
        return

    path = os.path.join(out_dir, f"paired_contrasts_{objective_set}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d contrasts, %d significant)",
                path, len(rows), sum(r["significant"] for r in rows))


# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Analyse the HHH Pareto frontier")
    p.add_argument("--rewards", type=str, default="runs/hhh_pareto/hhh_rewards.csv")
    p.add_argument("--objective-set", type=str, default="armorm", choices=list(OBJECTIVE_SETS))
    p.add_argument("--out-dir", type=str, default="runs/hhh_pareto/analysis")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    objectives = OBJECTIVE_SETS[args.objective_set]
    axes = list(objectives)

    rows = load_rewards(args.rewards)
    agg, per_prompt, invalid = aggregate(rows, objectives)
    if not agg:
        logger.error("No valid rows for objective set '%s' — are those columns in the CSV?",
                     args.objective_set)
        sys.exit(1)

    metrics = frontier_metrics(agg, axes)
    os.makedirs(args.out_dir, exist_ok=True)

    write_tables(agg, axes, metrics, invalid, args.out_dir, args.objective_set)
    write_contrasts(per_prompt, agg, axes, args.out_dir, args.objective_set,
                    args.n_boot, args.seed)
    if not args.no_figures:
        plot_pairwise_frontiers(
            agg, axes,
            os.path.join(args.out_dir, f"pareto_frontiers_{args.objective_set}.png"),
            args.objective_set,
        )

    with open(os.path.join(args.out_dir, f"frontier_metrics_{args.objective_set}.json"),
              "w", encoding="utf-8") as f:
        json.dump({"objective_set": args.objective_set, "axes": axes,
                   "invalid_counts": {f"{m}/{c}": n for (m, c), n in invalid.items()},
                   **metrics}, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  HHH PARETO SUMMARY — {args.objective_set} rewards, axes {axes}")
    print("=" * 78)
    # ASCII only in the console block: a cp1252 Windows terminal raises
    # UnicodeEncodeError on arrows and box-drawing characters.
    print(f"  {'Method':<22} {'HV(up)':>10} {'Delta(dn)':>10} {'front':>6} {'cfgs':>5} {'tok/s':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*6} {'-'*5} {'-'*8}")
    for m, d in metrics.get("methods", {}).items():
        delta = f"{d['spacing_delta']:.3f}" if d["spacing_delta"] is not None else "n/a"
        toks = f"{d['mean_tok_s']:.1f}" if d["mean_tok_s"] else "n/a"
        print(f"  {METHOD_STYLE.get(m,{}).get('label', m):<22} {d['hypervolume']:>10.4f} "
              f"{delta:>10} {d['n_frontier']:>6} {d['n_configs']:>5} {toks:>8}")
    if invalid:
        print(f"\n  invalid/unscored responses excluded: {sum(invalid.values())}")
    print("\n  HV    = hypervolume (optimality, higher better)")
    print("  Delta = Schott spacing on normalised frontier (uniformity, LOWER better)")
    print("  Read Delta next to coverage: a method that barely moves also scores low Delta.")
    print(f"\n  Artifacts -> {args.out_dir}/")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
