"""
Analyse overlap across the three PKU DPO objective JSONL files.

Reports, per field (prompt / chosen / rejected / full-triple):
  - size per objective (unique values)
  - pairwise overlap counts + Jaccard
  - three-way overlap count
  - percentage of each objective's rows that also appear in >=1 other objective

"Overlap" here is exact match after whitespace/case normalisation.

Usage:
  python analyse_overlap.py
"""

import os
import json
import hashlib
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt

DATA_DIR = "./pku_dpo_datasets"
PLOTS_DIR = os.path.join(DATA_DIR, "overlap_plots")
FILES = {
    "helpfulness":  os.path.join(DATA_DIR, "pku_helpfulness.jsonl"),
    "safety":       os.path.join(DATA_DIR, "pku_safety.jsonl"),
    "harmlessness": os.path.join(DATA_DIR, "pku_harmlessness.jsonl"),
}
OUT_FILE = os.path.join(DATA_DIR, "overlap_stats.json")
FIELDS = ["prompt", "chosen", "rejected", "triple"]

PALETTE = {
    "helpfulness":  "#2E86AB",
    "safety":       "#A23B72",
    "harmlessness": "#F18F01",
}
FIELD_COLOR = {
    "prompt":   "#4C9A2A",
    "chosen":   "#2E86AB",
    "rejected": "#C44536",
    "triple":   "#6C47B8",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def key_for(rec, field):
    if field == "triple":
        return h(norm(rec["prompt"]) + "\x1f" + norm(rec["chosen"]) + "\x1f" + norm(rec["rejected"]))
    return h(norm(rec[field]))


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(a, b):
    return (100.0 * a / b) if b else 0.0


def main():
    datasets = {}
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"WARN: {path} missing, skipping {name}")
            continue
        datasets[name] = load(path)
        print(f"{name:13s}: {len(datasets[name]):>7,} rows")

    if len(datasets) < 2:
        print("Need at least 2 objective files to compute overlap.")
        return

    objectives = list(datasets.keys())

    # sets of hashed keys per (objective, field); also row-level keys for "any-other-overlap" %
    sets = {obj: {f: set() for f in FIELDS} for obj in objectives}
    row_keys = {obj: {f: [] for f in FIELDS} for obj in objectives}

    for obj, rows in datasets.items():
        for r in rows:
            for f in FIELDS:
                k = key_for(r, f)
                sets[obj][f].add(k)
                row_keys[obj][f].append(k)

    stats = {
        "dataset_sizes": {o: len(datasets[o]) for o in objectives},
        "unique_counts": {o: {f: len(sets[o][f]) for f in FIELDS} for o in objectives},
        "pairwise": {},
        "three_way": {},
        "row_any_other_overlap_pct": {},
    }

    # pairwise + jaccard
    print("\n=== PAIRWISE OVERLAP ===")
    for a, b in combinations(objectives, 2):
        pair = f"{a} & {b}"
        stats["pairwise"][pair] = {}
        print(f"\n[{pair}]")
        for f in FIELDS:
            inter = sets[a][f] & sets[b][f]
            union = sets[a][f] | sets[b][f]
            jac = len(inter) / len(union) if union else 0.0
            stats["pairwise"][pair][f] = {
                "intersection": len(inter),
                "union": len(union),
                "jaccard": round(jac, 4),
                "pct_of_a": round(pct(len(inter), len(sets[a][f])), 2),
                "pct_of_b": round(pct(len(inter), len(sets[b][f])), 2),
            }
            print(f"  {f:8s}  inter={len(inter):>6,}  "
                  f"jaccard={jac:.4f}  "
                  f"{pct(len(inter), len(sets[a][f])):5.1f}% of {a}  "
                  f"{pct(len(inter), len(sets[b][f])):5.1f}% of {b}")

    # three-way
    if len(objectives) >= 3:
        print("\n=== THREE-WAY OVERLAP ===")
        a, b, c = objectives[:3]
        for f in FIELDS:
            inter3 = sets[a][f] & sets[b][f] & sets[c][f]
            union3 = sets[a][f] | sets[b][f] | sets[c][f]
            jac3 = len(inter3) / len(union3) if union3 else 0.0
            stats["three_way"][f] = {
                "intersection": len(inter3),
                "union": len(union3),
                "jaccard": round(jac3, 4),
                "pct_of_" + a: round(pct(len(inter3), len(sets[a][f])), 2),
                "pct_of_" + b: round(pct(len(inter3), len(sets[b][f])), 2),
                "pct_of_" + c: round(pct(len(inter3), len(sets[c][f])), 2),
            }
            print(f"  {f:8s}  inter={len(inter3):>6,}  jaccard3={jac3:.4f}  "
                  f"{pct(len(inter3), len(sets[a][f])):5.1f}% / "
                  f"{pct(len(inter3), len(sets[b][f])):5.1f}% / "
                  f"{pct(len(inter3), len(sets[c][f])):5.1f}%")

    # per-row "appears in any other objective" %
    print("\n=== ROW-LEVEL: % of rows whose key also appears in >=1 other objective ===")
    for obj in objectives:
        stats["row_any_other_overlap_pct"][obj] = {}
        others = [o for o in objectives if o != obj]
        for f in FIELDS:
            other_union = set().union(*(sets[o][f] for o in others))
            shared_rows = sum(1 for k in row_keys[obj][f] if k in other_union)
            p = pct(shared_rows, len(row_keys[obj][f]))
            stats["row_any_other_overlap_pct"][obj][f] = {
                "shared_rows": shared_rows,
                "total_rows": len(row_keys[obj][f]),
                "pct": round(p, 2),
            }
            print(f"  {obj:13s} {f:8s}  {shared_rows:>7,}/{len(row_keys[obj][f]):>7,}  ({p:5.1f}%)")

    # a few example prompts shared across all 3
    examples = []
    if len(objectives) >= 3:
        a, b, c = objectives[:3]
        shared_prompt_hashes = sets[a]["prompt"] & sets[b]["prompt"] & sets[c]["prompt"]
        by_prompt = defaultdict(dict)
        for obj in objectives[:3]:
            for r in datasets[obj]:
                k = key_for(r, "prompt")
                if k in shared_prompt_hashes and obj not in by_prompt[k]:
                    by_prompt[k][obj] = r
        for k, variants in list(by_prompt.items())[:3]:
            examples.append({
                "prompt": variants[objectives[0]]["prompt"][:240],
                "variants": {
                    o: {"chosen": v["chosen"][:160], "rejected": v["rejected"][:160]}
                    for o, v in variants.items()
                },
            })
    stats["shared_prompt_examples"] = examples

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_FILE}")

    make_plots(stats, objectives)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


def make_plots(stats, objectives):  # objectives kept for API symmetry
    del objectives
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\nGenerating overlap plots...")

    # One plot per field: pairwise Jaccard % (3 bars) + 3-way Jaccard % (1 bar).
    pair_keys = list(stats["pairwise"].keys())   # e.g. "helpfulness & safety"

    for field in FIELDS:
        labels = []
        vals = []
        colors = []

        # Pairwise: two bars per pair (intersection as % of each side)
        for pk in pair_keys:
            a, b = pk.split(" & ")
            d = stats["pairwise"][pk][field]
            labels.append(f"{a}∩{b}\n(% of {a})")
            vals.append(d["pct_of_a"])
            colors.append(PALETTE[a])
            labels.append(f"{a}∩{b}\n(% of {b})")
            vals.append(d["pct_of_b"])
            colors.append(PALETTE[b])

        # Three-way: one bar per objective (intersection as % of that objective)
        tw = stats["three_way"][field]
        for obj in ["helpfulness", "safety", "harmlessness"]:
            labels.append(f"all 3\n(% of {obj})")
            vals.append(tw.get(f"pct_of_{obj}", 0.0))
            colors.append(PALETTE[obj])

        fig, ax = plt.subplots(figsize=(12, 4.6))
        bars = ax.bar(labels, vals, color=colors, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("intersection as % of objective")
        ax.set_ylim(0, max(max(vals) * 1.2, 5))
        ax.set_title(f"Overlap % — {field}  (pairwise + 3-way)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        _save(fig, f"overlap_pct_{field}.png")


if __name__ == "__main__":
    main()
