"""
Preprocess PKU-SafeRLHF into three DISJOINT DPO datasets (v2).

v2 change vs v1: the three objectives now partition the raw dataset by the
safety-label pair, so no row appears in more than one objective and each
objective trains a distinct capability.

Objectives (mutually exclusive):
  helpfulness   -> (safe, safe) pairs
                   chosen = better_response_id
                   signal: quality among safe answers (no safety bleed-through)
  harmlessness  -> (safe, unsafe) pairs, severity gap >= SEVERITY_GAP_MIN
                   chosen = the safe response
                   signal: refuse when a harmful option exists
  safety        -> (unsafe, unsafe) pairs
                   chosen = safer_response_id (lower severity)
                   signal: harm minimisation when both options are bad

Output (JSONL, one record per line):
  {"prompt": "<chat-templated>", "chosen": "...", "rejected": "..."}
"""

import os
import json
from collections import Counter
from huggingface_hub import login, HfFolder
from datasets import load_dataset
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np

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
PALETTE = {
    "helpfulness":  "#2E86AB",
    "safety":       "#A23B72",
    "harmlessness": "#F18F01",
    "kept":         "#4C9A2A",
    "dropped":      "#C44536",
    "neutral":      "#6C757D",
}

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"
DATASET_SPLIT = "train"
OUT_DIR = "./pku_dpo_datasets_v2"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SEED = 42
SEVERITY_GAP_MIN = 1   # for harmlessness, require severity gap >= this

# --------------------------------------------------------------------------
# HF auth — non-interactive
# --------------------------------------------------------------------------
_token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
if _token:
    try:
        login(token=_token, add_to_git_credential=False)
        print("Authenticated with HuggingFace Hub (non-interactive).")
    except Exception as e:
        print(f"WARNING: login() failed ({e}). Continuing anonymously.")
else:
    print("No HF token found. Continuing anonymously (PKU-SafeRLHF is public).")

# --------------------------------------------------------------------------
# Tokenizer (for chat template only)
# --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

# --------------------------------------------------------------------------
# Load raw dataset
# --------------------------------------------------------------------------
print(f"Loading {DATASET_NAME} [{DATASET_SPLIT}] ...")
raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
print(f"Raw size: {len(raw)}")
print(f"Columns:  {raw.column_names}")

# --------------------------------------------------------------------------
# Per-objective builders (disjoint partition)
# --------------------------------------------------------------------------
printed_objective = set()
def build_record(example, objective):
    if objective not in printed_objective:
        print(f"\nFirst dispatch for objective '{objective}'")
        printed_objective.add(objective)

    r0, r1 = example["response_0"], example["response_1"]
    s0, s1 = example["is_response_0_safe"], example["is_response_1_safe"]

    if objective == "helpfulness":
        # only (safe, safe) pairs
        if not (s0 and s1):
            return None
        chosen_id = example["better_response_id"]

    elif objective == "harmlessness":
        # exactly one safe, with a severity gap
        if s0 == s1:
            return None
        sev0 = example.get("response_0_severity_level", 0)
        sev1 = example.get("response_1_severity_level", 0)
        if abs(sev0 - sev1) < SEVERITY_GAP_MIN:
            return None
        chosen_id = 0 if s0 else 1

    elif objective == "safety":
        # only (unsafe, unsafe) pairs -- harm minimisation
        if s0 or s1:
            return None
        chosen_id = example["safer_response_id"]

    else:
        raise ValueError(objective)

    chosen = r0 if chosen_id == 0 else r1
    rejected = r1 if chosen_id == 0 else r0
    if not chosen or not rejected or chosen == rejected:
        return None

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}


# --------------------------------------------------------------------------
# Write JSONL per objective
# --------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

safety_pair_counter = Counter()
severity_pair_counter = Counter()
severity_gap_counter = Counter()
harm_category_counter = Counter()
source_counter = Counter()
better_safer_agreement = Counter()
prompt_len_chars = []
chosen_len_chars_per_obj = {"helpfulness": [], "safety": [], "harmlessness": []}
rejected_len_chars_per_obj = {"helpfulness": [], "safety": [], "harmlessness": []}

stats = {}
objectives = ["helpfulness", "safety", "harmlessness"]
out_files = {obj: open(os.path.join(OUT_DIR, f"pku_{obj}.jsonl"), "w", encoding="utf-8")
             for obj in objectives}
counts = {obj: {"kept": 0, "dropped": 0} for obj in objectives}

# Sanity: track which objective each raw row lands in — must be <= 1
membership_counter = Counter()

print("\nStreaming raw dataset once for stats + 3 disjoint outputs...")
for ex in raw:
    s0, s1 = ex["is_response_0_safe"], ex["is_response_1_safe"]
    sev0 = ex.get("response_0_severity_level", 0)
    sev1 = ex.get("response_1_severity_level", 0)

    safety_pair_counter[("safe" if s0 else "unsafe", "safe" if s1 else "unsafe")] += 1
    severity_pair_counter[(sev0, sev1)] += 1
    if s0 != s1:
        severity_gap_counter[abs(sev0 - sev1)] += 1

    for key, val in (ex.get("response_0_harm_category") or {}).items():
        if val:
            harm_category_counter[key] += 1
    for key, val in (ex.get("response_1_harm_category") or {}).items():
        if val:
            harm_category_counter[key] += 1

    source_counter[ex.get("response_0_source", "unknown")] += 1
    source_counter[ex.get("response_1_source", "unknown")] += 1

    better_safer_agreement["agree" if ex["better_response_id"] == ex["safer_response_id"]
                           else "disagree"] += 1

    prompt_len_chars.append(len(ex["prompt"]))

    row_hits = 0
    for obj in objectives:
        rec = build_record(ex, obj)
        if rec is None:
            counts[obj]["dropped"] += 1
            continue
        out_files[obj].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[obj]["kept"] += 1
        chosen_len_chars_per_obj[obj].append(len(rec["chosen"]))
        rejected_len_chars_per_obj[obj].append(len(rec["rejected"]))
        row_hits += 1
    membership_counter[row_hits] += 1

for f in out_files.values():
    f.close()

for obj in objectives:
    k, d = counts[obj]["kept"], counts[obj]["dropped"]
    path = os.path.join(OUT_DIR, f"pku_{obj}.jsonl")
    stats[obj] = {"kept": k, "dropped": d, "path": path}
    print(f"[{obj:13s}]  kept={k:>7d}  dropped={d:>7d}  -> {path}")

print("\nRow-membership sanity (should be 0 or 1 per row; 2+ means overlap):")
for hits, c in sorted(membership_counter.items()):
    print(f"  rows assigned to {hits} objective(s): {c:,}")
assert max(membership_counter.keys()) <= 1, "Objectives are not disjoint!"

# --------------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------------
print("\nGenerating plots...")

def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


# (1) Per-objective kept vs dropped
fig, ax = plt.subplots(figsize=(7, 4.2))
kept_vals = [stats[o]["kept"] for o in objectives]
dropped_vals = [stats[o]["dropped"] for o in objectives]
x = np.arange(len(objectives))
ax.bar(x, kept_vals, color=PALETTE["kept"], label="kept", edgecolor="white")
ax.bar(x, dropped_vals, bottom=kept_vals, color=PALETTE["dropped"],
       label="dropped", edgecolor="white", alpha=0.85)
total = len(raw)
for i, (k, d) in enumerate(zip(kept_vals, dropped_vals)):
    pct = 100.0 * k / total
    ax.text(i, k / 2, f"{k:,}\n({pct:.1f}%)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(objectives)
ax.set_ylabel("examples")
ax.set_title(f"v2 disjoint partition — per-objective yield  (raw = {total:,})")
ax.legend(loc="upper right", frameon=False)
_save(fig, "01_objective_yield.png")


# (2) Safety-pair distribution annotated with objective assignment
fig, ax = plt.subplots(figsize=(7.2, 4.2))
labels_sp = [("safe", "safe"), ("safe", "unsafe"), ("unsafe", "safe"), ("unsafe", "unsafe")]
assignment = {
    ("safe", "safe"):     "helpfulness",
    ("safe", "unsafe"):   "harmlessness",
    ("unsafe", "safe"):   "harmlessness",
    ("unsafe", "unsafe"): "safety",
}
vals_sp = [safety_pair_counter.get(l, 0) for l in labels_sp]
colors_sp = [PALETTE[assignment[l]] for l in labels_sp]
bars = ax.bar([f"{a}/{b}" for a, b in labels_sp], vals_sp, color=colors_sp, edgecolor="white")
for b, v, lbl in zip(bars, vals_sp, labels_sp):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n→ {assignment[lbl]}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("pairs")
ax.set_xlabel("(response_0, response_1) safety")
ax.set_title("Safety pairs coloured by v2 objective assignment")
ax.set_ylim(0, max(vals_sp) * 1.22)
_save(fig, "02_safety_pair_assignment.png")


# (3) Severity heatmap
levels = [0, 1, 2, 3]
grid = np.zeros((4, 4), dtype=int)
for (a, b), c in severity_pair_counter.items():
    if a in levels and b in levels:
        grid[a, b] = c
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(grid, cmap="YlOrRd", aspect="equal")
for i in levels:
    for j in levels:
        v = grid[i, j]
        txt_color = "white" if v > grid.max() * 0.55 else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center", color=txt_color, fontsize=9)
ax.set_xticks(levels); ax.set_yticks(levels)
ax.set_xlabel("response_1 severity")
ax.set_ylabel("response_0 severity")
ax.set_title("Severity co-occurrence  (0 = safe … 3 = severe)")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.85, label="pair count")
_save(fig, "03_severity_heatmap.png")


# (4) Severity gap among harmlessness-eligible rows
fig, ax = plt.subplots(figsize=(6, 3.8))
gaps = sorted(severity_gap_counter.keys())
vals_g = [severity_gap_counter[g] for g in gaps]
bars = ax.bar(gaps, vals_g, color=PALETTE["harmlessness"], edgecolor="white")
for b, v in zip(bars, vals_g):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
            ha="center", va="bottom", fontsize=9)
ax.axvline(SEVERITY_GAP_MIN - 0.5, ls="--", color=PALETTE["dropped"], alpha=0.8,
           label=f"cutoff (gap ≥ {SEVERITY_GAP_MIN})")
ax.set_xlabel("|severity_0 − severity_1|")
ax.set_ylabel("pairs (exactly one safe)")
ax.set_title("Severity gap — harmlessness eligibility")
ax.set_xticks(gaps)
ax.legend(frameon=False)
_save(fig, "04_severity_gap.png")


# (5) Length distributions
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plen = np.array(prompt_len_chars)
clip = int(np.percentile(plen, 99))
axes[0].hist(np.clip(plen, 0, clip), bins=40, color=PALETTE["neutral"], edgecolor="white")
axes[0].set_title(f"Prompt length (chars, clipped at p99 = {clip})")
axes[0].set_xlabel("chars"); axes[0].set_ylabel("prompts")

width = 0.38
xs = np.arange(len(objectives))
chosen_means = [np.mean(chosen_len_chars_per_obj[o]) if chosen_len_chars_per_obj[o] else 0
                for o in objectives]
rejected_means = [np.mean(rejected_len_chars_per_obj[o]) if rejected_len_chars_per_obj[o] else 0
                  for o in objectives]
axes[1].bar(xs - width/2, chosen_means, width, color=PALETTE["kept"],
            label="chosen", edgecolor="white")
axes[1].bar(xs + width/2, rejected_means, width, color=PALETTE["dropped"],
            label="rejected", edgecolor="white")
for i, (cm, rm) in enumerate(zip(chosen_means, rejected_means)):
    axes[1].text(i - width/2, cm, f"{cm:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].text(i + width/2, rm, f"{rm:.0f}", ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(xs); axes[1].set_xticklabels(objectives)
axes[1].set_ylabel("mean chars")
axes[1].set_title("Mean chosen vs rejected length per objective")
axes[1].legend(frameon=False)
_save(fig, "05_length_distributions.png")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "version": 2,
        "scheme": "disjoint partition by safety-label pair",
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "raw_size": len(raw),
        "tokenizer": TOKENIZER_NAME,
        "severity_gap_min_for_harmlessness": SEVERITY_GAP_MIN,
        "assignment_rule": {
            "helpfulness":  "(safe, safe) pairs; chosen = better_response_id",
            "harmlessness": "(safe, unsafe) pairs with severity gap >= %d; chosen = safe response" % SEVERITY_GAP_MIN,
            "safety":       "(unsafe, unsafe) pairs; chosen = safer_response_id",
        },
        "objectives": stats,
        "row_membership_counts": dict(membership_counter),
        "safety_pair_counts": {f"{a}/{b}": c for (a, b), c in safety_pair_counter.items()},
        "preference_agreement": dict(better_safer_agreement),
        "top_harm_categories": harm_category_counter.most_common(),
        "source_counts": dict(source_counter),
    }, f, indent=2)
print(f"\nSummary written to {summary_path}")
print(f"Plots written to    {PLOTS_DIR}/")
