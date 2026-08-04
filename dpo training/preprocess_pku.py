"""
Preprocess PKU-SafeRLHF into three DPO datasets (one per objective)
and save each as a JSONL file with chat-templated prompts.

Objectives:
  helpfulness  -> keep pairs where BOTH responses are safe;
                  chosen = better_response_id
  safety       -> keep all pairs;
                  chosen = safer_response_id
  harmlessness -> keep pairs where exactly ONE response is safe;
                  chosen = the safe one

Output (JSONL, one record per line):
  {"prompt": "<chat-templated>", "chosen": "...", "rejected": "..."}

Prereqs:
  - HF token already cached on this machine (e.g. `huggingface-cli login` run previously).
    login() will pick it up from ~/.cache/huggingface/token automatically.
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
OUT_DIR = "./pku_dpo_datasets"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SEED = 42
SEVERITY_GAP_MIN = 1   # for harmlessness, require severity gap >= this

# --------------------------------------------------------------------------
# HF auth — non-interactive
# --------------------------------------------------------------------------
# Never prompt. Prefer HF_TOKEN env var, then the cached token written by
# `huggingface-cli login`. PKU-SafeRLHF is public, so login is optional anyway.
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
# Per-objective builders
# --------------------------------------------------------------------------
printed_objective = set()
def build_record(example, objective):

    if objective not in printed_objective:
        print(f"\nExample record for objective '{objective}':")
        printed_objective.add(objective)

    r0, r1 = example["response_0"], example["response_1"]
    s0, s1 = example["is_response_0_safe"], example["is_response_1_safe"]

    if objective == "helpfulness":
        if not (s0 and s1):
            return None
        chosen_id = example["better_response_id"]

    elif objective == "safety":
        chosen_id = example["safer_response_id"]

    elif objective == "harmlessness":
        if s0 == s1:
            return None
        sev0 = example.get("response_0_severity_level", 0)
        sev1 = example.get("response_1_severity_level", 0)
        if abs(sev0 - sev1) < SEVERITY_GAP_MIN:
            return None
        chosen_id = 0 if s0 else 1

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

# Aggregators for dataset-level analysis (single pass over raw)
safety_pair_counter = Counter()              # ("safe","safe")/("safe","unsafe")/... on (s0,s1)
severity_pair_counter = Counter()            # (sev0, sev1)
severity_gap_counter = Counter()             # abs(sev0-sev1) on unsafe-vs-safe rows
harm_category_counter = Counter()            # union across both responses
source_counter = Counter()                   # r0 + r1 sources pooled
better_safer_agreement = Counter()           # (better_id == safer_id)
prompt_len_chars = []
chosen_len_chars_per_obj = {"helpfulness": [], "safety": [], "harmlessness": []}
rejected_len_chars_per_obj = {"helpfulness": [], "safety": [], "harmlessness": []}

stats = {}
objectives = ["helpfulness", "safety", "harmlessness"]
out_files = {obj: open(os.path.join(OUT_DIR, f"pku_{obj}.jsonl"), "w", encoding="utf-8")
             for obj in objectives}
counts = {obj: {"kept": 0, "dropped": 0} for obj in objectives}

print("\nStreaming raw dataset once for stats + 3 outputs...")
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


    for obj in objectives:
        rec = build_record(ex, obj)
        if rec is None:
            counts[obj]["dropped"] += 1
            continue
        out_files[obj].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[obj]["kept"] += 1
        chosen_len_chars_per_obj[obj].append(len(rec["chosen"]))
        rejected_len_chars_per_obj[obj].append(len(rec["rejected"]))

for f in out_files.values():
    f.close()

for obj in objectives:
    k, d = counts[obj]["kept"], counts[obj]["dropped"]
    path = os.path.join(OUT_DIR, f"pku_{obj}.jsonl")
    stats[obj] = {"kept": k, "dropped": d, "path": path}
    print(f"[{obj:13s}]  kept={k:>7d}  dropped={d:>7d}  -> {path}")

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


# (1) Per-objective kept vs dropped — stacked bar showing yield
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
ax.set_title(f"Per-objective filter yield  (raw = {total:,})")
ax.legend(loc="upper right", frameon=False)
_save(fig, "01_objective_yield.png")


# (2) Safety-pair distribution (s0, s1)
fig, ax = plt.subplots(figsize=(6.5, 4))
labels_sp = [("safe", "safe"), ("safe", "unsafe"), ("unsafe", "safe"), ("unsafe", "unsafe")]
vals_sp = [safety_pair_counter.get(l, 0) for l in labels_sp]
colors_sp = ["#4C9A2A", "#F1B24A", "#F1B24A", "#C44536"]
bars = ax.bar([f"{a}/{b}" for a, b in labels_sp], vals_sp, color=colors_sp, edgecolor="white")
for b, v in zip(bars, vals_sp):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100*v/total:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("pairs")
ax.set_xlabel("(response_0, response_1) safety")
ax.set_title("Safety label distribution across the two responses")
ax.set_ylim(0, max(vals_sp) * 1.18)
_save(fig, "02_safety_pair_distribution.png")


# (3) Severity heatmap (sev0 x sev1)
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


# (4) Severity gap among harmlessness-eligible rows (exactly one safe)
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


# (5) Harm-category frequency (top N, horizontal)
fig, ax = plt.subplots(figsize=(7.2, 4.8))
top_items = harm_category_counter.most_common(14)
if top_items:
    names = [k for k, _ in top_items][::-1]
    freqs = [v for _, v in top_items][::-1]
    ax.barh(names, freqs, color=PALETTE["safety"], edgecolor="white")
    for i, v in enumerate(freqs):
        ax.text(v, i, f"  {v:,}", va="center", fontsize=8)
    ax.set_xlabel("responses flagged (either response_0 or response_1)")
    ax.set_title("Harm-category frequency")
_save(fig, "05_harm_categories.png")


# (6) Helpfulness ↔ Safety preference agreement
fig, ax = plt.subplots(figsize=(4.6, 4.2))
labels_a = ["agree", "disagree"]
vals_a = [better_safer_agreement.get(k, 0) for k in labels_a]
colors_a = [PALETTE["kept"], PALETTE["dropped"]]
ax.pie(vals_a, labels=[f"{l}\n{v:,} ({100*v/sum(vals_a):.1f}%)" for l, v in zip(labels_a, vals_a)],
       colors=colors_a, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("better_response_id  vs  safer_response_id")
_save(fig, "06_preference_agreement.png")


# (7) Response-source distribution
fig, ax = plt.subplots(figsize=(6.8, 3.6))
src_items = source_counter.most_common()
if src_items:
    names = [k for k, _ in src_items]
    vals_s = [v for _, v in src_items]
    bars = ax.bar(names, vals_s, color=PALETTE["neutral"], edgecolor="white")
    for b, v in zip(bars, vals_s):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("responses")
    ax.set_title("Response source model distribution (pooled over both slots)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
_save(fig, "07_source_distribution.png")


# (8) Length distributions: prompt + chosen/rejected per objective
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# 8a: prompt length (clipped to 99th pctile for readability)
plen = np.array(prompt_len_chars)
clip = int(np.percentile(plen, 99))
axes[0].hist(np.clip(plen, 0, clip), bins=40, color=PALETTE["neutral"],
             edgecolor="white")
axes[0].set_title(f"Prompt length (chars, clipped at p99 = {clip})")
axes[0].set_xlabel("chars"); axes[0].set_ylabel("prompts")

# 8b: chosen vs rejected mean length per objective
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
_save(fig, "08_length_distributions.png")


# (9) Summary dashboard — one PNG giving the whole picture at a glance
fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.4)

axA = fig.add_subplot(gs[0, 0])
axA.bar(objectives, kept_vals, color=[PALETTE[o] for o in objectives], edgecolor="white")
for i, v in enumerate(kept_vals):
    axA.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
axA.set_title("Kept per objective")
axA.set_ylabel("examples")

axB = fig.add_subplot(gs[0, 1])
axB.bar([f"{a}/{b}" for a, b in labels_sp], vals_sp, color=colors_sp, edgecolor="white")
axB.set_title("Safety-pair distribution")
plt.setp(axB.get_xticklabels(), rotation=20, ha="right")

axC = fig.add_subplot(gs[0, 2])
axC.pie(vals_a, labels=labels_a, colors=colors_a, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        autopct="%1.0f%%")
axC.set_title("Helpfulness vs safety\npreference agreement")

axD = fig.add_subplot(gs[1, 0:2])
if top_items:
    top8 = harm_category_counter.most_common(8)
    names8 = [k for k, _ in top8][::-1]
    freqs8 = [v for _, v in top8][::-1]
    axD.barh(names8, freqs8, color=PALETTE["safety"], edgecolor="white")
    axD.set_title("Top harm categories")

axE = fig.add_subplot(gs[1, 2])
im2 = axE.imshow(grid, cmap="YlOrRd", aspect="equal")
axE.set_xticks(levels); axE.set_yticks(levels)
axE.set_title("Severity heatmap")
axE.set_xlabel("sev₁"); axE.set_ylabel("sev₀")
axE.grid(False)

fig.suptitle("PKU-SafeRLHF — preprocessing dashboard", fontsize=14, fontweight="bold")
_save(fig, "00_dashboard.png")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "raw_size": len(raw),
        "tokenizer": TOKENIZER_NAME,
        "severity_gap_min_for_harmlessness": SEVERITY_GAP_MIN,
        "objectives": stats,
        "safety_pair_counts": {f"{a}/{b}": c for (a, b), c in safety_pair_counter.items()},
        "preference_agreement": dict(better_safer_agreement),
        "top_harm_categories": harm_category_counter.most_common(),
        "source_counts": dict(source_counter),
    }, f, indent=2)
print(f"\nSummary written to {summary_path}")
print(f"Plots written to    {PLOTS_DIR}/")
