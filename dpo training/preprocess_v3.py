"""
Preprocess three disjoint DPO datasets for Swiss Knife auditor blades.

Sources (one per objective, no mixing):
  helpfulness     -> PKU-SafeRLHF (safe, safe) pairs,    chosen = better_response_id
  harmlessness    -> PKU-SafeRLHF (safe, unsafe) pairs,  chosen = the safe response
  informativeness -> UltraFeedback truthfulness-axis,    chosen = higher truthfulness rating

Output (JSONL, one record per line per objective):
  {"prompt": "<chat-templated>", "chosen": "...", "rejected": "...", "source": "pku"|"ultrafeedback"}

Run:
  python preprocess.py

Requires:
  pip install datasets transformers huggingface_hub matplotlib numpy
  HF_TOKEN in env OR `huggingface-cli login` previously run (both datasets are public anyway)
"""

import os
import json
import hashlib
from collections import Counter, defaultdict
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
    "helpfulness":     "#2E86AB",
    "harmlessness":    "#F18F01",
    "informativeness": "#6A4C93",
    "kept":            "#4C9A2A",
    "dropped":         "#C44536",
    "neutral":         "#6C757D",
}

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TOKENIZER_NAME       = "Qwen/Qwen2.5-7B-Instruct"
PKU_DATASET          = "PKU-Alignment/PKU-SafeRLHF"
PKU_SPLIT            = "train"
UF_DATASET           = "openbmb/UltraFeedback"
UF_SPLIT             = "train"

OUT_DIR              = "./dpo_datasets"
PLOTS_DIR            = os.path.join(OUT_DIR, "plots")
SEED                 = 42

# Informativeness: require this rating gap on the truthfulness axis
# UltraFeedback ratings are strings like "1"..."5"; gap >=2 keeps signal clean
UF_TRUTHFULNESS_GAP  = 2

# --------------------------------------------------------------------------
# HF auth (non-interactive; both datasets are public)
# --------------------------------------------------------------------------
_token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
if _token:
    try:
        login(token=_token, add_to_git_credential=False)
        print("Authenticated with HuggingFace Hub (non-interactive).")
    except Exception as e:
        print(f"WARNING: login() failed ({e}). Continuing anonymously.")
else:
    print("No HF token found. Continuing anonymously.")

# --------------------------------------------------------------------------
# Tokenizer (for chat template only)
# --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def chat_templated(prompt_str):
    """Single-turn user prompt through the Qwen chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_str}],
        tokenize=False,
        add_generation_prompt=True,
    )

def prompt_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# --------------------------------------------------------------------------
# 1) PKU  ->  helpfulness + harmlessness (disjoint by safety-label pair)
# --------------------------------------------------------------------------
print(f"\n=== Loading {PKU_DATASET} [{PKU_SPLIT}] ===")
pku = load_dataset(PKU_DATASET, split=PKU_SPLIT)
print(f"Raw PKU size: {len(pku)}")
print(f"Columns:      {pku.column_names}")

pku_safety_pair_counter = Counter()
pku_severity_pair_counter = Counter()
pku_harm_category_counter = Counter()
pku_prompt_len_chars = []

help_path = os.path.join(OUT_DIR, "helpfulness.jsonl")
harm_path = os.path.join(OUT_DIR, "harmlessness.jsonl")

pku_counts = {"helpfulness": {"kept": 0, "dropped": 0},
              "harmlessness": {"kept": 0, "dropped": 0}}
pku_lens = {"helpfulness":  {"chosen": [], "rejected": []},
            "harmlessness": {"chosen": [], "rejected": []}}
pku_membership = Counter()   # rows assigned to 0 or 1 objective (never >1)

with open(help_path, "w", encoding="utf-8") as fh_help, \
     open(harm_path, "w", encoding="utf-8") as fh_harm:

    for ex in pku:
        r0, r1 = ex["response_0"], ex["response_1"]
        s0, s1 = ex["is_response_0_safe"], ex["is_response_1_safe"]
        sev0 = ex.get("response_0_severity_level", 0) or 0
        sev1 = ex.get("response_1_severity_level", 0) or 0

        pku_safety_pair_counter[("safe" if s0 else "unsafe",
                                 "safe" if s1 else "unsafe")] += 1
        pku_severity_pair_counter[(sev0, sev1)] += 1
        pku_prompt_len_chars.append(len(ex["prompt"]))

        for d in (ex.get("response_0_harm_category") or {},
                  ex.get("response_1_harm_category") or {}):
            for k, v in d.items():
                if v:
                    pku_harm_category_counter[k] += 1

        hits = 0
        prompt_text = chat_templated(ex["prompt"])

        # helpfulness: (safe, safe) pairs only
        if s0 and s1:
            chosen_id = ex["better_response_id"]
            chosen, rejected = (r0, r1) if chosen_id == 0 else (r1, r0)
            if chosen and rejected and chosen != rejected:
                fh_help.write(json.dumps({
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "pku",
                }, ensure_ascii=False) + "\n")
                pku_counts["helpfulness"]["kept"] += 1
                pku_lens["helpfulness"]["chosen"].append(len(chosen))
                pku_lens["helpfulness"]["rejected"].append(len(rejected))
                hits += 1
            else:
                pku_counts["helpfulness"]["dropped"] += 1
        else:
            pku_counts["helpfulness"]["dropped"] += 1

        # harmlessness: exactly one safe
        if s0 != s1:
            chosen_id = 0 if s0 else 1
            chosen, rejected = (r0, r1) if chosen_id == 0 else (r1, r0)
            if chosen and rejected and chosen != rejected:
                fh_harm.write(json.dumps({
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "pku",
                }, ensure_ascii=False) + "\n")
                pku_counts["harmlessness"]["kept"] += 1
                pku_lens["harmlessness"]["chosen"].append(len(chosen))
                pku_lens["harmlessness"]["rejected"].append(len(rejected))
                hits += 1
            else:
                pku_counts["harmlessness"]["dropped"] += 1
        else:
            pku_counts["harmlessness"]["dropped"] += 1

        pku_membership[hits] += 1

print("\nPKU yield:")
for obj in ["helpfulness", "harmlessness"]:
    k, d = pku_counts[obj]["kept"], pku_counts[obj]["dropped"]
    print(f"  [{obj:15s}]  kept={k:>7d}  dropped={d:>7d}")
print(f"\nPKU row-membership sanity (should be 0 or 1; never 2):")
for hits, c in sorted(pku_membership.items()):
    print(f"  rows assigned to {hits} objective(s): {c:,}")
assert max(pku_membership.keys()) <= 1, "PKU objectives are not disjoint!"

# --------------------------------------------------------------------------
# 2) UltraFeedback  ->  informativeness (truthfulness-axis preference)
# --------------------------------------------------------------------------
print(f"\n=== Loading {UF_DATASET} [{UF_SPLIT}] ===")
uf = load_dataset(UF_DATASET, split=UF_SPLIT)
print(f"Raw UltraFeedback size: {len(uf)}")
print(f"Columns:                {uf.column_names}")

info_path = os.path.join(OUT_DIR, "informativeness.jsonl")
uf_counts = {"kept": 0, "dropped_no_truthfulness": 0,
             "dropped_small_gap": 0, "dropped_degenerate": 0}
uf_gap_hist = Counter()
uf_lens = {"chosen": [], "rejected": []}
uf_prompt_len_chars = []

def _truthfulness_rating(completion):
    """Extract truthfulness rating as int 1..5, or None if missing/unparseable.

    UltraFeedback rows have completions[i]["annotations"]["truthfulness"]["Rating"]
    as a string. Some rows have 'N/A' or missing entries.
    """
    try:
        ann = completion.get("annotations") or {}
        truth = ann.get("truthfulness") or {}
        rating = truth.get("Rating")
        if rating is None:
            return None
        rating = str(rating).strip()
        if rating in {"N/A", ""}:
            return None
        return int(rating)
    except (ValueError, AttributeError, TypeError):
        return None

with open(info_path, "w", encoding="utf-8") as fh_info:
    for ex in uf:
        uf_prompt_len_chars.append(len(ex["instruction"]))
        completions = ex.get("completions") or []
        scored = []
        for c in completions:
            r = _truthfulness_rating(c)
            resp = c.get("response")
            if r is not None and resp:
                scored.append((r, resp))

        if len(scored) < 2:
            uf_counts["dropped_no_truthfulness"] += 1
            continue

        scored.sort(key=lambda x: x[0])
        lo_rating, lo_resp = scored[0]
        hi_rating, hi_resp = scored[-1]
        gap = hi_rating - lo_rating
        uf_gap_hist[gap] += 1

        if gap < UF_TRUTHFULNESS_GAP:
            uf_counts["dropped_small_gap"] += 1
            continue
        if hi_resp == lo_resp:
            uf_counts["dropped_degenerate"] += 1
            continue

        prompt_text = chat_templated(ex["instruction"])
        fh_info.write(json.dumps({
            "prompt": prompt_text,
            "chosen": hi_resp,
            "rejected": lo_resp,
            "source": "ultrafeedback",
        }, ensure_ascii=False) + "\n")
        uf_counts["kept"] += 1
        uf_lens["chosen"].append(len(hi_resp))
        uf_lens["rejected"].append(len(lo_resp))

print("\nUltraFeedback yield:")
for k, v in uf_counts.items():
    print(f"  {k:30s} {v:>7d}")

# --------------------------------------------------------------------------
# 3) Prompt-overlap sanity: are any prompts shared across the 3 files?
#    We expect near-zero overlap; report it if present.
# --------------------------------------------------------------------------
def prompt_set(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            s.add(prompt_hash(rec["prompt"]))
    return s

ph = {
    "helpfulness":     prompt_set(help_path),
    "harmlessness":    prompt_set(harm_path),
    "informativeness": prompt_set(info_path),
}
overlap_report = {}
keys = list(ph.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        inter = len(ph[a] & ph[b])
        overlap_report[f"{a} ∩ {b}"] = inter

print("\nCross-objective prompt overlap (by prompt hash):")
for k, v in overlap_report.items():
    print(f"  {k}: {v}")

# --------------------------------------------------------------------------
# 4) Plots
# --------------------------------------------------------------------------
print("\nGenerating plots...")

def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")

# (1) Per-objective yield
fig, ax = plt.subplots(figsize=(7.5, 4.2))
objs = ["helpfulness", "harmlessness", "informativeness"]
kept_vals = [pku_counts["helpfulness"]["kept"],
             pku_counts["harmlessness"]["kept"],
             uf_counts["kept"]]
colors = [PALETTE[o] for o in objs]
bars = ax.bar(objs, kept_vals, color=colors, edgecolor="white")
for b, v in zip(bars, kept_vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("kept pairs")
ax.set_title("DPO pairs per objective (one source each)")
for i, obj in enumerate(objs):
    source = "PKU" if obj != "informativeness" else "UltraFeedback"
    ax.text(i, -max(kept_vals) * 0.08, source, ha="center",
            fontsize=9, style="italic", color="#444")
ax.set_ylim(0, max(kept_vals) * 1.15)
_save(fig, "01_yield.png")

# (2) PKU safety-pair distribution with objective assignment
fig, ax = plt.subplots(figsize=(7, 4.2))
labels_sp = [("safe", "safe"), ("safe", "unsafe"),
             ("unsafe", "safe"), ("unsafe", "unsafe")]
assignment = {
    ("safe", "safe"):     "helpfulness",
    ("safe", "unsafe"):   "harmlessness",
    ("unsafe", "safe"):   "harmlessness",
    ("unsafe", "unsafe"): "(unused — safety dropped)",
}
vals_sp = [pku_safety_pair_counter.get(l, 0) for l in labels_sp]
colors_sp = [PALETTE.get(assignment[l], PALETTE["neutral"]) for l in labels_sp]
bars = ax.bar([f"{a}/{b}" for a, b in labels_sp], vals_sp,
              color=colors_sp, edgecolor="white")
for b, v, lbl in zip(bars, vals_sp, labels_sp):
    ax.text(b.get_x() + b.get_width() / 2, v,
            f"{v:,}\n→ {assignment[lbl]}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("pairs")
ax.set_xlabel("(response_0, response_1) safety")
ax.set_title("PKU safety-pair → objective assignment")
ax.set_ylim(0, max(vals_sp) * 1.28)
_save(fig, "02_pku_safety_assignment.png")

# (3) UltraFeedback truthfulness-gap histogram
fig, ax = plt.subplots(figsize=(6.5, 4))
gaps = sorted(uf_gap_hist.keys())
vals_g = [uf_gap_hist[g] for g in gaps]
bars = ax.bar(gaps, vals_g, color=PALETTE["informativeness"], edgecolor="white")
for b, v in zip(bars, vals_g):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
            ha="center", va="bottom", fontsize=8)
ax.axvline(UF_TRUTHFULNESS_GAP - 0.5, ls="--", color=PALETTE["dropped"],
           alpha=0.8, label=f"cutoff (gap ≥ {UF_TRUTHFULNESS_GAP})")
ax.set_xlabel("max truthfulness rating − min truthfulness rating  (per prompt)")
ax.set_ylabel("prompts")
ax.set_title("UltraFeedback: truthfulness-axis rating gap")
ax.set_xticks(gaps)
ax.legend(frameon=False)
_save(fig, "03_uf_truthfulness_gap.png")

# (4) Length distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
# 4a prompt length per objective
for ax_, lens, label, color in [
    (axes[0], pku_prompt_len_chars, "PKU prompts (help+harm)", PALETTE["neutral"]),
    (axes[0], uf_prompt_len_chars, "UltraFeedback prompts (info)", PALETTE["informativeness"]),
]:
    arr = np.array(lens)
    clip = int(np.percentile(arr, 99))
    axes[0].hist(np.clip(arr, 0, clip), bins=40, alpha=0.6,
                 label=label, color=color, edgecolor="white")
axes[0].set_xlabel("chars (clipped at p99)")
axes[0].set_ylabel("prompts")
axes[0].set_title("Prompt length distribution")
axes[0].legend(frameon=False)

# 4b mean chosen/rejected length per objective
width = 0.38
xs = np.arange(3)
chosen_means = [
    float(np.mean(pku_lens["helpfulness"]["chosen"])) if pku_lens["helpfulness"]["chosen"] else 0,
    float(np.mean(pku_lens["harmlessness"]["chosen"])) if pku_lens["harmlessness"]["chosen"] else 0,
    float(np.mean(uf_lens["chosen"])) if uf_lens["chosen"] else 0,
]
rejected_means = [
    float(np.mean(pku_lens["helpfulness"]["rejected"])) if pku_lens["helpfulness"]["rejected"] else 0,
    float(np.mean(pku_lens["harmlessness"]["rejected"])) if pku_lens["harmlessness"]["rejected"] else 0,
    float(np.mean(uf_lens["rejected"])) if uf_lens["rejected"] else 0,
]
axes[1].bar(xs - width/2, chosen_means, width, color=PALETTE["kept"],
            label="chosen", edgecolor="white")
axes[1].bar(xs + width/2, rejected_means, width, color=PALETTE["dropped"],
            label="rejected", edgecolor="white")
for i, (cm, rm) in enumerate(zip(chosen_means, rejected_means)):
    axes[1].text(i - width/2, cm, f"{cm:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].text(i + width/2, rm, f"{rm:.0f}", ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(xs); axes[1].set_xticklabels(objs)
axes[1].set_ylabel("mean chars")
axes[1].set_title("Mean chosen vs rejected length per objective")
axes[1].legend(frameon=False)
_save(fig, "04_length_distributions.png")

# (5) Top PKU harm categories (for context, not training)
fig, ax = plt.subplots(figsize=(7, 4.5))
top = pku_harm_category_counter.most_common(14)
if top:
    names = [k for k, _ in top][::-1]
    freqs = [v for _, v in top][::-1]
    ax.barh(names, freqs, color=PALETTE["harmlessness"], edgecolor="white")
    for i, v in enumerate(freqs):
        ax.text(v, i, f"  {v:,}", va="center", fontsize=8)
    ax.set_xlabel("responses flagged (either slot)")
    ax.set_title("PKU harm categories (reference, not a training signal)")
_save(fig, "05_pku_harm_categories.png")

# --------------------------------------------------------------------------
# 5) Summary JSON
# --------------------------------------------------------------------------
summary = {
    "scheme": "three disjoint objectives, one source per objective",
    "assignment": {
        "helpfulness":     "PKU (safe, safe) pairs; chosen = better_response_id",
        "harmlessness":    "PKU (safe, unsafe) pairs; chosen = safe response",
        "informativeness": f"UltraFeedback truthfulness-axis; chosen = highest-rated, gap >= {UF_TRUTHFULNESS_GAP}",
    },
    "tokenizer": TOKENIZER_NAME,
    "seed": SEED,
    "pku": {
        "raw_size": len(pku),
        "helpfulness_kept":  pku_counts["helpfulness"]["kept"],
        "harmlessness_kept": pku_counts["harmlessness"]["kept"],
        "safety_pair_counts": {f"{a}/{b}": c
                               for (a, b), c in pku_safety_pair_counter.items()},
        "row_membership":   dict(pku_membership),
    },
    "ultrafeedback": {
        "raw_size": len(uf),
        **uf_counts,
        "truthfulness_gap_histogram": dict(uf_gap_hist),
    },
    "prompt_overlap_across_objectives": overlap_report,
    "paths": {
        "helpfulness":     help_path,
        "harmlessness":    harm_path,
        "informativeness": info_path,
    },
}
with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary -> {os.path.join(OUT_DIR, 'summary.json')}")
print(f"Plots   -> {PLOTS_DIR}/")
print("\nDone.")