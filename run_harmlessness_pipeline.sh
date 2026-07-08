#!/usr/bin/env bash
# =============================================================================
# Swiss Knife — Harmlessness Benchmark + GEval/AQI Scoring (no vLLM)
# =============================================================================
# Two-phase pipeline, run sequentially on one GPU box:
#
#   Phase 1: generate all 6 strategies' outputs on the same 100 harmless-base
#            prompts, with the harmlessness blade, saving raw text + steering
#            diagnostics (blade reward, refusal heuristic, override rate).
#            No judge model loaded in this phase — only base + blade models.
#
#   Phase 2: load a judge model directly via transformers (no vLLM, no
#            server) and score every saved response with GEval (harmfulness,
#            toxicity, refusal, response_quality, relevance) + the AQI
#            composite, writing one *_geval_aqi.json per strategy.
#
# Phases are separate processes so VRAM from phase 1's models is fully
# released (process exit) before phase 2 loads the judge — avoids needing
# 2x VRAM simultaneously for base+blade+judge all at once.
#
# Usage:
#   chmod +x run_harmlessness_pipeline.sh
#   ./run_harmlessness_pipeline.sh
#
# Override any setting via environment variables, e.g.:
#   NUM_PROMPTS=20 JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct ./run_harmlessness_pipeline.sh
# =============================================================================

set -euo pipefail

# ── CUDA allocator tuning ────────────────────────────────────────────────
# gsi_softmax/gsi_pairwise/gsi_swiss run many forward passes per prompt and
# can fragment the CUDA allocator over a long run (seen: OOM at prompt 50/100
# with only 4.29GB requested on a 47GB card). expandable_segments reduces
# fragmentation; this is the exact setting PyTorch's own OOM message suggests.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── GPU selection ────────────────────────────────────────────────────────
# You have 8x A6000 (48GB each) but everything currently runs on GPU 0 only
# (Model_mechanics doesn't shard across devices). Pick an idle GPU explicitly
# instead of always hitting GPU 0, which may already be loaded/fragmented
# from a previous run. Override with: CUDA_VISIBLE_DEVICES=3 ./run_harmlessness_pipeline.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Config (env-overridable) ────────────────────────────────────────────────
NUM_PROMPTS="${NUM_PROMPTS:-100}"
GSI_N="${GSI_N:-8}"
BETA="${BETA:-0.1}"
ALPHA="${ALPHA:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-200}"
BLADE="${BLADE:-harmlessness}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-runs/gsi_harmlessness_benchmark}"

JUDGE_BACKEND="${JUDGE_BACKEND:-local_hf}"          # local_hf | openai_compat
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"
USE_DETOXIFY="${USE_DETOXIFY:-0}"                   # 1 to enable, requires `pip install detoxify`
SKIP_EXISTING="${SKIP_EXISTING:-1}"                 # 1 = skip strategies already completed (resume after crash)

STRATEGIES=(
  baseline_base_model
  baseline_adapter
  original_swiss_knife
  gsi_softmax
  gsi_pairwise
  gsi_swiss
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/evaluation/benchmark_gsi_strategies_harmlessness.py"
GEVAL_SCRIPT="${SCRIPT_DIR}/run_geval_only.py"

echo "======================================================================"
echo "  Swiss Knife — Harmlessness Pipeline (no vLLM)"
echo "======================================================================"
echo "  Strategies     : ${STRATEGIES[*]}"
echo "  Blade          : ${BLADE}"
echo "  Prompts        : ${NUM_PROMPTS}  (Anthropic/hh-rlhf:harmless-base)"
echo "  gsi-n / beta   : ${GSI_N} / ${BETA}"
echo "  max-tokens     : ${MAX_TOKENS}"
echo "  Gen output dir : ${GEN_OUTPUT_DIR}"
echo "  Skip existing  : ${SKIP_EXISTING} (1 = resume, skip strategies already completed)"
echo "  Judge backend  : ${JUDGE_BACKEND} (${JUDGE_MODEL})"
echo "  Detoxify       : ${USE_DETOXIFY}"
echo "======================================================================"

# ── Sanity checks before burning GPU time ───────────────────────────────────
if [ ! -f "${GEN_SCRIPT}" ]; then
  echo "ERROR: ${GEN_SCRIPT} not found." >&2
  echo "       Place benchmark_gsi_strategies_harmlessness.py in evaluation/ first." >&2
  exit 1
fi
if [ ! -f "${GEVAL_SCRIPT}" ]; then
  echo "ERROR: ${GEVAL_SCRIPT} not found." >&2
  echo "       Place run_geval_only.py alongside this script (or update GEVAL_SCRIPT)." >&2
  exit 1
fi

python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ── Phase 1: generation + steering diagnostics ──────────────────────────────
echo ""
echo "── Phase 1/2: Generation (6 strategies x ${NUM_PROMPTS} prompts) ──────"
SKIP_FLAG=()
if [ "${SKIP_EXISTING}" = "1" ]; then
  SKIP_FLAG=(--skip-existing)
fi

python3 "${GEN_SCRIPT}" \
  --strategies "${STRATEGIES[@]}" \
  --num-prompts "${NUM_PROMPTS}" \
  --gsi-n "${GSI_N}" \
  --beta "${BETA}" \
  --alpha "${ALPHA}" \
  --max-tokens "${MAX_TOKENS}" \
  --blade "${BLADE}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --output-dir "${GEN_OUTPUT_DIR}" \
  "${SKIP_FLAG[@]}"

echo ""
echo "  Phase 1 complete. Raw generations + diagnostics in ${GEN_OUTPUT_DIR}/"

# ── Phase 2: GEval + AQI scoring (separate process -> frees GPU first) ─────
echo ""
echo "── Phase 2/2: GEval + AQI scoring (judge: ${JUDGE_MODEL}, no vLLM) ────"

DETOXIFY_FLAG=()
if [ "${USE_DETOXIFY}" = "1" ]; then
  DETOXIFY_FLAG=(--use-detoxify)
fi

python3 "${GEVAL_SCRIPT}" \
  --results-dir "${GEN_OUTPUT_DIR}" \
  --judge-backend "${JUDGE_BACKEND}" \
  --judge-model "${JUDGE_MODEL}" \
  --judge-dtype "${JUDGE_DTYPE}" \
  "${DETOXIFY_FLAG[@]}"

echo ""
echo "======================================================================"
echo "  Pipeline complete."
echo "  Raw generations + diagnostics : ${GEN_OUTPUT_DIR}/"
echo "  GEval + AQI scores            : ${GEN_OUTPUT_DIR}/geval_aqi_scored/"
echo "======================================================================"