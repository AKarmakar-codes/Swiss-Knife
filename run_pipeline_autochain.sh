#!/usr/bin/env bash
# run_pipeline_autochain.sh
# ---------------------------------------------------------------------------
# Waits for all N shard generation jobs to finish, then automatically runs:
#   1. --merge-shards   (creates tribunal/inputs/hhh_pareto/*.jsonl)
#   2. run_tribunal_judge_multigpu.py   (scores everything across all GPUs)
#
# Designed to run DETACHED (via tmux or setsid) so a Jupyter/container
# disconnect while you're off lab wifi does not kill it. See launch
# instructions at the bottom of this file.
#
# !!! DOUBLE-CHECK GRID BEFORE LEAVING THIS UNATTENDED !!!
# The --grid value here MUST exactly match what you used when you launched
# the original 8-shard generation command, or merge-shards will fail its
# prompt-count check and this whole chain aborts at step 1. Earlier commands
# in this session used --grid edges — RUN_PIPELINE.md's documented command
# uses --grid symmetric7. These are NOT the same grid. Set GRID below to
# whichever one you actually used to generate.
# ---------------------------------------------------------------------------

set -uo pipefail

# ── Config — edit these to match your actual run ────────────────────────────
REPO_ROOT="/LAB/Swiss-Knife"
NUM_SHARDS=8
GRID="symmetric7"                          # <-- MUST match your generation run. Not symmetric7 unless that's what you used.
METHODS="swiss mod rs args bon base"
PROMPTS="data/hhh_eval_prompts.jsonl"

SHARD_LOG_DIR="${REPO_ROOT}/runs/logs"
SHARD_LOG_PREFIX="pareto_shard"

TRIBUNAL_INPUT_DIR="${REPO_ROOT}/tribunal/inputs/hhh_pareto"
TRIBUNAL_OUTPUT_DIR="${REPO_ROOT}/tribunal/eval_results/hhh_pareto"
MULTIGPU_SCRIPT="${REPO_ROOT}/benchmarking/run_tribunal_judge_multigpu.py"
GPU_IDS="0 1 2 3 4 5 6 7"
WORKERS_PER_GPU=8

POLL_INTERVAL_S=60

# ── Logging ──────────────────────────────────────────────────────────────
CHAIN_LOG="${REPO_ROOT}/runs/logs/autochain.log"
mkdir -p "$(dirname "$CHAIN_LOG")"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$CHAIN_LOG"; }

cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }

log "=== autochain started (PID $$) ==="
log "Waiting for all $NUM_SHARDS shards to log 'Benchmark complete.' ..."

# ── Stage 0: poll until every shard log shows completion ───────────────────
while true; do
    done_count=0
    status_line=""
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        f="${SHARD_LOG_DIR}/${SHARD_LOG_PREFIX}_${i}.log"
        if [[ -f "$f" ]] && grep -q "Benchmark complete\." "$f"; then
            done_count=$((done_count + 1))
            status_line+="shard$i:OK "
        else
            last_line="(no log yet)"
            [[ -f "$f" ]] && last_line=$(tail -n 1 "$f" 2>/dev/null)
            status_line+="shard$i:running "
        fi
    done

    log "Progress: ${done_count}/${NUM_SHARDS} shards complete. [$status_line]"

    if [[ "$done_count" -eq "$NUM_SHARDS" ]]; then
        log "All shards complete."
        break
    fi

    sleep "$POLL_INTERVAL_S"
done

# ── Stage 1: merge shards ───────────────────────────────────────────────────
log "=== Stage 1: merge-shards (grid=$GRID) ==="
python benchmarking/benchmark_hhh_pareto.py \
    --prompts "$PROMPTS" \
    --methods $METHODS \
    --grid "$GRID" \
    --num-shards "$NUM_SHARDS" \
    --merge-shards \
    >> "$CHAIN_LOG" 2>&1
merge_rc=$?

if [[ $merge_rc -ne 0 ]]; then
    log "!!! merge-shards FAILED (exit $merge_rc). Aborting — NOT starting judge stage."
    log "!!! Check $CHAIN_LOG above for the traceback (common cause: --grid mismatch"
    log "!!! with what was used during generation, or a shard's prompt count off)."
    exit 1
fi

n_jsonl=$(find "$TRIBUNAL_INPUT_DIR" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
log "merge-shards succeeded. $n_jsonl .jsonl file(s) in $TRIBUNAL_INPUT_DIR"

if [[ "$n_jsonl" -eq 0 ]]; then
    log "!!! merge-shards exited 0 but produced 0 .jsonl files. Aborting — nothing to judge."
    exit 1
fi

# ── Stage 2: multi-GPU judge ────────────────────────────────────────────────
log "=== Stage 2: multi-GPU judge scoring ==="
python "$MULTIGPU_SCRIPT" \
    --input-dir "$TRIBUNAL_INPUT_DIR" \
    --output-dir "$TRIBUNAL_OUTPUT_DIR" \
    --gpu-ids $GPU_IDS \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    >> "$CHAIN_LOG" 2>&1
judge_rc=$?

if [[ $judge_rc -ne 0 ]]; then
    log "!!! Judge stage FAILED or was PARTIAL (exit $judge_rc). Check $CHAIN_LOG"
    log "!!! and $TRIBUNAL_OUTPUT_DIR/_multigpu_work/logs/ for per-GPU details."
    exit 1
fi

log "=== ALL STAGES COMPLETE ==="
log "Final report: ${TRIBUNAL_OUTPUT_DIR}/report/index.html"
log "Final tables: ${TRIBUNAL_OUTPUT_DIR}/model_summary.csv , ${TRIBUNAL_OUTPUT_DIR}/summary.csv"

# ---------------------------------------------------------------------------
# HOW TO LAUNCH THIS DETACHED (survives you leaving lab wifi / kernel disconnect)
# ---------------------------------------------------------------------------
# Option A — tmux (recommended, lets you reattach and watch live):
#   tmux new-session -d -s autochain 'bash /LAB/Swiss-Knife/run_pipeline_autochain.sh'
#   # reattach later:  tmux attach -t autochain
#   # just check progress without attaching:  tail -f /LAB/Swiss-Knife/runs/logs/autochain.log
#
# Option B — setsid + nohup (no tmux needed):
#   setsid nohup bash /LAB/Swiss-Knife/run_pipeline_autochain.sh < /dev/null \
#       >> /LAB/Swiss-Knife/runs/logs/autochain.log 2>&1 &
#   disown
#
# Either way, from anywhere you have SSH/container access later, just run:
#   tail -f /LAB/Swiss-Knife/runs/logs/autochain.log
# ---------------------------------------------------------------------------