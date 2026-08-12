#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_honesty.sh — DISCONNECT-SAFE honesty blade run.
#   Step 1: preprocess UltraFeedback honesty axis -> honesty_train/eval.jsonl
#   Step 2: DPO-train the honesty LoRA adapter
#
# The whole pipeline runs under `nohup ... &`, so it keeps going even if your
# SSH session drops. The outer script returns immediately and prints the PID
# and log path.
#
# Usage:
#   bash launch_honesty.sh
#   MODE=sanity bash launch_honesty.sh          # quick 0.3-epoch smoke test
#   GPU=1 BETA=0.15 bash launch_honesty.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ---- edit these ----------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/Swiss-Knife}"
ENV_ACTIVATE="${ENV_ACTIVATE:-}"      # e.g. "source $HOME/miniconda3/bin/activate sk"
GPU="${GPU:-0}"                       # CUDA device index
MODE="${MODE:-full}"                  # sanity | full
BETA="${BETA:-0.1}"                   # 0.1–0.15 recommended for honesty
# Preprocess flags: confound control on + TruthfulQA decontam (see script header)
PREPROCESS_ARGS="${PREPROCESS_ARGS:---max_confound_gap 1 --drop_truthfulqa}"
# --------------------------------------------------------------------------------

cd "$REPO_DIR"
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO_DIR/logs/honesty_${TS}.log"
PIDFILE="$REPO_DIR/logs/honesty.pid"

nohup bash -c "
  set -euo pipefail
  ${ENV_ACTIVATE:+$ENV_ACTIVATE; }
  export CUDA_VISIBLE_DEVICES=${GPU}
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo '================ [1/2] PREPROCESS: UltraFeedback honesty axis ================'
  python preprocess_honesty.py ${PREPROCESS_ARGS}

  echo '================ [2/2] TRAIN: DPO honesty blade =============================='
  python dpo_train_full.py --mode ${MODE} --objective honesty --beta ${BETA}

  echo '================ DONE ========================================================'
" > "$LOG" 2>&1 &

echo $! > "$PIDFILE"

cat <<EOF

Honesty run launched (survives disconnect).
  PID  : $(cat "$PIDFILE")
  Mode : ${MODE}   GPU: ${GPU}   beta: ${BETA}
  Log  : ${LOG}

  Follow : tail -f ${LOG}
  Check  : ps -p \$(cat ${PIDFILE}) -o pid,etime,cmd
  Stop   : kill \$(cat ${PIDFILE})
EOF
