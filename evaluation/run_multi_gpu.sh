#!/usr/bin/env bash
# Swiss Knife — Multi-GPU sharded generation launcher
# =====================================================
# Splits the prompt set across all visible GPUs and runs one independent
# process per GPU, each pinned via CUDA_VISIBLE_DEVICES. This is plain
# data parallelism over prompts (not model parallelism) — after the
# blades.py / models.py fixes, one full pipeline replica (drafter +
# shared base/blade weights) fits comfortably in ~21GB, so an A6000 (48GB)
# can run one replica with plenty of headroom; there's no need to split a
# single model across GPUs.
#
# Usage:
#   ./evaluation/run_multi_gpu.sh 8 50 768
#   (num_gpus, num_samples, max_new_tokens)

set -euo pipefail

NUM_GPUS="${1:-8}"
NUM_SAMPLES="${2:-50}"
MAX_NEW_TOKENS="${3:-768}"

echo "Launching ${NUM_GPUS} shards for ${NUM_SAMPLES} prompts..."

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    echo "  shard $i -> GPU $i"
    CUDA_VISIBLE_DEVICES="$i" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python evaluation/run_hh_experiments.py \
            --test all \
            --mode generate \
            --num_samples "$NUM_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --num-shards "$NUM_GPUS" \
            --shard-id "$i" \
            > "runs/logs/shard_${i}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for all ${NUM_GPUS} shards to finish (tail -f runs/logs/shard_*.log to watch)..."
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
    echo "One or more shards failed — check runs/logs/shard_*.log" >&2
    exit 1
fi

echo "All shards complete. Merging..."
python evaluation/merge_shards.py --num-shards "$NUM_GPUS"
echo "Done. Next: start the Tribunal judge, then run --mode analyze as usual."
