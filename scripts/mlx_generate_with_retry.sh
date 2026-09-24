#!/usr/bin/env bash
# Apple Silicon only: run MLX generation, halving the batch size after each Metal OOM crash (generation resumes from the JSONL).
# usage: scripts/mlx_generate_with_retry.sh <model> <n_rollouts> [start_batch_size]
set -u
model=$1; n=$2; bs=${3:-16}
until uv run python -m eval_awareness.generate --backend mlx --model "$model" --n-rollouts "$n" --batch-size "$bs"; do
  bs=$(( bs > 4 ? bs / 2 : 4 ))
  echo "crashed; restarting with batch size $bs"
  sleep 10
done
