#!/bin/bash
# Judge quantization-study rollouts (QUANT_DATA/rollouts) with the unified v2 judges in ONE process, so the
# 27B judge loads once for every pass of every variant (eval_awareness.quant.judge_all; the passes match
# scripts/run_unified.sh). Judge: Qwen3.8-27B bf16 at the v2 revision.
#   VARIANTS="Qwen3.5-4B@bf16 Qwen3.5-4B@gptq-w4" TP=2 GPUS=0,1 scripts/quant_judge.sh
#   VARIANTS="Qwen3.5-4B@bf16" TP=1 PP=3 GPUS=0,1,2 scripts/quant_judge.sh   # all three A100s of a node
# With PP > 1, a failed run falls back to TP=2 on the first two GPUs (resuming from the judgments written).
set -u
cd "$(dirname "$0")/.."
: "${QUANT_DATA:?set QUANT_DATA (source .sauce/env.sh)}"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$QUANT_DATA/logs"
run() {  # run <tp> <pp> <gpus>
  local L=$QUANT_DATA/logs/judge_$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID:-local}_tp$1pp$2.log
  echo "judging ${VARIANTS} (TP=$1 PP=$2 GPUs $3) -> $L"
  CUDA_VISIBLE_DEVICES=$3 uv run --no-sync python -m eval_awareness.quant.judge_all \
    --variants $VARIANTS --splits ${SPLITS:-adversarial benign} --tp "$1" --pp "$2" \
    --gpu-memory-utilization "${GPU_MEM:-0.90}" --max-num-seqs "${MAX_NUM_SEQS:-256}" > "$L" 2>&1
  local status=$?
  tail -3 "$L"
  return $status
}
TP=${TP:-2}; PP=${PP:-1}; GPUS=${GPUS:-0,1}
run "$TP" "$PP" "$GPUS" && exit 0
status=$?
if (( PP > 1 )); then
  echo "PP=$PP run failed (exit $status); falling back to TP=2 on GPUs ${GPUS%%,*},$(echo "$GPUS" | cut -d, -f2)"
  run 2 1 "$(echo "$GPUS" | cut -d, -f1-2)"
  exit $?
fi
exit $status
