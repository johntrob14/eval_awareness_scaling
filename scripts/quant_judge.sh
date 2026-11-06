#!/bin/bash
# Judge quantization-study rollouts (QUANT_DATA/rollouts) with the unified v2 judges in ONE process, so the
# 27B judge loads once for every pass of every variant (eval_awareness.quant.judge_all; the passes match
# scripts/run_unified.sh). Judge: Qwen3.8-27B bf16 at the v2 revision.
#   VARIANTS="Qwen3.5-4B@bf16 Qwen3.5-4B@gptq-w4" TP=2 GPUS=0,1 scripts/quant_judge.sh
#   VARIANTS="Qwen3.5-4B@bf16.smoke" SPLITS=adversarial scripts/quant_judge.sh
set -u
cd "$(dirname "$0")/.."
: "${QUANT_DATA:?set QUANT_DATA (source .sauce/env.sh)}"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$QUANT_DATA/logs"
L=$QUANT_DATA/logs/judge_$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID:-local}.log
echo "judging ${VARIANTS} -> $L"
CUDA_VISIBLE_DEVICES=${GPUS:-0,1} uv run --no-sync python -m eval_awareness.quant.judge_all \
  --variants $VARIANTS --splits ${SPLITS:-adversarial benign} --tp "${TP:-2}" \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" --max-num-seqs "${MAX_NUM_SEQS:-256}" > "$L" 2>&1
status=$?
tail -3 "$L"
exit $status
