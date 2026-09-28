#!/usr/bin/env bash
# Overnight driver (resumable; rerun to continue):
#   0. Nemotron intervention `none` condition (generate on GPUs 0-3; judge on 4-5 while Qwen runs)
#   1. Qwen3.5-122B-A10B-FP8 through scripts/run_qwen35.sh (generate on 0-3, judge on 4-5)
#   2. Qwen3.5-397B-A17B-FP8 (all 6 GPUs; judging in 3 GPU pairs)
# Qwen weights live on /ssd1 and are symlinked into the /home HF cache. If a model's smoke test fails,
# that model's remaining steps are skipped (no cascade of failing steps) and the next block still runs.
set -uo pipefail
cd "$(dirname "$0")/.."
H=/home/jtr2875/.cache/huggingface/hub
S=/ssd1/john/.cache/huggingface/hub
export HF_HUB_CACHE=$H VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
log() { echo "[$(date '+%F %T')] $*"; }
run() { log "run: $*"; env "$@" scripts/run_qwen35.sh; local rc=$?; log "exit $rc: $*"; return $rc; }
link() {  # link a fully downloaded /ssd1 model into the /home cache
  local dir=models--${1//\//--}
  if [[ ! -L $H/$dir ]]; then [[ -d $H/$dir ]] && mv "$H/$dir" "/tmp/$dir.stub.$$"; ln -s "$S/$dir" "$H/$dir"; fi
}
wait_gpus_free() {  # until no process holds more than 4 GB on any of the given GPUs
  while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$1" | awk -F', ' '$2>4000{f=1} END{exit !f}'; do
    sleep 60; done
}

# ---------------------------------------------------------------- 0. Nemotron `none`
NEMO="--model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --dir results/nemotron3_super"
NC=results/nemotron3_super/intervention/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.continuations.jsonl
nemo_judge() {
  for task in refusal vea_cal; do
    extra=$([[ $task == vea_cal ]] && echo "--judge-thinking low --judge-max-tokens 4096" || echo "--judge-max-tokens 1024")
    CUDA_VISIBLE_DEVICES=4,5 uv run python -m eval_awareness.judge $NC --task $task --split adversarial --backend vllm \
      --judge-model Qwen/Qwen3.8-27B --tp 2 --text-only --no-prefilter --max-model-len 40960 \
      --out-dir results/nemotron3_super/intervention/judgments $extra >> logs/intervene_judge_none.log 2>&1
  done
  uv run python -m eval_awareness.intervene analyze $NEMO > logs/intervene_analyze.log 2>&1
  log "Nemotron none: judged and analyzed"
}
log "Nemotron none: generate"
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m eval_awareness.intervene generate $NEMO --tp 4 >> logs/intervene_gen_none.log 2>&1
log "Nemotron none: generate exit $?"
nemo_judge &
NEMO_PID=$!

# ---------------------------------------------------------------- 1. 122B-A10B
qwen122() {
  local M="MODEL=Qwen/Qwen3.5-122B-A10B-FP8 D=results/qwen35_122b" G="GPUS=0,1,2,3 TP=4"
  link Qwen/Qwen3.5-122B-A10B-FP8
  run $M $G STEPS=smoke || return 1
  run $M $G STEPS="gen retry" || return 1
  wait $NEMO_PID  # Nemotron judging holds GPUs 4-5
  run $M $G STEPS=cap &
  run $M JUDGE_GPUS=4,5 JUDGE_SPLITS=adversarial STEPS=judge
  wait
  run $M JUDGE_GPUS=4,5 JUDGE_SPLITS="gsm8k math500" JUDGE_TASKS="vea_cal category" STEPS=judge &
  run $M $G STEPS="analyze iprep igen"
  wait
  run $M JUDGE_GPUS=4,5 JUDGE_TASKS=vea_cal STEPS=ijudge &
  run $M JUDGE_GPUS=2,3 JUDGE_TASKS=refusal STEPS=ijudge
  wait
  run $M STEPS="analyze ianalyze"
}
qwen122 || log "122B block stopped early"
wait  # anything still running from block 1 (and Nemotron judging)

# ---------------------------------------------------------------- 2. 397B-A17B (all 6 GPUs)
qwen397() {
  local M="MODEL=Qwen/Qwen3.5-397B-A17B-FP8 D=results/qwen35_397b" G="GPUS=0,1,2,3,4,5 TP=2 PP=3"
  link Qwen/Qwen3.5-397B-A17B-FP8
  wait_gpus_free 0,1,2,3,4,5
  if ! run $M $G STEPS=smoke; then
    log "397B smoke failed with TP=2 PP=3; retrying TP=1 PP=6 MAX_NUM_SEQS=256"
    rm -f results/qwen35_397b/rollouts/*.adversarial.jsonl
    G="GPUS=0,1,2,3,4,5 TP=1 PP=6 MAX_NUM_SEQS=256"
    run $M $G STEPS=smoke || return 1
  fi
  run $M $G STEPS="gen retry cap" || return 1
  run $M JUDGE_GPUS=0,1 JUDGE_SPLITS=adversarial JUDGE_TASKS="vea_cal category" STEPS=judge &
  run $M JUDGE_GPUS=2,3 JUDGE_SPLITS=adversarial JUDGE_TASKS=refusal STEPS=judge &
  run $M JUDGE_GPUS=4,5 JUDGE_SPLITS="gsm8k math500" JUDGE_TASKS="vea_cal category" STEPS=judge &
  wait
  run $M $G STEPS="analyze iprep igen" || return 1
  run $M JUDGE_GPUS=0,1 JUDGE_TASKS=vea_cal STEPS=ijudge &
  run $M JUDGE_GPUS=2,3 JUDGE_TASKS=refusal STEPS=ijudge &
  wait
  run $M STEPS="analyze ianalyze"
}
qwen397 || log "397B block stopped early"
log "overnight run finished"
