#!/bin/bash
# The ladder's 12 models under the unified protocol (docs/UNIFIED_PROTOCOL.md). Needs the collaborator's cache
# copied to ladder/cache/ (exp02/generations, exp03/generations, exp03/interventions incl. *_bases.json).
#   build      harmonize rollouts -> data/unified/rollouts/<model>.{adversarial,benign}.jsonl
#   judge      vea_cal + refusal + category (scripts/run_unified.sh, one model per GPU)
#   iconvert   exp03 continuations -> data/unified/interventions/<model>.continuations.jsonl (removal bases
#              kept only where vea_cal agrees); needs the judge step first
#   ijudge     refusal on the continuations
#   analyze    unified + intervention analysis -> results/unified/
#   STEPS="build judge" GPUS="0 1 2 3 4 5" scripts/run_unified_ladder.sh
set -u
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub} HF_HOME=${HF_HOME:-/home/jtr2875/.cache/huggingface}
export VLLM_USE_FLASHINFER_SAMPLER=0
STEPS=${STEPS:-"build judge iconvert ijudge analyze"}
GPUS=${GPUS:-"0 1 2 3 4 5"}
[[ -d ladder/cache/exp02/generations ]] || { echo "ladder/cache is missing: copy the collaborator's cache/ there first"; exit 1; }
MODELS=${MODELS:-$(uv run python -c "from eval_awareness.unified import ladder_models; print(*ladder_models())")}
D=data/unified/interventions
mkdir -p logs

[[ " $STEPS " == *" build "* ]] && uv run python -m eval_awareness.unified build --model $MODELS
[[ " $STEPS " == *" judge "* ]] && MODELS="$MODELS" GPUS="$GPUS" scripts/run_unified.sh
if [[ " $STEPS " == *" iconvert "* ]]; then
  for m in $MODELS; do uv run python -m eval_awareness.intervene_unified convert-ladder --model "$m"; done
fi
if [[ " $STEPS " == *" ijudge "* ]]; then
  g=(${GPUS}); i=0
  for m in $MODELS; do
    ( CUDA_VISIBLE_DEVICES=${g[$((i % ${#g[@]}))]} uv run python -m eval_awareness.judge "$D/$m.continuations.jsonl" \
        --task refusal --backend vllm --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --max-model-len 40960 \
        --max-num-seqs 256 --gpu-memory-utilization 0.8 --judge-max-tokens 1024 --out-dir $D/judgments \
        >> logs/unified_ijudge_$m.log 2>&1 || echo "FAILED ijudge $m" ) &
    i=$((i+1)); (( i % ${#g[@]} == 0 )) && wait
  done
  wait
fi
if [[ " $STEPS " == *" analyze "* ]]; then
  uv run python -m eval_awareness.unified analyze
  uv run python -m eval_awareness.intervene_unified analyze
fi
