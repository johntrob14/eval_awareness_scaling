#!/bin/bash
# The ladder's 12 models under the unified protocol (docs/UNIFIED_PROTOCOL.md). Needs the collaborator's cache
# copied to ladder/cache/: exp02/{generations/, prompts.jsonl, prompt_prep.json} and exp03/{generations/,
# interventions/ incl. *_bases.json}.
#   build      harmonize rollouts -> data/unified/<VERSION>/rollouts/<model>.{adversarial,benign}.jsonl
#   judge      vea_cal + refusal + category (scripts/run_unified.sh, one model per GPU)
#   iconvert   exp03 continuations -> data/unified/<VERSION>/interventions/<model>.continuations.jsonl (removal
#              bases kept only where vea_cal agrees and the whole quote is in the removed span); needs judge first
#   ijudge     refusal + vea_cal (re-verbalization) on the continuations
#   analyze    unified + intervention analysis -> results/unified/<VERSION>/ (FINAL=1 adds --final)
# Each step stops the script if it fails; later steps never run on incomplete inputs.
#   STEPS="build judge" GPUS="0 1 2 3 4 5" scripts/run_unified_ladder.sh
set -eu -o pipefail
cd "$(dirname "$0")/.."
# The ladder's code finds pinned tokenizers under $HF_HOME/hub; on this box the login shell's HF_HOME points at the
# full /ssd1 cache, so it is set explicitly (LADDER_HF_HOME overrides).
export HF_HOME=${LADDER_HF_HOME:-/home/jtr2875/.cache/huggingface}
export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_USE_FLASHINFER_SAMPLER=0
VERSION=${VERSION:-v2}
STEPS=${STEPS:-"build judge iconvert ijudge analyze"}
GPUS=${GPUS:-"0 1 2 3 4 5"}
GPU_MEM=${GPU_MEM:-0.8}
for f in exp02/generations exp02/prompts.jsonl exp02/prompt_prep.json exp03/generations exp03/interventions; do
  [[ -e ladder/cache/$f ]] || { echo "ladder/cache/$f is missing: copy the collaborator's cache first"; exit 1; }
done
MODELS=${MODELS:-$(uv run python -c "from eval_awareness.unified import ladder_models; print(*ladder_models())")}
D=data/unified/$VERSION/interventions
mkdir -p logs

if [[ " $STEPS " == *" build "* ]]; then uv run python -m eval_awareness.unified build --model $MODELS; fi
if [[ " $STEPS " == *" judge "* ]]; then MODELS="$MODELS" GPUS="$GPUS" VERSION=$VERSION scripts/run_unified.sh; fi
if [[ " $STEPS " == *" iconvert "* ]]; then
  for m in $MODELS; do uv run python -m eval_awareness.intervene_unified convert-ladder --model "$m"; done
fi
if [[ " $STEPS " == *" ijudge "* ]]; then
  g=($GPUS); i=0; pids=()
  for m in $MODELS; do
    gpu=${g[$((i % ${#g[@]}))]}
    ( set -e; C=$D/$m.continuations.jsonl; L=logs/unified_${VERSION}_ijudge_$m.log
      j() { CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval_awareness.judge "$C" --task "$1" --backend vllm \
              --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --no-prefilter --max-model-len 40960 --max-num-seqs 256 \
              --gpu-memory-utilization "$GPU_MEM" --out-dir $D/judgments --redo-invalid "${@:2}" >> "$L" 2>&1; }
      j refusal --judge-max-tokens 1024
      j vea_cal --judge-thinking low --judge-max-tokens 4096
      j vea_cal --judge-thinking low --judge-max-tokens 16384 ) &
    pids+=($!); i=$((i+1))
    if (( i % ${#g[@]} == 0 )); then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done  # set -e: a failed judge job stops the script here
fi
if [[ " $STEPS " == *" analyze "* ]]; then
  uv run python -m eval_awareness.unified analyze ${FINAL:+--final}
  uv run python -m eval_awareness.intervene_unified analyze ${FINAL:+--final}
fi
echo "ladder steps done: $STEPS"
