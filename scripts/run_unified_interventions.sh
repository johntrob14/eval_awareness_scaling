#!/bin/bash
# Ladder-design insertion / removal (docs/UNIFIED_PROTOCOL.md) on this repo's models, then the judges:
# refusal on the first 384 answer tokens, and vea_cal on the new reasoning (re-verbalization diagnostic).
#   MODELS="gemma-4-31B-it" STEPS="bases smoke" scripts/run_unified_interventions.sh   # 2 bases per arm first
#   MODELS="Nemotron-3-Super" scripts/run_unified_interventions.sh                    # bases gen judge
# Engine layouts are the ones the main runs used: Gemma TP=2, Nemotron / Qwen3.5-122B TP=4,
# Qwen3.5-397B TP=2 x PP=3 on all six GPUs. Judging runs after generation on JUDGE_GPU.
# Exits non-zero if any step failed for any model (a failed model is skipped, the others still run).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
VERSION=${VERSION:-v2}
MODELS=${MODELS:-"gemma-4-31B-it Nemotron-3-Super Qwen3.5-122B-A10B Qwen3.5-397B-A17B"}
STEPS=${STEPS:-"bases gen judge"}
JUDGE_GPU=${JUDGE_GPU:-0}
# The GPUs are shared (another user holds 11-15 GB per card), so leave headroom.
GPU_MEM=${GPU_MEM:-0.80}
D=data/unified/$VERSION/interventions
FAILED=()
mkdir -p logs

layout() {
  case $1 in
    gemma-4-31B-it)    echo "0,1 --tp 2" ;;
    Nemotron-3-Super)  echo "0,1,2,3 --tp 4" ;;
    Qwen3.5-122B-A10B) echo "0,1,2,3 --tp 4" ;;
    Qwen3.5-397B-A17B) echo "0,1,2,3,4,5 --tp 2 --pp 3 --max-num-seqs 256" ;;  # Mamba cache blocks: 444 at 0.80
  esac
}

judge() {  # judge <continuations file> <task> <log> [extra]
  CUDA_VISIBLE_DEVICES=$JUDGE_GPU uv run python -m eval_awareness.judge "$1" --task "$2" --backend vllm \
    --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --no-prefilter --max-model-len 40960 --max-num-seqs 256 \
    --gpu-memory-utilization "$GPU_MEM" --out-dir $D/judgments --redo-invalid "${@:4}" >> "$3" 2>&1
}

for m in $MODELS; do
  read -r gpus eng <<<"$(layout "$m")"
  L=logs/unified_${VERSION}_intervene_$m.log
  if [[ " $STEPS " == *" bases "* ]]; then
    uv run python -m eval_awareness.intervene_unified bases --model "$m" 2>&1 | tee -a "$L"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "bases FAILED: $m ($L)"; FAILED+=("$m:bases"); continue; }
  fi
  if [[ " $STEPS " == *" smoke "* ]]; then
    CUDA_VISIBLE_DEVICES=$gpus uv run python -m eval_awareness.intervene_unified generate --model "$m" $eng \
      --limit-bases 2 --gpu-memory-utilization "$GPU_MEM" >> "$L" 2>&1 \
      || { echo "smoke FAILED: $m ($L)"; FAILED+=("$m:smoke"); continue; }
    echo "smoke done: $m -> $D/$m.smoke2.continuations.jsonl"
  fi
  if [[ " $STEPS " == *" gen "* ]]; then
    ok=0
    for attempt in 1 2 3; do  # generation resumes from the continuations already written
      CUDA_VISIBLE_DEVICES=$gpus uv run python -m eval_awareness.intervene_unified generate --model "$m" $eng \
        --gpu-memory-utilization "$GPU_MEM" >> "$L" 2>&1 && { ok=1; break; }
      echo "generation attempt $attempt FAILED: $m ($L)"
    done
    (( ok )) || { echo "generation FAILED: $m ($L)"; FAILED+=("$m:gen"); continue; }
    echo "generated: $m"
  fi
  if [[ " $STEPS " == *" judge "* ]]; then
    C=$D/$m.continuations.jsonl
    judge "$C" refusal "$L" --judge-max-tokens 1024 \
      && judge "$C" vea_cal "$L" --judge-thinking low --judge-max-tokens 4096 \
      && judge "$C" vea_cal "$L" --judge-thinking low --judge-max-tokens 16384 \
      || { echo "judging FAILED: $m ($L)"; FAILED+=("$m:judge"); continue; }
    echo "judged: $m"
  fi
done
if (( ${#FAILED[@]} )); then echo "interventions FAILED: ${FAILED[*]}"; exit 1; fi
echo "interventions done: $MODELS"
