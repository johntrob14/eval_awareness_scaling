#!/bin/bash
# Ladder-design insertion / removal (docs/UNIFIED_PROTOCOL.md) on this repo's models, then the refusal judge.
#   MODELS="gemma-4-31B-it" STEPS="bases smoke" scripts/run_unified_interventions.sh   # 2 bases per arm first
#   MODELS="Nemotron-3-Super" scripts/run_unified_interventions.sh                    # bases gen judge
# Engine layouts are the ones the main runs used: Gemma TP=2, Nemotron / Qwen3.5-122B TP=4,
# Qwen3.5-397B TP=2 x PP=3 on all six GPUs. The judge runs after generation on JUDGE_GPU.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
MODELS=${MODELS:-"gemma-4-31B-it Nemotron-3-Super Qwen3.5-122B-A10B Qwen3.5-397B-A17B"}
STEPS=${STEPS:-"bases gen judge"}
JUDGE_GPU=${JUDGE_GPU:-0}
D=data/unified/interventions
mkdir -p logs

layout() {
  case $1 in
    gemma-4-31B-it)    echo "0,1 --tp 2" ;;
    Nemotron-3-Super)  echo "0,1,2,3 --tp 4" ;;
    Qwen3.5-122B-A10B) echo "0,1,2,3 --tp 4" ;;
    Qwen3.5-397B-A17B) echo "0,1,2,3,4,5 --tp 2 --pp 3" ;;
  esac
}

for m in $MODELS; do
  read -r gpus eng <<<"$(layout "$m")"
  L=logs/unified_intervene_$m.log
  if [[ " $STEPS " == *" bases "* ]]; then
    uv run python -m eval_awareness.intervene_unified bases --model "$m" 2>&1 | tee -a "$L"
  fi
  if [[ " $STEPS " == *" smoke "* ]]; then
    CUDA_VISIBLE_DEVICES=$gpus uv run python -m eval_awareness.intervene_unified generate --model "$m" $eng \
      --limit-bases 2 >> "$L" 2>&1 || { echo "smoke FAILED: $m ($L)"; continue; }
    echo "smoke done: $m -> $D/$m.smoke2.continuations.jsonl"
  fi
  if [[ " $STEPS " == *" gen "* ]]; then
    CUDA_VISIBLE_DEVICES=$gpus uv run python -m eval_awareness.intervene_unified generate --model "$m" $eng \
      >> "$L" 2>&1 || { echo "generation FAILED: $m ($L)"; continue; }
    echo "generated: $m"
  fi
  if [[ " $STEPS " == *" judge "* ]]; then
    CUDA_VISIBLE_DEVICES=$JUDGE_GPU uv run python -m eval_awareness.judge "$D/$m.continuations.jsonl" --task refusal \
      --backend vllm --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --max-model-len 40960 --max-num-seqs 256 \
      --gpu-memory-utilization 0.8 --judge-max-tokens 1024 --out-dir $D/judgments >> "$L" 2>&1 \
      || { echo "judging FAILED: $m ($L)"; continue; }
    echo "judged: $m"
  fi
done
echo "interventions done: $MODELS"
