#!/bin/bash
# Judge the harmonized rollouts (data/unified/rollouts) with this repo's judges, one model per GPU.
#   vea_cal   the calibrated VEA judge on the full reasoning (low thinking)
#   refusal   F2 on the 384-token answer view
#   category  post's categories on each VEA quote
# Every judge.py call skips (qid, sample_idx) pairs already judged, so reruns and reused judgments are free.
#   MODELS="gemma-4-31B-it Nemotron-3-Super" GPUS="0 1" scripts/run_unified.sh
#   TASKS=refusal SPLITS=adversarial scripts/run_unified.sh
set -u
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub} VLLM_USE_FLASHINFER_SAMPLER=0
MODELS=${MODELS:-$(ls data/unified/rollouts/*.adversarial.jsonl | xargs -n1 basename | sed 's/\.adversarial\.jsonl//')}
GPUS=(${GPUS:-0 1 2 3})
TASKS=${TASKS:-"vea_cal refusal category"}
SPLITS=${SPLITS:-"adversarial benign"}
D=data/unified
mkdir -p logs

judge_model() {  # judge_model <model> <gpu>
  local m=$1 g=$2
  for split in $SPLITS; do
    local R=$D/rollouts/$m.$split.jsonl
    [[ -f $R ]] || continue
    for task in $TASKS; do
      [[ $task == vea_cal && $split == benign ]] && continue  # benign VEA is not part of the protocol
      [[ $task == category && $split == benign ]] && continue
      case $task in
        vea_cal)  extra="--judge-thinking low --judge-max-tokens 4096" ;;
        refusal)  extra="--judge-max-tokens 1024" ;;
        category) extra="--judge-max-tokens 1024 --vea-judgments $D/judgments/$m.$split.vea_cal.Qwen3.8-27B.jsonl" ;;
      esac
      CUDA_VISIBLE_DEVICES=$g uv run python -m eval_awareness.judge "$R" --task "$task" --split "$split" \
        --backend vllm --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --no-prefilter --max-model-len 40960 \
        --max-num-seqs 256 --gpu-memory-utilization 0.8 --out-dir $D/judgments $extra \
        >> logs/unified_judge_$m.log 2>&1 || echo "FAILED $m $split $task (logs/unified_judge_$m.log)"
      echo "done $m $split $task"
    done
  done
}

i=0; declare -A queue
for m in $MODELS; do g=${GPUS[$((i % ${#GPUS[@]}))]}; queue[$g]+="$m "; i=$((i+1)); done
for g in "${!queue[@]}"; do
  ( for m in ${queue[$g]}; do judge_model "$m" "$g"; done ) &
done
wait
echo "all judging done"
