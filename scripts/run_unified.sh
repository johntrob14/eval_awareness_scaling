#!/bin/bash
# Judge the harmonized rollouts (data/unified/<VERSION>/rollouts) with this repo's judges, one model per GPU.
#   vea_cal   the calibrated VEA judge on the full reasoning (low thinking, 4,096 tokens), then a rescue pass at
#             16,384 tokens for outputs whose thinking never closed (parse_ok False), then, for the few where the
#             thinking judge loops without a verdict, the same rubric with thinking off (judge_thinking is stored
#             on every row, so these are identifiable)
#   refusal   F2 on the 384-token answer view
#   category  post's categories on each VEA quote
# Every call passes --redo-invalid: keys already judged with a valid verdict are skipped, invalid ones are redone.
# Exit status is non-zero if any judge call failed; "all judging done" is printed only when all succeeded.
#   MODELS="gemma-4-31B-it Nemotron-3-Super" GPUS="0 1" scripts/run_unified.sh
#   TASKS=refusal SPLITS=adversarial scripts/run_unified.sh
set -u
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub} VLLM_USE_FLASHINFER_SAMPLER=0
VERSION=${VERSION:-v2}
D=data/unified/$VERSION
MODELS=${MODELS:-$(ls $D/rollouts/*.adversarial.jsonl | xargs -n1 basename | sed 's/\.adversarial\.jsonl//')}
GPUS=(${GPUS:-0 1 2 3})
TASKS=${TASKS:-"vea_cal refusal category"}
SPLITS=${SPLITS:-"adversarial benign"}
GPU_MEM=${GPU_MEM:-0.8}
FAILS=$(mktemp)
mkdir -p logs

judge() {  # judge <rollouts> <task> <split> <gpu> <log> [extra args]
  CUDA_VISIBLE_DEVICES=$4 uv run python -m eval_awareness.judge "$1" --task "$2" --split "$3" \
    --backend vllm --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --no-prefilter --max-model-len 40960 \
    --max-num-seqs 256 --gpu-memory-utilization "$GPU_MEM" --out-dir $D/judgments --redo-invalid "${@:6}" >> "$5" 2>&1
}

judge_model() {  # judge_model <model> <gpu>
  local m=$1 g=$2 L=logs/unified_${VERSION}_judge_$1.log
  for split in $SPLITS; do
    local R=$D/rollouts/$m.$split.jsonl
    [[ -f $R ]] || continue
    for task in $TASKS; do
      [[ $split == benign && $task != refusal ]] && continue  # benign: refusal only
      case $task in
        vea_cal)  judge "$R" vea_cal "$split" "$g" "$L" --judge-thinking low --judge-max-tokens 4096 &&
                  judge "$R" vea_cal "$split" "$g" "$L" --judge-thinking low --judge-max-tokens 16384 &&
                  judge "$R" vea_cal "$split" "$g" "$L" --judge-thinking off --judge-max-tokens 1024 ;;
        refusal)  judge "$R" refusal "$split" "$g" "$L" --judge-max-tokens 1024 ;;
        category) judge "$R" category "$split" "$g" "$L" --judge-max-tokens 1024 \
                    --vea-judgments $D/judgments/$m.$split.vea_cal.Qwen3.8-27B.jsonl ;;
      esac
      if [[ $? -ne 0 ]]; then echo "FAILED $m $split $task ($L)" | tee -a "$FAILS"; return; fi
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
if [[ -s $FAILS ]]; then echo "judging FAILED for:"; cat "$FAILS"; rm -f "$FAILS"; exit 1; fi
rm -f "$FAILS"
echo "all judging done"
