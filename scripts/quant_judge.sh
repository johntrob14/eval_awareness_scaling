#!/bin/bash
# Judge quantization-study rollouts (QUANT_DATA/rollouts) with the unified v2 judges, one variant at a time.
# Same passes and settings as scripts/run_unified.sh: vea_cal (low thinking 4,096 tokens, rescue at 16,384, then
# thinking off), refusal (F2 on the 384-token answer view), category (on each VEA quote). Benign: refusal only.
# The judge is Qwen3.8-27B bf16 pinned to the revision the v2 judgments used.
#   VARIANTS="Qwen3.5-4B@bf16 Qwen3.5-4B@gptq-w4" TP=2 GPUS=0,1 scripts/quant_judge.sh
#   VARIANTS="Qwen3.5-4B@bf16.smoke" SPLITS=adversarial scripts/quant_judge.sh
set -u
cd "$(dirname "$0")/.."
: "${QUANT_DATA:?set QUANT_DATA (source .sauce/env.sh)}"
export VLLM_USE_FLASHINFER_SAMPLER=0
D=$QUANT_DATA
TP=${TP:-1}
GPUS=${GPUS:-0}
TASKS=${TASKS:-"vea_cal refusal category"}
SPLITS=${SPLITS:-"adversarial benign"}
GPU_MEM=${GPU_MEM:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
JUDGE_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
mkdir -p "$D/logs"
status=0

judge() {  # judge <rollouts> <task> <split> <log> [extra args]
  CUDA_VISIBLE_DEVICES=$GPUS uv run --no-sync python -m eval_awareness.judge "$1" --task "$2" --split "$3" \
    --backend vllm --judge-model Qwen/Qwen3.8-27B --judge-revision $JUDGE_REVISION --tp "$TP" --text-only \
    --no-prefilter --max-model-len 40960 --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEM" \
    --out-dir "$D/judgments" --redo-invalid "${@:5}" >> "$4" 2>&1
}

for v in $VARIANTS; do
  L=$D/logs/judge_$v.log
  for split in $SPLITS; do
    R=$D/rollouts/$v.$split.jsonl
    [[ -f $R ]] || continue
    for task in $TASKS; do
      [[ $split == benign && $task != refusal ]] && continue
      case $task in
        vea_cal)  judge "$R" vea_cal "$split" "$L" --judge-thinking low --judge-max-tokens 4096 &&
                  judge "$R" vea_cal "$split" "$L" --judge-thinking low --judge-max-tokens 16384 &&
                  judge "$R" vea_cal "$split" "$L" --judge-thinking off --judge-max-tokens 1024 ;;
        refusal)  judge "$R" refusal "$split" "$L" --judge-max-tokens 1024 ;;
        category) judge "$R" category "$split" "$L" --judge-max-tokens 1024 \
                    --vea-judgments "$D/judgments/$v.$split.vea_cal.Qwen3.8-27B.jsonl" ;;
      esac
      if [[ $? -ne 0 ]]; then echo "FAILED $v $split $task (see $L)"; status=1; break 2; fi
      echo "done $v $split $task"
    done
  done
done
[[ $status -eq 0 ]] && echo "all judging done"
exit $status
