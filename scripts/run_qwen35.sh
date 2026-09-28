#!/usr/bin/env bash
# Qwen3.5 (122B-A10B / 397B-A17B, official FP8) rerun of the Nemotron experiments. Output: $D.
# Sampling: the model card's thinking-mode setting for general tasks (T=1.0, top-p 0.95, top-k 20,
# presence penalty 1.5); generation_config.json's T=0.6 is the card's setting for precise coding.
# Steps (STEPS, all resumable):
#   smoke    20 pilot Stage 0 prompts x 5 + `followup smoke`
#   gen      490 adversarial prompts x 20
#   retry    regenerate rollouts truncated at 16,384 tokens with 32,768 (before judging)
#   cap      GSM8K (first 500) and MATH-500 x 5
#   judge    JUDGE_TASKS on JUDGE_SPLITS with JUDGE_GPUS (vea_cal, refusal, category)
#   analyze  `followup analyze`
#   iprep / igen / ijudge / ianalyze   CoT intervention (intervene.py), same design as Nemotron
#
#   MODEL=Qwen/Qwen3.5-122B-A10B-FP8 D=results/qwen35_122b STEPS="smoke" GPUS=0,1,2,3 TP=4 scripts/run_qwen35.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:?set MODEL}
D=${D:?set D}
STEPS=${STEPS:-smoke}
GPUS=${GPUS:-0,1,2,3}
TP=${TP:-4}
PP=${PP:-1}
JUDGE_GPUS=${JUDGE_GPUS:-4,5}
JUDGE_SPLITS=${JUDGE_SPLITS:-adversarial}
JUDGE_TASKS=${JUDGE_TASKS:-"vea_cal refusal category"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-}
SEED=${SEED:-1}
SAMPLING="--temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5"
SLUG=${MODEL##*/}
R=$D/rollouts
J=$D/judgments
JUDGE_MODEL=Qwen/Qwen3.8-27B
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}  # no nvcc on this box
# DeepGEMM (vLLM's default FP8 MoE kernels) JIT-compiles and needs a CUDA toolkit; fall back to the
# precompiled / Triton FP8 kernels
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0} VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}
mkdir -p "$R" "$J" logs
LOG=logs/$(basename "$D")
SEQS=${MAX_NUM_SEQS:+--max-num-seqs $MAX_NUM_SEQS}

EXCLUDE=$(uv run python -c "from eval_awareness.data import PILOT_EXCLUDE_QIDS as q; print(*q)")
gen() {  # gen <split> <n> [extra args]
  local split=$1 n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPUS uv run python -m eval_awareness.generate --backend vllm --model "$MODEL" \
    --tp "$TP" --pp "$PP" --text-only $SAMPLING $SEQS --seed "$SEED" --split "$split" --n-rollouts "$n" \
    --max-tokens "${MAX_TOKENS:-16384}" --max-model-len 40960 --out "$R/$SLUG.$split.jsonl" \
    --log-prompt "$D/rendered_prompt.$split.json" "$@" 2>&1 | tee -a "${LOG}_gen_$split.log"
}
judge() {  # judge <rollouts file> <task> <split> [extra args]
  CUDA_VISIBLE_DEVICES=$JUDGE_GPUS uv run python -m eval_awareness.judge "$1" --task "$2" --split "$3" \
    --backend vllm --judge-model $JUDGE_MODEL --tp 2 --text-only --no-prefilter --max-model-len 40960 \
    --out-dir "$(dirname "$1")/$( [[ $1 == */intervention/* ]] && echo judgments || echo ../judgments )" "${@:4}" \
    2>&1 | tee -a "${LOG}_judge.log"
}
task_args() {  # judge flags per task
  case $1 in
    vea_cal) echo "--judge-thinking low --judge-max-tokens 4096" ;;
    refusal) echo "--judge-max-tokens 1024" ;;
    category) echo "--judge-max-tokens 1024" ;;
  esac
}

if [[ " $STEPS " == *" smoke "* ]]; then
  gen adversarial 5 --exclude-qids $EXCLUDE --qids $(cat results/gemma4_pilot/stage0_qids.txt)
  uv run python -m eval_awareness.followup smoke --slug "$SLUG" --dir "$D" --rollouts "$R"
fi
if [[ " $STEPS " == *" gen "* ]]; then
  gen adversarial 20 --exclude-qids $EXCLUDE --questions-per-chunk 50
fi
if [[ " $STEPS " == *" retry "* ]]; then
  MAX_TOKENS=32768 gen adversarial 20 --exclude-qids $EXCLUDE --questions-per-chunk 50 --retry-truncated
fi
if [[ " $STEPS " == *" cap "* ]]; then
  gen gsm8k 5 --questions-per-chunk 200
  gen math500 5 --questions-per-chunk 200
fi
if [[ " $STEPS " == *" judge "* ]]; then
  for split in $JUDGE_SPLITS; do
    for task in $JUDGE_TASKS; do
      [[ $task == refusal && $split != adversarial ]] && continue
      extra=$(task_args $task)
      [[ $task == category ]] && extra="$extra --vea-judgments $J/$SLUG.$split.vea_cal.Qwen3.8-27B.jsonl"
      judge "$R/$SLUG.$split.jsonl" "$task" "$split" $extra
    done
  done
fi
if [[ " $STEPS " == *" analyze "* ]]; then
  # non-fatal: a mid-run analyze can meet judgment files still being written; a final analyze runs at the end
  uv run python -m eval_awareness.followup analyze --task vea_cal --slug "$SLUG" --dir "$D" --rollouts "$R" || true
fi
IV="--model $MODEL --dir $D"
if [[ " $STEPS " == *" iprep "* ]]; then
  [[ -f $D/intervention/items.jsonl ]] || uv run python -m eval_awareness.intervene prepare $IV
fi
if [[ " $STEPS " == *" igen "* ]]; then
  CUDA_VISIBLE_DEVICES=$GPUS uv run python -m eval_awareness.intervene generate $IV --tp "$TP" --pp "$PP" \
    --text-only $SAMPLING $SEQS 2>&1 | tee -a "${LOG}_intervene_gen.log"
fi
if [[ " $STEPS " == *" ijudge "* ]]; then
  C=$D/intervention/$SLUG.continuations.jsonl
  for task in ${JUDGE_TASKS/category/}; do judge "$C" "$task" adversarial $(task_args $task); done
fi
if [[ " $STEPS " == *" ianalyze "* ]]; then
  uv run python -m eval_awareness.intervene analyze $IV
fi
