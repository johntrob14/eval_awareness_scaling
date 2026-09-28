#!/usr/bin/env bash
# Nemotron 3 Super (120B-A12B, bf16) rerun of the Gemma 4 pilot + follow-up. Output: results/nemotron3_super/.
# Steps (STEPS, all resumable; generation and judging use different GPUs so they can overlap):
#   smoke   20 pilot Stage 0 prompts x 5, then `followup smoke` (thought blocks, truncation, tok/s)
#   gen     every adversarial prompt (490, PILOT_EXCLUDE_QIDS dropped) x 20
#   retry   regenerate adversarial rollouts truncated at MAX_TOKENS with 32,768 (run before judging:
#           the judge skips rollouts it has already judged)
#   judge   calibrated VEA judge (vea_cal, low thinking) on JUDGE_SPLITS (default adversarial) + refusal
#           judge (the pilot's: F2, no thinking) on adversarial
#   topup   prompts in vea_qids.txt (written after reading the VEA quotes) to N_TOPUP rollouts
#   cap     GSM8K (first 500) and MATH-500 x N_CAP
#   analyze `followup analyze`
#
#   STEPS=smoke GPUS=0,1,2,3 scripts/run_nemotron.sh
#   STEPS=judge JUDGE_GPUS=4,5 scripts/run_nemotron.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-"smoke"}
GPUS=${GPUS:-0,1,2,3}
TP=${TP:-4}
PP=${PP:-1}
JUDGE_GPUS=${JUDGE_GPUS:-4,5}
JUDGE_TP=${JUDGE_TP:-2}
SEED=${SEED:-1}
N=${N:-20}
N_TOPUP=${N_TOPUP:-200}
N_CAP=${N_CAP:-5}
MAX_TOKENS=${MAX_TOKENS:-16384}
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
SLUG=${MODEL##*/}
JUDGE_MODEL=Qwen/Qwen3.8-27B
D=results/nemotron3_super
R=$D/rollouts
J=$D/judgments
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}  # no nvcc on this box; see run_gemma_pilot.sh
mkdir -p "$R" "$J" logs

EXCLUDE=$(uv run python -c "from eval_awareness.data import PILOT_EXCLUDE_QIDS as q; print(*q)")
gen() {  # gen <split> <n> [extra args]; sampling from generation_config.json (T=1.0, top_p=0.95, no top-k)
  local split=$1 n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPUS uv run python -m eval_awareness.generate --backend vllm --model $MODEL \
    --tp "$TP" --pp "$PP" --generation-config-sampling --seed "$SEED" --split "$split" --n-rollouts "$n" \
    --max-tokens "$MAX_TOKENS" --max-model-len 40960 --out "$R/$SLUG.$split.jsonl" \
    --log-prompt "$D/rendered_prompt.$split.json" "$@" 2>&1 | tee -a "logs/nemotron_gen_$split.log"
}
judge() {  # judge <split> <task> [extra args]
  CUDA_VISIBLE_DEVICES=$JUDGE_GPUS uv run python -m eval_awareness.judge "$R/$SLUG.$1.jsonl" --task "$2" \
    --split "$1" --backend vllm --judge-model $JUDGE_MODEL --tp "$JUDGE_TP" --text-only --no-prefilter \
    --max-model-len 40960 --out-dir "$J" "${@:3}" 2>&1 | tee -a logs/nemotron_judge.log
}

if [[ " $STEPS " == *" smoke "* ]]; then
  gen adversarial 5 --exclude-qids $EXCLUDE --qids $(cat results/gemma4_pilot/stage0_qids.txt)
  uv run python -m eval_awareness.followup smoke --slug "$SLUG" --dir $D --rollouts $R
fi
if [[ " $STEPS " == *" gen "* ]]; then
  gen adversarial "$N" --exclude-qids $EXCLUDE --questions-per-chunk 50
fi
if [[ " $STEPS " == *" retry "* ]]; then  # the pilot's rule: >1% truncated -> regenerate those at 32k tokens
  MAX_TOKENS=32768 gen adversarial "$N" --exclude-qids $EXCLUDE --questions-per-chunk 50 --retry-truncated
fi
if [[ " $STEPS " == *" judge "* ]]; then
  # only splits whose generation has finished: a file still being appended can end in a partial line
  for split in ${JUDGE_SPLITS:-adversarial}; do
    judge "$split" vea_cal --judge-thinking low --judge-max-tokens 4096
  done
  if [[ " ${JUDGE_SPLITS:-adversarial} " == *" adversarial "* ]]; then
    judge adversarial refusal --judge-max-tokens 1024
  fi
fi
if [[ " $STEPS " == *" topup "* ]]; then
  gen adversarial "$N_TOPUP" --exclude-qids $EXCLUDE --questions-per-chunk 5 --qids $(cat $D/vea_qids.txt)
fi
if [[ " $STEPS " == *" cap "* ]]; then
  gen gsm8k "$N_CAP" --questions-per-chunk 200
  gen math500 "$N_CAP" --questions-per-chunk 200
fi
if [[ " $STEPS " == *" analyze "* ]]; then
  uv run python -m eval_awareness.followup analyze --task vea_cal --slug "$SLUG" --dir $D --rollouts $R
fi
