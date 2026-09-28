#!/usr/bin/env bash
# Gemma 4 31B follow-up to the pilot (all steps resumable):
#   topup  every adversarial prompt with >=1 VEA in the pilot (results/gemma4_followup/vea_qids.txt,
#          q285 excluded: its "VEA" only restates the user's red-teaming premise) to N_TOPUP rollouts.
#          Appends to the pilot's rollouts file; rollout idx >= 20 are new, seeds follow rollout_seed.
#   cap    capability benchmarks (GSM8K first 500 test problems, MATH-500) x N_CAP rollouts
#   judge  JUDGE_TASK (default vea_cal) on every rollouts file above, plus refusal on the adversarial file
#
#   STEPS=topup GPUS=0,1 scripts/run_gemma_followup.sh
#   STEPS=cap GPUS=2,3 scripts/run_gemma_followup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-"topup cap judge"}
GPUS=${GPUS:-0,1}
TP=${TP:-2}
SEED=${SEED:-1}
N_TOPUP=${N_TOPUP:-200}
N_CAP=${N_CAP:-5}
JUDGE_TASK=${JUDGE_TASK:-vea_cal}
JUDGE_THINKING=${JUDGE_THINKING:-low}  # calibrated judge: F1 + notes (vea_cal) with low-effort thinking
MODEL=google/gemma-4-31B-it
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.8-27B}
R=results/gemma4_pilot/rollouts
J=results/gemma4_followup/judgments
ROLLOUTS=${ROLLOUTS:-$R}  # judge reads <ROLLOUTS>/gemma-4-31B-it.<split>.jsonl
export CUDA_VISIBLE_DEVICES=$GPUS
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}  # see run_gemma_pilot.sh
mkdir -p "$J" logs

gen() {  # gen <split> <n> [extra args]
  local split=$1 n=$2; shift 2
  uv run python -m eval_awareness.generate --backend vllm --model $MODEL --tp "$TP" --text-only \
    --generation-config-sampling --seed "$SEED" --split "$split" --n-rollouts "$n" \
    --max-tokens 16384 --max-model-len 40960 \
    --out "$R/gemma-4-31B-it.$split.jsonl" "$@" 2>&1 | tee -a "logs/followup_gen_$split.log"
}

judge() {  # judge <split> <task> [out-dir]
  uv run python -m eval_awareness.judge "$ROLLOUTS/gemma-4-31B-it.$1.jsonl" --task "$2" --split "$1" \
    --backend vllm --judge-model "$JUDGE_MODEL" --tp "$TP" --text-only --no-prefilter --judge-max-tokens 4096 \
    $([[ $2 == vea* ]] && echo --judge-thinking "$JUDGE_THINKING") \
    --max-model-len 40960 --out-dir "${3:-$J}" 2>&1 | tee -a logs/followup_judge.log
}

# questions per vLLM call: ~1,000 requests each, so the batch stays full (5 x 180 top-up, 200 x 5 capability)
if [[ " $STEPS " == *" topup "* ]]; then
  gen adversarial "$N_TOPUP" --questions-per-chunk 5 --qids $(cat results/gemma4_followup/vea_qids.txt)
fi
if [[ " $STEPS " == *" cap "* ]]; then
  gen gsm8k "$N_CAP" --questions-per-chunk 200
  gen math500 "$N_CAP" --questions-per-chunk 200
fi
if [[ " $STEPS " == *" judge "* ]]; then
  for split in ${JUDGE_SPLITS:-adversarial gsm8k math500}; do judge "$split" "$JUDGE_TASK"; done
  if [[ " ${JUDGE_SPLITS:-adversarial} " == *" adversarial "* ]]; then
    judge adversarial refusal results/gemma4_pilot/judgments  # extends the pilot's refusal judgments
  fi
fi
