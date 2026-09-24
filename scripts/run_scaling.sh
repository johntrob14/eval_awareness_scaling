#!/usr/bin/env bash
# Full Qwen3 scaling pipeline on NVIDIA GPUs (vLLM): generate -> judge (vea, refusal, category) -> analyze.
# Every stage is resumable, so rerunning after an interruption continues where it stopped.
#
#   scripts/run_scaling.sh                                   # all sizes, sequentially, 1 GPU each
#   MODELS="Qwen3-0.6B Qwen3-1.7B" CUDA_VISIBLE_DEVICES=0 scripts/run_scaling.sh &
#   MODELS="Qwen3-32B" TP=2 CUDA_VISIBLE_DEVICES=0,1 scripts/run_scaling.sh
#
# Env: MODELS, N (rollouts/question, post uses 100), TP (tensor parallel for generation),
#      JUDGE_BACKEND (vllm|openai), JUDGE_MODEL, JUDGE_TP
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B"}
N=${N:-100}
TP=${TP:-1}
JUDGE_BACKEND=${JUDGE_BACKEND:-vllm}
JUDGE_TP=${JUDGE_TP:-1}
if [[ $JUDGE_BACKEND == openai ]]; then JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-mini}; else JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-8B}; fi
JUDGE_SLUG=${JUDGE_MODEL##*/}
mkdir -p logs

judge() {  # judge <model-slug> <task> [extra args]
  local m=$1 task=$2; shift 2
  uv run python -m eval_awareness.judge "data/rollouts/$m.jsonl" --task "$task" \
    --backend "$JUDGE_BACKEND" --judge-model "$JUDGE_MODEL" --tp "$JUDGE_TP" "$@" 2>&1 | tee -a "logs/judge_$m.log"
}

for m in $MODELS; do
  echo "=== $m: generate ($N rollouts/question)"
  uv run python -m eval_awareness.generate --backend vllm --model "Qwen/$m" --n-rollouts "$N" --tp "$TP" \
    2>&1 | tee -a "logs/gen_$m.log"
  echo "=== $m: judge with $JUDGE_MODEL ($JUDGE_BACKEND)"
  judge "$m" vea --no-prefilter  # judge every CoT; the regex prefilter is only for slow local judges
  judge "$m" refusal
  judge "$m" category
done

uv run python -m eval_awareness.analyze --judge "$JUDGE_SLUG" \
  --models $(for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B; do [[ -f data/rollouts/$m.jsonl ]] && echo "$m"; done)
