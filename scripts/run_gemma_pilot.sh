#!/usr/bin/env bash
# Gemma 4 31B VEA pilot on Fortress. Stages (STAGES="0 1 2 3" by default), all resumable:
#   0  smoke test: 20 adversarial prompts x 5 rollouts, then `pilot smoke` (thought blocks, truncation, tok/s)
#   1  generation: every adversarial prompt to 20 rollouts (tops up the Stage 0 prompts), benign x 5,
#      then regenerate truncated rollouts at MAX_TOKENS_RETRY if more than 1% were truncated
#   2  judging with Qwen3.8-27B (Gemma unloaded first): VEA on CoT, VEA on response, refusal,
#      category and safety tag on every VEA quote
#   3  analysis: results/gemma4_pilot/analysis.json, labeling.csv, plots
#
#   STAGES=0 scripts/run_gemma_pilot.sh
#   GPUS=0,1,2,3 STAGES="1 2 3" scripts/run_gemma_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES=${STAGES:-"0 1 2 3"}
GPUS=${GPUS:-0,1,2,3}
TP=${TP:-4}
PP=${PP:-1}  # the A6000s have no NVLink (topology SYS); PP=4 TP=1 avoids per-layer all-reduces
SEED=${SEED:-1}
MAX_TOKENS=${MAX_TOKENS:-16384}
MAX_TOKENS_RETRY=${MAX_TOKENS_RETRY:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MODEL=google/gemma-4-31B-it
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.8-27B}
JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-40960}
D=results/gemma4_pilot
R=$D/rollouts
mkdir -p "$R" "$D/judgments" logs
export CUDA_VISIBLE_DEVICES=$GPUS
# FlashInfer's top-k/top-p sampler JIT-compiles on first use and needs nvcc, which the H100 box
# lacks. vLLM's PyTorch sampler is statistically equivalent (not token-identical for a given seed).
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

EXCLUDE=$(uv run python -c "from eval_awareness.data import PILOT_EXCLUDE_QIDS as q; print(*q)")
gen() {  # gen <split> <n> [extra args]
  local split=$1 n=$2; shift 2
  uv run python -m eval_awareness.generate --backend vllm --model $MODEL --tp "$TP" --pp "$PP" --text-only \
    --generation-config-sampling --seed "$SEED" --split "$split" --n-rollouts "$n" \
    --exclude-qids $EXCLUDE --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --out "$R/gemma-4-31B-it.$split.jsonl" --log-prompt "$D/rendered_prompt.$split.json" "$@" \
    2>&1 | tee -a logs/gemma_gen.log
}

if [[ " $STAGES " == *" 0 "* ]]; then
  # 20 smoke-test prompts, drawn once with a fixed seed and saved
  [[ -f $D/stage0_qids.txt ]] || uv run python -c "
import numpy as np
from eval_awareness.data import PILOT_EXCLUDE_QIDS, load_fortress
q = load_fortress(exclude_explicit_eval=False).qid
q = q[~q.isin(PILOT_EXCLUDE_QIDS)].to_numpy()
print(*sorted(np.random.default_rng(0).choice(q, 20, replace=False)))" > "$D/stage0_qids.txt"
  gen adversarial 5 --qids $(cat "$D/stage0_qids.txt")
  uv run python -m eval_awareness.pilot smoke --dir $D
fi

if [[ " $STAGES " == *" 1 "* ]]; then
  gen adversarial 20
  gen benign 5
  for split in adversarial benign; do
    frac=$(uv run python -c "
import pandas as pd; from eval_awareness.io import read_jsonl
d = pd.DataFrame(read_jsonl('$R/gemma-4-31B-it.$split.jsonl')).drop_duplicates(['qid','sample_idx'], keep='last')
print((d.finish_reason == 'length').mean())")
    echo "$split truncated fraction: $frac"
    if uv run python -c "import sys; sys.exit(0 if $frac > 0.01 else 1)"; then
      MAX_TOKENS=$MAX_TOKENS_RETRY gen "$split" "$([[ $split == adversarial ]] && echo 20 || echo 5)" --retry-truncated
    fi
  done
fi

judge() {  # judge <split> <task> [extra args]
  local split=$1 task=$2; shift 2
  uv run python -m eval_awareness.judge "$R/gemma-4-31B-it.$split.jsonl" --task "$task" --split "$split" \
    --backend vllm --judge-model "$JUDGE_MODEL" --tp "$TP" --pp "$PP" --text-only --no-prefilter --judge-max-tokens 1024 \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" --out-dir "$D/judgments" "$@" 2>&1 | tee -a logs/gemma_judge.log
}

if [[ " $STAGES " == *" 2 "* ]]; then
  J=${JUDGE_MODEL##*/}
  for split in adversarial benign; do
    judge "$split" vea
    judge "$split" vea_response --max-vea-chars 40000
    judge "$split" refusal
    V="$D/judgments/gemma-4-31B-it.$split.vea.$J.jsonl $D/judgments/gemma-4-31B-it.$split.vea_response.$J.jsonl"
    judge "$split" category --vea-judgments $V
    judge "$split" safety --vea-judgments $V
  done
fi

if [[ " $STAGES " == *" 3 "* ]]; then
  uv run python -m eval_awareness.pilot analyze --dir $D --judge "${JUDGE_MODEL##*/}"
fi
