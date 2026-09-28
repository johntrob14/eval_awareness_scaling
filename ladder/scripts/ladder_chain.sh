#!/usr/bin/env bash
# exp02 run order after the Qwen3 ladder generation, chained so the GPU does not wait between steps.
# Resumable: models already generated or labeled are skipped (generation and labels are cached).
#
#   setsid nohup scripts/ladder_chain.sh [pid to wait for first] > log/exp02/chain.log 2>&1 &
#
# 1. optionally wait for a running generation to finish
# 2. Qwen3-32B download in the background if its weights are missing
# 3. judge Qwen3; preliminary Qwen3 analysis and labeling sheet 1 (CPU, failure does not stop the chain)
# 4. generate the Qwen3.5 ladder, then Qwen3.8-27B (its bf16 weights stay: they are the judge;
#    QWEN38_STAGE1_ONLY=1 applies the second pre-agreed fallback step)
# 5. judge Qwen3.5 + Qwen3.8; validation subset re-judged in thinking mode, then by the FP8 checkpoint
# The Qwen3-32B add-on, retries, the final report and the weight clean-up are run by scripts/ladder_tail.sh.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
LOG_DIR=log/exp02
MAIN_PY=/venv/main/bin/python
VLLM_PY=scripts/vllm_python.sh
QWEN3=(Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B)
QWEN35=(Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B)
QWEN38=Qwen3.8-27B
export PYTHONDONTWRITEBYTECODE=1
# ladder_run.sh passes this on to ladder_generate.py: the stage-2 size in force (ladder_config.STAGE2_EXTRA_USED)
export STAGE2_EXTRA="$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.STAGE2_EXTRA_USED)")"

stage() {
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] start ${name}"
  "$@" > "${LOG_DIR}/${name}.log" 2>&1 || { echo "FAILED ${name}; see ${LOG_DIR}/${name}.log" >&2; exit 1; }
  echo "[$(date -u +%H:%M:%S)] done  ${name}"
}

soft_stage() {  # CPU-only steps whose failure must not idle the GPU
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] start ${name}"
  if "$@" > "${LOG_DIR}/${name}.log" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] done  ${name}"
  else
    echo "[$(date -u +%H:%M:%S)] FAILED (non-fatal) ${name}; see ${LOG_DIR}/${name}.log"
  fi
}

status() { "$MAIN_PY" scripts/ladder_status.py "$@"; }

not_generated() {  # prints the given models that still need generation
  local m
  for m in "$@"; do status generated "$m" || echo "$m"; done
}

if [[ $# -ge 1 ]]; then
  echo "[$(date -u +%H:%M:%S)] waiting for pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
missing="$(not_generated "${QWEN3[@]}")"
[[ -z "$missing" ]] || { echo "Qwen3 generation incomplete: ${missing}" >&2; exit 1; }

if ! status weights Qwen3-32B; then
  qwen3_32b_rev="$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.ADDON_SUBJECTS['Qwen3-32B']['revision'])")"
  echo "[$(date -u +%H:%M:%S)] background download Qwen/Qwen3-32B@${qwen3_32b_rev}"
  ( /venv/main/bin/hf download Qwen/Qwen3-32B --revision "$qwen3_32b_rev" > "${LOG_DIR}/download_qwen3_32b.log" 2>&1 \
    && echo "[$(date -u +%H:%M:%S)] download Qwen3-32B done" || echo "[$(date -u +%H:%M:%S)] download Qwen3-32B FAILED" ) &
fi

stage judge_qwen3 "$VLLM_PY" scripts/ladder_judge.py label --models "${QWEN3[@]}"
soft_stage analysis_preliminary_qwen3 "$MAIN_PY" scripts/ladder_analysis.py --out-name preliminary_qwen3 \
  --models "${QWEN3[@]}" --labeling-sheet 1

todo=( $(not_generated "${QWEN35[@]}") )
[[ ${#todo[@]} -eq 0 ]] || scripts/ladder_run.sh generate "${todo[@]}"
if [[ -n "$(not_generated "$QWEN38")" ]]; then
  if [[ "${QWEN38_STAGE1_ONLY:-0}" == 1 ]]; then
    STAGE1_ONLY=1 KEEP_WEIGHTS=1 scripts/ladder_run.sh generate "$QWEN38"
  else
    KEEP_WEIGHTS=1 scripts/ladder_run.sh generate "$QWEN38"
  fi
fi

stage judge_qwen35_qwen38 "$VLLM_PY" scripts/ladder_judge.py label --models "${QWEN35[@]}" "$QWEN38"
stage validate_thinking "$VLLM_PY" scripts/ladder_judge.py validate --engine thinking \
  --models "${QWEN3[@]}" "${QWEN35[@]}" "$QWEN38"
stage validate_fp8 "$VLLM_PY" scripts/ladder_judge.py validate --engine fp8 \
  --models "${QWEN3[@]}" "${QWEN35[@]}" "$QWEN38"
wait
echo "[$(date -u +%H:%M:%S)] ladder_chain complete"
