#!/usr/bin/env bash
# exp02 run order after the Qwen3 ladder generation, chained so the GPU does not wait between steps.
#
#   setsid nohup scripts/ladder_chain.sh <pid of the running Qwen3 ladder_run.sh> > log/exp02/chain.log 2>&1 &
#
# 1. wait for the Qwen3 generation to finish (its weights are deleted by ladder_run.sh)
# 2. Qwen3-32B download starts in the background (disk freed by step 1)
# 3. judge Qwen3; preliminary Qwen3 analysis and labeling sheet 1 (CPU, failure does not stop the chain)
# 4. generate Qwen3.5 ladder, then Qwen3.8-27B (bf16 weights kept for the bf16 judge check)
# 5. judge Qwen3.5 + Qwen3.8; validation subset with the FP8 judge in thinking mode, then the bf16 judge
# 6. delete the Qwen3.8-27B bf16 weights
# The Qwen3-32B add-on is started by hand after checking the projection against T+21.
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

qwen3_pid="$1"
echo "[$(date -u +%H:%M:%S)] waiting for the Qwen3 generation (pid ${qwen3_pid})"
while kill -0 "$qwen3_pid" 2>/dev/null; do sleep 30; done
grep -q "ladder_run generate complete" "${LOG_DIR}/run_qwen3.log" \
  || { echo "Qwen3 generation did not complete; see ${LOG_DIR}/run_qwen3.log" >&2; exit 1; }

qwen3_32b_rev="$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.ADDON_SUBJECTS['Qwen3-32B']['revision'])")"
echo "[$(date -u +%H:%M:%S)] background download Qwen/Qwen3-32B@${qwen3_32b_rev}"
( /venv/main/bin/hf download Qwen/Qwen3-32B --revision "$qwen3_32b_rev" > "${LOG_DIR}/download_qwen3_32b.log" 2>&1 \
  && echo "[$(date -u +%H:%M:%S)] download Qwen3-32B done" || echo "[$(date -u +%H:%M:%S)] download Qwen3-32B FAILED" ) &

stage judge_qwen3 "$VLLM_PY" scripts/ladder_judge.py label --models "${QWEN3[@]}"
soft_stage analysis_preliminary_qwen3 "$MAIN_PY" scripts/ladder_analysis.py --out-name preliminary_qwen3 \
  --models "${QWEN3[@]}" --labeling-sheet 1

scripts/ladder_run.sh generate "${QWEN35[@]}"
KEEP_WEIGHTS=1 scripts/ladder_run.sh generate "$QWEN38"

stage judge_qwen35_qwen38 "$VLLM_PY" scripts/ladder_judge.py label --models "${QWEN35[@]}" "$QWEN38"
stage validate_thinking "$VLLM_PY" scripts/ladder_judge.py validate --engine thinking \
  --models "${QWEN3[@]}" "${QWEN35[@]}" "$QWEN38"
stage validate_bf16 "$VLLM_PY" scripts/ladder_judge.py validate --engine bf16 \
  --models "${QWEN3[@]}" "${QWEN35[@]}" "$QWEN38"

dir="$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.hf_repo_dir(c.SUBJECTS['${QWEN38}']['repo']))")"
echo "[$(date -u +%H:%M:%S)] deleting weights ${dir}"
rm -rf "$dir"
wait
echo "[$(date -u +%H:%M:%S)] ladder_chain complete"
