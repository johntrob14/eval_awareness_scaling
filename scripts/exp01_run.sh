#!/usr/bin/env bash
# Runs exp01 after the manifest exists: build models -> logprob gate -> generate -> score.
# Each stage logs to log/exp01/<stage>.log; the first failure stops everything
# (including a failed logprob gate, which blocks vLLM generation).
#
#   nohup scripts/exp01_run.sh > log/exp01/run.log 2>&1 &
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_PY=/venv/main/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
LOG_DIR="${REPO_DIR}/log/exp01"
cd "${REPO_DIR}/scripts"
mkdir -p "$LOG_DIR"

test -f "${REPO_DIR}/results/exp01_organism_gap/manifest.json" || { echo "manifest.json missing" >&2; exit 1; }

stage() {
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] start ${name}"
  "$@" > "${LOG_DIR}/${name}.log" 2>&1 || { echo "FAILED ${name}; see ${LOG_DIR}/${name}.log" >&2; exit 1; }
  echo "[$(date -u +%H:%M:%S)] done  ${name}"
}

stage build_models        "$MAIN_PY" exp01_build_models.py
stage gate_hf             "$MAIN_PY" exp01_logprob_gate.py hf
stage gate_vllm_organism  "$VLLM_PY" exp01_logprob_gate.py vllm --variant organism
stage gate_vllm_base      "$VLLM_PY" exp01_logprob_gate.py vllm --variant base
stage gate_compare        "$MAIN_PY" exp01_logprob_gate.py compare
stage generate_organism   "$VLLM_PY" exp01_generate_vllm.py --variant organism
stage generate_base       "$VLLM_PY" exp01_generate_vllm.py --variant base
stage score_and_report    "$MAIN_PY" exp01_score_and_report.py
echo "[$(date -u +%H:%M:%S)] exp01 complete"
