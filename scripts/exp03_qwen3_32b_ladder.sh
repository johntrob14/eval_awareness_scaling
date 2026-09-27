#!/bin/bash
# Qwen3-32B on the full ladder (cfg.LADDER_MODELS_ADDED, added after the pre-registration; qwen3_32b_addition.json).
# Runs after the main chain and the insertion extension: stage 1 + stage 2 with exp02's generation code, relabel,
# then an analysis in its own folder. The pre-registered "final" analysis is not touched.
set -euo pipefail
cd "$(dirname "$0")"
VLLM=./vllm_python.sh
PY=/venv/main/bin/python
LOG=../log/exp03
step() { echo "== $* $(date -u +%FT%TZ)"; }

if ! tail -1 $LOG/full_chain.log | grep -q "== done"; then
    echo "main chain did not finish (last line: $(tail -1 $LOG/full_chain.log)); not starting"; exit 1
fi

step "Qwen3-32B stage 1 + stage 2"
$VLLM exp03_generate.py ladder-added --models Qwen3-32B >> $LOG/ladder_added_Qwen3-32B.log 2>&1
step "Qwen3-32B judge: relabel"
$VLLM exp03_judge.py model --models Qwen3-32B >> $LOG/judge_Qwen3-32B.log 2>&1
step "analysis"
$PY exp03_analysis.py --out-name final_with_qwen3_32b --with-added-ladder > $LOG/analysis_final_with_qwen3_32b.log 2>&1
step "done"
