#!/bin/bash
# Sentence insertion/removal on Qwen3.5-0.8B/2B/4B (exp03_config.INTERVENTION_MODELS_ADDED, added after the
# pre-registration). Starts when exp03_full_chain.sh has finished successfully; the analysis is written to its own
# folder, so the pre-registered "final" analysis stays as it was.
set -euo pipefail
cd "$(dirname "$0")"
VLLM=./vllm_python.sh
PY=/venv/main/bin/python
LOG=../log/exp03
step() { echo "== $* $(date -u +%FT%TZ)"; }
download() { $PY -c "import ladder_config as b; s=b.ALL_SUBJECTS['$1']; print(s['repo'], s['revision'])" | \
             { read -r repo rev; /venv/main/bin/hf download "$repo" --revision "$rev" > /dev/null; }; }
repo_dir() { $PY -c "import ladder_config as b; s=b.ALL_SUBJECTS['$1']; print(b.hf_repo_dir(s['repo']))"; }

while pgrep -f exp03_full_chain.sh > /dev/null; do sleep 60; done
if ! tail -1 $LOG/full_chain.log | grep -q "== done"; then
    echo "main chain did not finish (last line: $(tail -1 $LOG/full_chain.log)); not starting"; exit 1
fi

MODELS="Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B"
for m in $MODELS; do
    step "$m download + interventions"
    download $m
    $VLLM exp03_generate.py interventions --models $m > $LOG/interventions_$m.log 2>&1
    step "delete $(repo_dir $m)"
    rm -rf "$(repo_dir $m)"
done
step "judge: intervention answers of the added models"
$VLLM exp03_judge.py interventions --models $MODELS > $LOG/judge_interventions_added.log 2>&1
step "analysis"
$PY exp03_analysis.py --out-name final_with_qwen35_small_insertion > $LOG/analysis_final_with_qwen35_small_insertion.log 2>&1
step "done"
