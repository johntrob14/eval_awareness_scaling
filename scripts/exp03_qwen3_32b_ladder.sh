#!/bin/bash
# Last step of the exp03 run, after the main chain and the insertion extension: Qwen3-32B's stage 1 + stage 2 (exp02's
# generation code and design; cfg.GENERATED_IN_EXP03), its relabel, its sentence insertion/removal (bases chosen with
# the relabel), then the final analysis over all models. The main chain wrote a "final" before Qwen3-32B had
# labels; it is replaced here.
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
$VLLM exp03_generate.py ladder --models Qwen3-32B >> $LOG/ladder_Qwen3-32B.log 2>&1
step "Qwen3-32B judge: relabel"
$VLLM exp03_judge.py label --models Qwen3-32B >> $LOG/judge_Qwen3-32B.log 2>&1
step "Qwen3-32B interventions"
$VLLM exp03_generate.py interventions --models Qwen3-32B >> $LOG/interventions_Qwen3-32B.log 2>&1
step "Qwen3-32B judge: intervention answers"
$VLLM exp03_judge.py interventions --models Qwen3-32B >> $LOG/judge_Qwen3-32B.log 2>&1
step "final analysis (all ladder models)"
RESULTS=$($PY -c "import exp03_config as c; print(c.RESULTS_DIR / 'final')")
FIGURES=$($PY -c "import exp03_config as c; print(c.figure_dir('final'))")
rm -rf "$RESULTS" "$FIGURES"
$PY exp03_analysis.py --out-name final > $LOG/analysis_final.log 2>&1
step "done"
